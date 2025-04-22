#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GGUF Quantization Analysis Tool

GUI application to analyze quantization impact and generate optimized
quantization strategies for GGUF models.
"""

import logging
import os
import pathlib
import sys

# Import tqdm for type hinting and base class if needed
try:
    # Use standard tqdm for type hints if needed, but inherit from asyncio
    from tqdm.auto import tqdm as tqdm_auto
    from tqdm.asyncio import tqdm as tqdm_asyncio
except ImportError:
    logging.warning("tqdm library not found. Progress bars will not be available.")
    tqdm_auto = None
    tqdm_asyncio = None # Ensure this is None if import fails

from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, # Added QProgressBar
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget
)
from huggingface_hub import snapshot_download, list_repo_files # Added list_repo_files
from huggingface_hub.utils import HfFolder, HfHubHTTPError

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Custom TQDM class for Qt Integration ---
# Inherit from tqdm_asyncio if available
class QtTqdm(tqdm_asyncio if tqdm_asyncio else object):
    """
    A tqdm-like class using tqdm.asyncio as a base, emitting Qt signals for progress updates
    instead of console output. Designed for huggingface_hub's snapshot_download tqdm_class.
    """
    # Class attributes to hold signals temporarily during download
    new_file_signal_cls = None
    progress_signal_cls = None
    file_checked_signal_cls = None # Signal for files processed in list mode
    # _lock is inherited from tqdm base class

    def __init__(self, iterable=None, *args, **kwargs): # Accept iterable explicitly
        logging.debug(f"QtTqdm.__init__ called. iterable={type(iterable)}, args={args}, kwargs={kwargs}")
        # Store signals from class attributes BEFORE calling super().__init__
        self.new_file_signal = QtTqdm.new_file_signal_cls
        self.progress_signal = QtTqdm.progress_signal_cls
        self.file_checked_signal = QtTqdm.file_checked_signal_cls

        # Detect mode: file list ('it' unit) or byte download ('B' unit)
        unit = kwargs.get('unit', 'B').lower() # Default to 'B' if unit not specified
        self.is_file_list_mode = unit != 'b'
        self._iterable_internal = iterable # Store iterable for __iter__ override
        logging.debug(f"QtTqdm.__init__: Detected mode: {'File List' if self.is_file_list_mode else 'Byte Download'} (unit='{unit}')")

        # Extract filename from description kwarg - primarily for byte download mode
        desc = kwargs.get('desc', '')
        # In file list mode, desc might be generic like "Fetching..."
        # In byte mode, it should contain the filename
        self.filename = desc.split(':')[0].strip() if ':' in desc else desc
        if self.is_file_list_mode and self.filename == desc:
             # Avoid using generic desc like "Fetching..." as filename
             self.filename = None
        logging.debug(f"QtTqdm.__init__: Initial filename='{self.filename}' from desc='{desc}'")

        # Call parent initializer (tqdm_asyncio)
        # Pass the original iterable back if it was provided
        super().__init__(iterable=iterable, *args, **kwargs, disable=False) # Pass along all args/kwargs

        # Emit the new_file_signal only in byte download mode AFTER super init
        # This signal now primarily updates the total size and marks as downloading
        if not self.is_file_list_mode and self.filename and self.total is not None and self.total > 0:
            if self.new_file_signal:
                logging.debug(f"QtTqdm.__init__ [Byte Mode]: Emitting new_file_signal for '{self.filename}', total={self.total}")
                self.new_file_signal.emit(self.filename, self.total)
            else:
                logging.error("QtTqdm.__init__ [Byte Mode]: new_file_signal_cls was not set!")
        elif not self.is_file_list_mode:
             logging.warning(f"QtTqdm.__init__ [Byte Mode]: Could not emit new_file_signal. filename='{self.filename}', total={self.total}")

    def __iter__(self):
        """
        Override iterator to emit file_checked_signal in file list mode.
        """
        if self.is_file_list_mode and self._iterable_internal is not None:
            logging.debug(f"QtTqdm.__iter__ [File List Mode]: Starting iteration.")
            count = 0
            for item in self._iterable_internal:
                # Assume item is the filename string in this mode
                filename = str(item)
                logging.debug(f"QtTqdm.__iter__ [File List Mode]: Yielding item '{filename}'")
                yield item
                # Emit signal AFTER yielding, indicating processing/checking is done
                if self.file_checked_signal:
                    logging.debug(f"QtTqdm.__iter__ [File List Mode]: Emitting file_checked_signal for '{filename}'")
                    self.file_checked_signal.emit(filename)
                else:
                    logging.error("QtTqdm.__iter__ [File List Mode]: file_checked_signal_cls was not set!")
                count += 1
            logging.debug(f"QtTqdm.__iter__ [File List Mode]: Finished iteration. Items processed: {count}")
        else:
            # Default behavior for byte download mode or if no iterable
            logging.debug(f"QtTqdm.__iter__ [Byte Mode or No Iterable]: Delegating to super().__iter__")
            # Need to handle the case where super() doesn't have __iter__ if tqdm_asyncio is None
            if tqdm_asyncio and hasattr(super(), '__iter__'):
                 yield from super().__iter__()
            elif self._iterable_internal is not None: # Fallback if no parent iter
                 yield from self._iterable_internal


    def display(self, msg=None, pos=None):
        """
        Overrides the default tqdm display method.
        In byte download mode, emits a Qt signal. In file list mode, does nothing.
        """
        # Only emit progress signals in byte download mode
        if not self.is_file_list_mode:
            # self.n is the current progress count, self.total is the total count.
            if self.filename and self.total is not None and self.total > 0 and self.progress_signal:
                # Clamp current value to total to avoid exceeding 100% visually
                current_clamped = min(self.n, self.total)
                logging.debug(f"QtTqdm.display [Byte Mode]: Emitting progress_signal for '{self.filename}': {current_clamped}/{self.total}")
                self.progress_signal.emit(self.filename, current_clamped, self.total)
            elif not self.progress_signal:
                logging.error("QtTqdm.display [Byte Mode]: progress_signal_cls was not set!")
            # Else: No filename or zero total, cannot emit progress
        else:
             # Optional: Log display calls in file list mode if needed for debugging
             # logging.debug(f"QtTqdm.display [File List Mode]: Called (n={self.n}, total={self.total}). No signal emitted.")
             pass

        # Do NOT call super().display() or write anything to console

    def close(self):
        """
        Overrides the default tqdm close method.
        Ensures the progress bar reaches 100% in byte mode and calls the parent close.
        """
        logging.debug(f"QtTqdm.close() called. Mode: {'File List' if self.is_file_list_mode else 'Byte Download'}, Filename: '{self.filename}', n={self.n}, total={self.total}")
        # Ensure the progress bar reaches 100% on close only in byte download mode
        if not self.is_file_list_mode:
            if self.filename and self.total is not None and self.total > 0 and self.n < self.total:
                if self.progress_signal:
                    logging.debug(f"QtTqdm.close [Byte Mode]: Emitting final (100%) progress_signal for '{self.filename}'")
                    self.progress_signal.emit(self.filename, self.total, self.total)
                else:
                    logging.error("QtTqdm.close [Byte Mode]: progress_signal_cls was not set!")
            # Else: No filename, zero total, or already at 100%

        # Call the parent class's close method to perform its cleanup
        if tqdm_asyncio: # Check if parent class exists
            super().close()
        logging.debug(f"QtTqdm: Closed progress for {self.filename}")

    # No need to override update, __iter__, __enter__, __exit__, refresh, set_lock, get_lock
    # The parent tqdm class handles these. Our hook is overriding display().
    # We also don't need set_description override if we extract filename once in init.


# --- Worker Thread for Downloads ---
class DownloadWorker(QThread):
    """
    Handles the Hugging Face model download in a separate thread
    to avoid blocking the GUI.
    """
    # Signals for detailed progress
    initial_files_signal = Signal(list) # Emits the full list of files upfront
    progress_signal = Signal(str, int, int) # filename, current_bytes, total_bytes
    new_file_signal = Signal(str, int) # filename, total_bytes (emitted when byte download starts)
    file_checked_signal = Signal(str) # filename (emitted when file is checked/processed in list mode)
    finished_signal = Signal(str, bool) # message, is_error
    status_update = Signal(str) # Intermediate status messages

    def __init__(self, repo_id, local_dir, token):
        super().__init__()
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.token = token
        self._is_running = True # Use a flag to signal stop if needed

    def run(self):
        """Executes the download."""
        if not self._is_running:
            return

        self.status_update.emit(f"Listing files in {self.repo_id}...")
        try:
            # --- Get the full list of files first ---
            logging.info(f"Calling list_repo_files for {self.repo_id}")
            all_files = list_repo_files(repo_id=self.repo_id, token=self.token)
            logging.info(f"Found {len(all_files)} files in repository.")
            if self._is_running:
                self.initial_files_signal.emit(all_files) # Emit the list to populate UI
            else:
                return # Stop requested before download

        except HfHubHTTPError as e:
            logging.error(f"HTTP Error listing files: {e}")
            if self._is_running:
                error_msg = f"Error listing files for {self.repo_id}: {e}. Check Repo ID/Token/Network."
                self.finished_signal.emit(error_msg, True)
            return # Cannot proceed without file list
        except Exception as e:
            logging.exception(f"An unexpected error occurred listing files for {self.repo_id}")
            if self._is_running:
                self.finished_signal.emit(f"Error listing files: {e}", True)
            return # Cannot proceed

        # --- Proceed with snapshot_download ---
        self.status_update.emit(f"Starting download/check of {len(all_files)} files for {self.repo_id}...")
        # Set signals as class attributes on QtTqdm before download
        QtTqdm.new_file_signal_cls = self.new_file_signal
        QtTqdm.progress_signal_cls = self.progress_signal
        QtTqdm.file_checked_signal_cls = self.file_checked_signal # Set the new signal
        try:
            logging.info(f"Starting snapshot_download for {self.repo_id} to {self.local_dir}")
            # Check if tqdm and our class are available
            effective_tqdm_class = QtTqdm if tqdm_asyncio else None
            logging.info(f"Using tqdm_class: {effective_tqdm_class}")
            path = snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.local_dir,
                local_dir_use_symlinks=False, # Avoid symlinks for simplicity across OS
                token=self.token,
                # Example: ignore pytorch_model.bin if safetensors exist
                # ignore_patterns=["*.bin"], # Add more patterns as needed
                resume_download=True,
                tqdm_class=effective_tqdm_class, # Pass the QtTqdm class directly
                # Consider adding user_agent
            )
            logging.info(f"Download finished. Model saved to: {path}")
            if self._is_running:
                self.finished_signal.emit(f"Download complete. Model path: {path}", False)
        except HfHubHTTPError as e:
            logging.error(f"HTTP Error during download: {e}")
            if self._is_running:
                error_msg = (f"Error downloading {self.repo_id}: {e}. "
                             f"Check Repo ID and network. Gated model? (HF_TOKEN)")
                self.finished_signal.emit(error_msg, True)
        except Exception as e:
            logging.exception(f"An unexpected error occurred during download of {self.repo_id}")
            if self._is_running:
                self.finished_signal.emit(f"Error: {e}", True)
        finally:
            # --- IMPORTANT: Clean up class attributes ---
            QtTqdm.new_file_signal_cls = None
            QtTqdm.progress_signal_cls = None
            QtTqdm.file_checked_signal_cls = None # Clear the new signal
            self._is_running = False

    def stop(self):
        """Signals the thread to stop."""
        logging.info("Stop requested for download worker.")
        self._is_running = False
        # Note: snapshot_download might not be interruptible mid-transfer easily.
        # This flag mainly prevents emitting signals after stop is called.


# --- Main Application Window ---
class MainWindow(QMainWindow):
    """Main window for the GGUF Quantization Analysis Tool."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GGUF Quantization Analysis Tool")
        self.setGeometry(100, 100, 800, 600) # x, y, width, height

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- Input Area ---
        input_group = QGroupBox("Model Download")
        input_layout = QFormLayout()
        input_group.setLayout(input_layout)

        self.repo_id_input = QLineEdit()
        self.repo_id_input.setPlaceholderText("e.g., meta-llama/Llama-3.1-8B-Instruct")
        input_layout.addRow("Hugging Face Repo ID:", self.repo_id_input)

        self.hf_token_input = QLineEdit()
        self.hf_token_input.setPlaceholderText("Optional: Your Hugging Face token (for gated models)")
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password) # Hide token
        input_layout.addRow("Hugging Face Token:", self.hf_token_input)

        self.local_dir_layout = QHBoxLayout()
        self.local_dir_input = QLineEdit()
        self.local_dir_input.setPlaceholderText("Select directory to save model files")
        # Set a default path if desired, e.g., user's home + /models
        default_path = os.path.join(pathlib.Path.home(), "quant_analysis_models")
        self.local_dir_input.setText(default_path)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_directory)
        self.local_dir_layout.addWidget(self.local_dir_input)
        self.local_dir_layout.addWidget(self.browse_button)
        input_layout.addRow("Download Directory:", self.local_dir_layout)

        self.download_button = QPushButton("Download Model")
        self.download_button.clicked.connect(self.start_download)
        input_layout.addRow(self.download_button) # Add button to form layout

        self.main_layout.addWidget(input_group)

        # --- Progress Area ---
        self.progress_group = QGroupBox("Download Progress")
        # Layout inside the group box to hold progress bars
        self.progress_area_layout = QVBoxLayout()
        self.progress_area_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Add bars from top
        self.progress_group.setLayout(self.progress_area_layout)

        # ScrollArea setup
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.progress_group)
        # Make scroll area expand vertically, but keep group box its natural height initially
        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.progress_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.main_layout.addWidget(scroll_area) # Add scroll area to main layout

        # --- Status Bar ---
        self.status_label = QLabel("Status: Ready")
        self.main_layout.addWidget(self.status_label)
        self.main_layout.setStretchFactor(scroll_area, 1) # Make scroll area take available space

        self.download_thread = None
        # Dictionary holds progress bar and its label {filename: {'bar': QProgressBar, 'label': QLabel}}
        self.progress_bars = {}

    @Slot()
    def browse_directory(self):
        """Opens a dialog to select the download directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Download Directory",
            self.local_dir_input.text() # Start browsing from current path
        )
        if directory:
            self.local_dir_input.setText(directory)

    @Slot()
    def start_download(self):
        """Validates inputs and starts the download worker thread."""
        if self.download_thread and self.download_thread.isRunning():
            logging.warning("Download already in progress.")
            # Optionally offer a cancel button here
            return

        repo_id = self.repo_id_input.text().strip()
        local_dir = self.local_dir_input.text().strip()

        # --- Input Validation ---
        if not repo_id:
            QMessageBox.warning(self, "Input Error", "Please enter a Hugging Face Repo ID.")
            return
        if not local_dir:
            QMessageBox.warning(self, "Input Error", "Please select a download directory.")
            return

        # Ensure the directory exists, create if not
        try:
            pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Directory Error", f"Failed to create directory {local_dir}:\n{e}")
            return
        if not pathlib.Path(local_dir).is_dir():
            QMessageBox.critical(self, "Directory Error", f"Selected path {local_dir} is not a valid directory.")
            return

        hf_token = self.hf_token_input.text().strip() or None # Use None if empty

        # --- Start Download ---
        self.clear_progress_bars()
        self.update_status(f"Status: Preparing download for {repo_id}...")
        self.download_button.setEnabled(False) # Disable button during download

        # Get token (optional, for gated models)
        # Use provided token first, otherwise try to get from environment/login
        token_to_use = hf_token or HfFolder.get_token()
        if not token_to_use and not hf_token: # Only warn if no token was provided *and* none found automatically
            logging.warning("HF Token not found. Accessing gated models may fail. "
                            "Set HF_TOKEN environment variable or login via `huggingface-cli login`.")

        # Start download in a separate thread
        self.download_thread = DownloadWorker(repo_id, local_dir, token_to_use)
        # Connect signals
        self.download_thread.status_update.connect(self.update_status)
        self.download_thread.initial_files_signal.connect(self.populate_initial_progress_bars) # New signal
        self.download_thread.new_file_signal.connect(self.add_or_update_progress_bar)
        self.download_thread.file_checked_signal.connect(self.mark_file_as_checked)
        self.download_thread.progress_signal.connect(self.update_progress_bar)
        self.download_thread.finished_signal.connect(self.download_finished)
        self.download_thread.finished.connect(self.download_thread_cleanup) # Clean up thread object
        self.download_thread.start()

    def clear_progress_bars(self):
        """Removes all widgets from the progress area layout."""
        while self.progress_area_layout.count():
            child = self.progress_area_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater() # Ensure proper widget deletion
        # Ensure proper widget deletion when clearing layout
        for pbar_info in self.progress_bars.values():
            pbar_info['bar'].deleteLater()
            pbar_info['label'].deleteLater()
        # Clear layout (alternative way)
        # while self.progress_area_layout.count():
        #     child = self.progress_area_layout.takeAt(0)
        #     if child.widget():
        #         child.widget().deleteLater()
        self.progress_bars.clear()


    # --- Slots for Progress Updates ---
    @Slot(list)
    def populate_initial_progress_bars(self, filenames):
        """Creates all progress bars based on the initial file list."""
        logging.debug(f"Populating initial progress bars for {len(filenames)} files.")
        self.clear_progress_bars() # Clear any previous state
        for filename in sorted(filenames): # Sort for consistent order
            if filename not in self.progress_bars:
                 self._create_progress_bar_widget(filename, 0, "Pending...")
            else:
                 # This case should ideally not happen if clear_progress_bars worked
                 logging.warning(f"populate_initial_progress_bars: Bar for '{filename}' already exists.")

    def _create_progress_bar_widget(self, filename, total_bytes, initial_format):
        """Helper to create and store label/pbar widgets."""
        size_str = ""
        if total_bytes > 0:
            # Format size for label only if known
            if total_bytes < 1024:
                size_str = f"{total_bytes} B"
            elif total_bytes < 1024**2:
                size_str = f"{total_bytes / 1024:.2f} KiB"
            elif total_bytes < 1024**3:
                size_str = f"{total_bytes / (1024**2):.2f} MiB"
            else:
                size_str = f"{total_bytes / (1024**3):.2f} GiB"
            size_str = f" ({size_str})" # Add parentheses

        label = QLabel(f"{filename}{size_str}")
        pbar = QProgressBar()
        pbar.setMinimum(0)
        # Set max to total_bytes if known, otherwise use 100 for percentage/pending state
        pbar.setMaximum(total_bytes if total_bytes > 0 else 100)
        pbar.setValue(0) # Start at 0
        pbar.setTextVisible(True)
        pbar.setFormat(initial_format)

        # Store label with pbar for later updates
        self.progress_bars[filename] = {'bar': pbar, 'label': label}
        self.progress_area_layout.addWidget(label)
        self.progress_area_layout.addWidget(pbar)
        logging.debug(f"_create_progress_bar_widget: Created bar for '{filename}' with format '{initial_format}'")


    @Slot(str, int)
    def add_or_update_progress_bar(self, filename, total_bytes):
        """
        Adds a progress bar if it doesn't exist, or updates the total size
        and state if it does. Called when a byte download *starts*.
        """
        size_str = ""
        if total_bytes > 0:
            # Format size for label only if known
            if total_bytes < 1024:
                size_str = f"{total_bytes} B"
            elif total_bytes < 1024**2:
                size_str = f"{total_bytes / 1024:.2f} KiB"
            elif total_bytes < 1024**3:
                size_str = f"{total_bytes / (1024**2):.2f} MiB"
            elif total_bytes < 1024**3:
                size_str = f"{total_bytes / (1024**2):.2f} MiB"
            else:
                size_str = f"{total_bytes / (1024**3):.2f} GiB"
            size_str = f" ({size_str})" # Add parentheses

        if filename not in self.progress_bars:
            # Fallback: Should have been created by populate_initial_progress_bars
            logging.warning(f"add_or_update_progress_bar: Bar for '{filename}' not found, creating now.")
            self._create_progress_bar_widget(filename, total_bytes, "Downloading: %p%" if total_bytes > 0 else "Pending...")
            # No need to update further if just created
            return

        # Bar exists, update it for download start
        logging.debug(f"add_or_update_progress_bar: Updating existing bar for '{filename}' to download state.")
        pbar_info = self.progress_bars[filename]
        pbar = pbar_info['bar']
        label = pbar_info['label']

        label.setText(f"{filename}{size_str}") # Update label text with size
        if total_bytes > 0:
             # Update max only if it was placeholder (100) or different
             if pbar.maximum() <= 100 or pbar.maximum() != total_bytes:
                 pbar.setMaximum(total_bytes)
        else:
             # If total_bytes is 0 here, something is odd, keep max=100
             pbar.setMaximum(100)

        pbar.setValue(0) # Reset value to 0 for download start
        pbar.setFormat("Downloading: %p%") # Set format for active download


    @Slot(str)
    def mark_file_as_checked(self, filename):
        """Marks a file's progress bar as 100% (cached/checked)."""
        if filename in self.progress_bars:
            pbar_info = self.progress_bars[filename]
            pbar = pbar_info['bar']
            # Only update if it's still pending, don't overwrite Downloading/Complete
            if pbar.format() == "Pending...":
                logging.debug(f"mark_file_as_checked: Marking '{filename}' as 100% / Cached")
                # Set value to max (should be 100 for pending bars)
                pbar.setValue(pbar.maximum())
                pbar.setFormat("Cached") # Update format
            else:
                logging.debug(f"mark_file_as_checked: Skipping update for '{filename}', state is '{pbar.format()}'")
        else:
             # Should have been created by populate_initial_progress_bars
             logging.error(f"mark_file_as_checked: Progress bar for '{filename}' not found.")


    @Slot(str, int, int)
    def update_progress_bar(self, filename, current_bytes, total_bytes):
        """Updates the value of a specific progress bar during byte download."""
        if filename in self.progress_bars:
            pbar_info = self.progress_bars[filename]
            pbar = pbar_info['bar']
            # Ensure max is correct (might be updated by add_or_update_progress_bar)
            if pbar.maximum() != total_bytes and total_bytes > 0:
                 logging.warning(f"update_progress_bar: Correcting max size for '{filename}' to {total_bytes}")
                 pbar.setMaximum(total_bytes)

            # Ensure format is correct for downloading state
            if not pbar.format().startswith("Downloading"):
                 pbar.setFormat("Downloading: %p%")

            pbar.setValue(current_bytes)

            # Check if download just completed
            if current_bytes == total_bytes and total_bytes > 0:
                 pbar.setFormat("Complete")
                 logging.debug(f"update_progress_bar: Download complete for '{filename}'")

        else:
            # This might happen with unfortunate signal timing. Create it if missing.
            logging.warning(f"update_progress_bar: Progress bar for '{filename}' not found. Attempting to create.")
            # Create with downloading state directly
            self._create_progress_bar_widget(filename, total_bytes, "Downloading: %p%")
            # Try updating again immediately after adding
            if filename in self.progress_bars:
                 pbar = self.progress_bars[filename]['bar']
                 if pbar.maximum() != total_bytes and total_bytes > 0:
                      pbar.setMaximum(total_bytes)
                 pbar.setValue(current_bytes)
                 if current_bytes == total_bytes and total_bytes > 0:
                      pbar.setFormat("Complete")
            else:
                 # If creation failed, log error
                 logging.error(f"update_progress_bar: Failed to create progress bar for '{filename}' during update.")

    @Slot(str)
    def update_status(self, message):
        """Updates the status label."""
        self.status_label.setText(message)
        logging.info(message) # Also log status updates

    @Slot(str, bool)
    def download_finished(self, message, is_error):
        """Handles the completion or error of the download thread."""
        self.update_status(f"Status: {message}")
        if is_error:
            QMessageBox.critical(self, "Download Error", message)
        else:
            # Mark any remaining pending files as cached (they weren't downloaded)
            logging.debug("Download finished. Marking remaining pending bars as Cached.")
            for filename, pbar_info in self.progress_bars.items():
                if pbar_info['bar'].format() == "Pending...":
                    logging.debug(f"Marking '{filename}' as Cached post-download.")
                    pbar_info['bar'].setValue(pbar_info['bar'].maximum())
                    pbar_info['bar'].setFormat("Cached")
            # Optionally trigger next step here
            pass
        # Button re-enabled in cleanup

    @Slot()
    def download_thread_cleanup(self):
        """Cleans up the thread reference after it finishes."""
        logging.debug("Download thread finished signal received.")
        self.download_button.setEnabled(True) # Re-enable button
        self.download_thread = None # Clear thread reference

    def closeEvent(self, event):
        """Handles the window closing event."""
        if self.download_thread and self.download_thread.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit',
                                         "A download is in progress. Are you sure you want to exit? "
                                         "The current download may not be properly stopped.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                logging.info("Attempting to stop download thread on close...")
                self.download_thread.stop()
                # Give it a brief moment, though it might not stop snapshot_download
                if not self.download_thread.wait(500):
                    logging.warning("Download thread did not stop gracefully.")
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply styles or themes here if desired
    # app.setStyle(...)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
