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
import json
import requests # Added for HTTP requests
import time # Added for throttling progress updates

from PySide6.QtCore import QThread, Signal, Slot, Qt, QMetaType # Import QMetaType
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget
)
# Removed huggingface_hub and tqdm imports

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# QtTqdm class removed

# --- Worker Thread for Downloads ---


class DownloadWorker(QThread):
    """
    Handles the Hugging Face model download in a separate thread
    to avoid blocking the GUI.
    """
    # Signals for detailed progress - Use int for large file sizes
    initial_files_signal = Signal(list) # Emits the full list of files upfront
    progress_signal = Signal(str, int, int) # filename, current_bytes, total_bytes
    new_file_signal = Signal(str, int) # filename, total_bytes (emitted when byte download starts)
    # file_checked_signal removed
    finished_signal = Signal(str, bool) # message, is_error
    status_update = Signal(str) # Intermediate status messages
    # Signal to mark a file as cached without downloading
    file_cached_signal = Signal(str, int) # filename, total_bytes

    def __init__(self, repo_id, local_dir, token):
        super().__init__()
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.token = token # HF token for authorization
        self._is_running = True # Flag to signal stop

    def _get_repo_files(self):
        """Fetches file list and sizes from Hugging Face Hub API."""
        api_url = f"https://huggingface.co/api/models/{self.repo_id}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            response = requests.get(api_url, headers=headers, timeout=10) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            repo_info = response.json()
            # Extract filenames and sizes from the 'siblings' list
            files = {}
            for file_info in repo_info.get("siblings", []):
                filename = file_info.get("rfilename")
                size = file_info.get("size") # Size might be None for LFS files not downloaded yet? API docs unclear.
                if filename:
                    # Store size, default to -1 if not available (indicates unknown size)
                    files[filename] = size if size is not None else -1
            logging.info(f"Found {len(files)} files in repository via API.")
            return files
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching repo info from API: {e}")
            raise ConnectionError(f"Failed to connect to Hugging Face Hub API: {e}") from e
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding API response: {e}")
            raise ValueError(f"Invalid response from Hugging Face Hub API: {e}") from e
        except Exception as e:
            logging.exception(f"Unexpected error fetching repo files: {e}")
            raise RuntimeError(f"An unexpected error occurred while fetching file list: {e}") from e

    def run(self):
        """Executes the download using requests."""
        if not self._is_running:
            return

        self.status_update.emit(f"Listing files in {self.repo_id} via API...")
        try:
            # --- Get the full list of files and sizes first ---
            repo_files = self._get_repo_files()
            if not self._is_running: return # Check stop flag

            # Emit the list of filenames to populate UI initially
            self.initial_files_signal.emit(list(repo_files.keys()))
            if not self._is_running: return # Check stop flag

        except (ConnectionError, ValueError, RuntimeError) as e:
            logging.error(f"Failed to get file list: {e}")
            if self._is_running:
                self.finished_signal.emit(f"Error listing files: {e}", True)
            return # Cannot proceed

        # --- Proceed with downloading each file ---
        self.status_update.emit(f"Starting download/check of {len(repo_files)} files for {self.repo_id}...")
        headers = {"Accept-Encoding": "identity"} # Try to prevent compression for accurate Content-Length
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        download_count = 0
        error_count = 0
        total_files = len(repo_files)

        for filename, api_size in repo_files.items():
            if not self._is_running:
                logging.info("Download stopped by user.")
                self.status_update.emit("Status: Download cancelled.")
                # Don't emit finished signal here, let the main loop handle cleanup
                return

            file_path = os.path.join(self.local_dir, filename)
            # Ensure subdirectory exists if filename includes path separators
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            except OSError as e:
                logging.error(f"Failed to create directory for {filename}: {e}")
                self.status_update.emit(f"Error: Could not create directory for {filename}. Skipping.")
                error_count += 1
                continue # Skip this file

            # --- Check if file exists and size matches (basic caching) ---
            if os.path.exists(file_path):
                local_size = os.path.getsize(file_path)
                # Use API size for check if available and > 0, otherwise skip size check
                if api_size is not None and api_size > 0 and local_size == api_size:
                    logging.info(f"File '{filename}' already exists and size matches ({api_size} bytes). Skipping download.")
                    self.file_cached_signal.emit(filename, api_size) # Signal UI to mark as cached (remove cast)
                    continue # Skip to next file
                else:
                    logging.info(f"File '{filename}' exists but size mismatch (local: {local_size}, api: {api_size}) or API size unknown. Re-downloading.")
            # --- Download the file ---
            download_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{filename}"
            temp_file_path = file_path + ".part" # Download to temporary file
            resumed_size = 0
            file_mode = 'wb'
            request_headers = headers.copy() # Copy base headers

            # --- Check for existing partial file and attempt resume ---
            if os.path.exists(temp_file_path):
                resumed_size = os.path.getsize(temp_file_path)
                logging.info(f"Partial file '{temp_file_path}' found with size {resumed_size}. Attempting resume.")
                request_headers["Range"] = f"bytes={resumed_size}-"
                file_mode = 'ab' # Append mode
            else:
                logging.info(f"No partial file found for '{filename}'. Starting download from beginning.")

            try:
                logging.info(f"Requesting '{filename}' from {download_url} (Range: {request_headers.get('Range', 'None')})")
                with requests.get(download_url, headers=request_headers, stream=True, timeout=30) as response:

                    # Handle potential resume responses
                    if response.status_code == 416: # Range Not Satisfiable
                        logging.warning(f"Server returned 416 Range Not Satisfiable for {filename}. "
                                        f"This might mean the partial file size ({resumed_size}) matches the total size. "
                                        f"Checking local vs API size.")
                        # Check if the existing partial file is actually complete
                        if api_size is not None and api_size > 0 and resumed_size == api_size:
                            logging.info(f"Partial file '{filename}' is complete ({resumed_size} bytes). Renaming.")
                            os.rename(temp_file_path, file_path)
                            self.file_cached_signal.emit(filename, api_size) # Treat as cached/complete (remove cast)
                            continue # Skip to next file
                        else:
                            logging.warning(f"Partial file size {resumed_size} doesn't match API size {api_size}. "
                                            f"Restarting download from beginning.")
                            resumed_size = 0 # Reset resume state
                            file_mode = 'wb' # Overwrite mode
                            # Re-request without Range header
                            with requests.get(download_url, headers=headers, stream=True, timeout=30) as fresh_response:
                                fresh_response.raise_for_status()
                                response = fresh_response # Use the new response object
                                # Proceed with download logic below...
                    elif response.status_code == 206: # Partial Content (successful resume)
                        logging.info(f"Server accepted resume request for '{filename}' (Status 206).")
                    elif response.status_code == 200: # OK (full download, or server ignored Range)
                        if resumed_size > 0:
                            logging.warning(f"Server returned 200 OK despite Range request for '{filename}'. "
                                            f"Restarting download from beginning.")
                            resumed_size = 0 # Reset resume state
                            file_mode = 'wb' # Overwrite mode
                        else:
                            logging.info(f"Starting full download for '{filename}' (Status 200).")
                    else:
                        # Raise any other non-successful status codes
                        response.raise_for_status()

                    # Get total size from Content-Length/Content-Range header or API
                    content_length_str = response.headers.get('Content-Length')
                    content_range_str = response.headers.get('Content-Range')
                    total_size = -1 # Default to unknown size

                    try:
                        if content_range_str: # Check Content-Range first (present in 206 responses)
                            # Format: "bytes start-end/total"
                            total_str = content_range_str.split('/')[-1]
                            if total_str != '*':
                                total_size = int(total_str)
                            else:
                                logging.warning(f"Content-Range total size is '*' for {filename}.")
                        elif content_length_str: # Fallback to Content-Length (present in 200 responses)
                            # This should be the full size if status is 200
                            total_size = int(content_length_str)

                        # If size still unknown, try API size as last resort
                        if total_size <= 0 and api_size is not None and api_size > 0:
                            total_size = api_size
                            logging.warning(f"Using API size ({api_size}) for {filename} as Content-Length/Range was missing or invalid.")

                    except (ValueError, IndexError) as e:
                        logging.warning(f"Error parsing size headers (Content-Length: '{content_length_str}', Content-Range: '{content_range_str}') for {filename}: {e}")
                        # Try API size if parsing failed
                        if api_size is not None and api_size > 0:
                            total_size = api_size
                            logging.warning(f"Using API size ({api_size}) due to header parsing error.")
                        else:
                            total_size = -1 # Still unknown

                    if total_size == 0: # Handle zero-byte files (check after determining size)
                        logging.info(f"File '{filename}' is zero bytes. Creating empty file.")
                        # Emit signals to show completion immediately (remove casts)
                        self.new_file_signal.emit(filename, 0)
                        if not self._is_running: return
                        pathlib.Path(file_path).touch() # Create empty file directly
                        self.progress_signal.emit(filename, 0, 0)
                        download_count += 1
                        continue # Skip to next file

                    # Emit signal that download is starting (pass total size, remove cast)
                    self.new_file_signal.emit(filename, total_size if total_size > 0 else 0)
                    if not self._is_running: return # Check stop flag again

                    # Initialize for download loop
                    downloaded_since_resume = 0 # Bytes downloaded in this session
                    current_bytes = resumed_size # Total bytes in file so far
                    chunk_size = 1024 * 1024 # Use larger 1MB chunks
                    last_update_time = time.time()
                    bytes_since_last_update = 0
                    update_interval_secs = 0.2 # Update at most every 0.2 seconds
                    update_interval_bytes = 1024 * 1024 # Or every 1MB

                    with open(temp_file_path, file_mode) as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not self._is_running:
                                logging.info(f"Download stopped by user during transfer of {filename}.")
                                # Clean up temporary file
                                try:
                                    os.remove(temp_file_path)
                                except OSError:
                                    pass
                                return # Exit run method

                            if chunk: # filter out keep-alive new chunks
                                chunk_len = len(chunk)
                                f.write(chunk)
                                downloaded_since_resume += chunk_len
                                current_bytes += chunk_len
                                bytes_since_last_update += chunk_len

                                # --- Throttle progress signal emission ---
                                current_time = time.time()
                                time_elapsed = current_time - last_update_time
                                should_update = False
                                if time_elapsed >= update_interval_secs:
                                    should_update = True
                                elif bytes_since_last_update >= update_interval_bytes:
                                    should_update = True

                                # Also update if download is complete (current_bytes matches total_size)
                                if total_size > 0 and current_bytes >= total_size:
                                    should_update = True # Ensure final update is sent

                                if should_update:
                                    # Emit progress (handle unknown total size for display, remove casts)
                                    display_total = total_size if total_size > 0 else current_bytes # Show increasing value if total unknown
                                    self.progress_signal.emit(filename, current_bytes, display_total)
                                    last_update_time = current_time
                                    bytes_since_last_update = 0 # Reset byte counter

                    # --- Final check and rename ---
                    # Check final size against determined total_size (if known)
                    final_size = os.path.getsize(temp_file_path)
                    if total_size > 0 and final_size != total_size:
                        # Size mismatch after download completion
                        raise IOError(f"Final file size ({final_size}) does not match expected size ({total_size}) for {filename}")
                    elif total_size <= 0:
                        # If total size was unknown, update it now based on final size for consistency
                        logging.info(f"Total size for '{filename}' was unknown, setting to final size: {final_size}")
                        total_size = final_size
                        # Optionally emit one last progress signal with the now known total size (remove casts)
                        self.progress_signal.emit(filename, final_size, final_size)

                    # Rename temporary file to final name
                    os.rename(temp_file_path, file_path)
                    logging.info(f"Successfully downloaded and saved '{filename}'")
                    download_count += 1

                    # Ensure final progress signal marks 100% if size was known and wasn't sent in loop
                    # (The check inside the loop should cover this, but double-check)
                    # if total_size > 0 and bytes_since_last_update > 0: # Check if last chunk wasn't reported
                    #    self.progress_signal.emit(filename, total_size, total_size)

            except requests.exceptions.RequestException as e:
                logging.error(f"Download request failed for {filename}: {e}")
                self.status_update.emit(f"Error downloading {filename}: {e}. Skipping.")
                error_count += 1
                # Clean up partial file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass
            except IOError as e:
                logging.error(f"File I/O error for {filename}: {e}")
                self.status_update.emit(f"File error for {filename}: {e}. Skipping.")
                error_count += 1
                # Clean up partial file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass
            except Exception as e:
                logging.exception(f"An unexpected error occurred during download of {filename}")
                self.status_update.emit(f"Unexpected error for {filename}: {e}. Skipping.")
                error_count += 1
                # Clean up partial file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass

        # --- Finished ---
        if self._is_running:
            final_message = f"Download process finished. {download_count}/{total_files} files downloaded."
            if error_count > 0:
                final_message += f" {error_count} files failed."
                self.finished_signal.emit(final_message, True) # Signal error if any file failed
            else:
                self.finished_signal.emit(final_message, False) # Signal success

        self._is_running = False # Mark as not running

    def stop(self):
        """Signals the thread to stop downloading."""
        logging.info("Stop requested for download worker.")
        self._is_running = False
        # The flag will be checked between chunks and between files


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
        # Use provided token if available
        token_to_use = hf_token or os.environ.get("HF_TOKEN") # Check env var as fallback
        if not token_to_use:
            logging.warning("HF Token not provided and HF_TOKEN environment variable not set. "
                            "Accessing gated models may fail.")

        # Start download in a separate thread
        self.download_thread = DownloadWorker(repo_id, local_dir, token_to_use)
        # Connect signals
        self.download_thread.status_update.connect(self.update_status)
        self.download_thread.initial_files_signal.connect(self.populate_initial_progress_bars)
        self.download_thread.new_file_signal.connect(self.add_or_update_progress_bar)
        self.download_thread.file_cached_signal.connect(self.mark_file_as_cached) # Connect cache signal
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

    def _format_size_string(self, total_bytes):
        """Helper to format file sizes consistently."""
        if total_bytes == -1: # Handle unknown size case first
            return " (Unknown size)"
        elif total_bytes <= 0: # Handle zero or negative size next
            return ""
        # Now handle positive sizes
        elif total_bytes < 1024:
            size_str = f"{total_bytes} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes / 1024:.2f} KiB"
        elif total_bytes < 1024**3:
            size_str = f"{total_bytes / (1024**2):.2f} MiB"
        else: # total_bytes >= 1024**3
            size_str = f"{total_bytes / (1024**3):.2f} GiB"

        return f" ({size_str})"

    # --- Slots for Progress Updates ---
    @Slot(list)
    def populate_initial_progress_bars(self, filenames):
        """Creates all progress bars based on the initial file list."""
        logging.info(f"Populating initial progress bars for {len(filenames)} files.") # More visible log
        self.clear_progress_bars() # Clear any previous state
        for filename in sorted(filenames): # Sort for consistent order
            logging.debug(f"Populating: Adding key '{filename}' to self.progress_bars")
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
        # Use a scaled maximum (e.g., 10000) for known sizes to avoid overflow
        # Use 100 for unknown sizes (pending/indeterminate state)
        scaled_max = 10000
        pbar.setMaximum(scaled_max if total_bytes > 0 else 100)
        pbar.setValue(0) # Start at 0
        pbar.setTextVisible(True)
        pbar.setFormat(initial_format)

        # Store label with pbar for later updates
        self.progress_bars[filename] = {'bar': pbar, 'label': label}
        self.progress_area_layout.addWidget(label)
        self.progress_area_layout.addWidget(pbar)
        logging.debug(f"_create_progress_bar_widget: Created bar for '{filename}' with format '{initial_format}'")

    @Slot(str, int) # Use int instead of QMetaType.Type.LongLong
    def add_or_update_progress_bar(self, filename, total_bytes):
        """
        Updates a progress bar when a file download starts (or resumes).
        Called when the new_file_signal is received.
        """
        logging.info(f"Starting download for: {filename} ({total_bytes} bytes)")

        # Extract just the filename part if it contains a path (should match QtTqdm logic)
        display_name = filename.split('/')[-1]

        # Format size string using helper
        size_str = self._format_size_string(total_bytes)

        if filename in self.progress_bars:
            # Update existing progress bar (expected case)
            pbar_info = self.progress_bars[filename]
            pbar = pbar_info['bar']
            label = pbar_info['label']

            # Update label with size
            label.setText(f"{display_name}{size_str}")

            # Update progress bar properties for download start
            scaled_max = 10000
            if total_bytes > 0:
                # Ensure max is set to the scaled maximum
                if pbar.maximum() != scaled_max:
                    pbar.setMaximum(scaled_max)
                pbar.setValue(0) # Reset to start of download
                pbar.setFormat("Downloading: %p%") # Set downloading state
            else: # Handle unknown size or zero-byte files
                # Use 100 for max, display appropriate format
                if pbar.maximum() != 100:
                    pbar.setMaximum(100)
                pbar.setValue(0)
                if total_bytes == 0: # Specifically zero-byte
                    pbar.setFormat("Complete (0 B)")
                    pbar.setValue(100) # Mark as complete visually
                else: # Unknown size (-1 or other)
                    pbar.setFormat("Pending...") # Or "Downloading..." if preferred

            logging.debug(f"add_or_update_progress_bar: Updated bar for '{filename}' to download state.")

        else:
            # Fallback: Create new progress bar if it wasn't created initially
            # This might happen if list_repo_files failed or had timing issues
            logging.warning(f"add_or_update_progress_bar: Progress bar for '{filename}' not found. Creating now.")
            initial_format = "Downloading: %p%" if total_bytes > 0 else ("Complete (0 B)" if total_bytes == 0 else "Pending...")
            self._create_progress_bar_widget(filename, total_bytes, initial_format)
            # Ensure the newly created bar reflects the current state (value 0 or 100 for 0 B)
            if filename in self.progress_bars:
                new_pbar = self.progress_bars[filename]['bar']
                if total_bytes == 0:
                    new_pbar.setValue(new_pbar.maximum()) # Set to max (100)
                else:
                    new_pbar.setValue(0)

    @Slot(str, int) # Use int instead of QMetaType.Type.LongLong
    def mark_file_as_cached(self, filename, total_bytes):
        """Marks a progress bar as 'Cached' when the file already exists."""
        logging.debug(f"Marking '{filename}' as Cached ({total_bytes} bytes) in UI.")
        if filename in self.progress_bars:
            pbar_info = self.progress_bars[filename]
            pbar = pbar_info['bar']
            label = pbar_info['label']

            # Update label with size if not already done
            display_name = filename.split('/')[-1]
            size_str = self._format_size_string(total_bytes)
            label.setText(f"{display_name}{size_str}")

            # Update progress bar state using scaled max
            scaled_max = 10000
            if total_bytes > 0:
                # Set to scaled max and full value
                if pbar.maximum() != scaled_max:
                    pbar.setMaximum(scaled_max)
                pbar.setValue(scaled_max) # Set to 100% of scaled value
            else: # Handle unknown or zero size (should ideally have size if cached, but handle defensively)
                if pbar.maximum() != 100:
                    pbar.setMaximum(100) # Use 100 base for visual completion
                pbar.setValue(100)
            pbar.setFormat("Cached")
        else:
            logging.warning(f"mark_file_as_cached: Progress bar for '{filename}' not found.")
            # Optionally create it here if needed, marked as cached
            self._create_progress_bar_widget(filename, total_bytes, "Cached")
            if filename in self.progress_bars:
                pbar = self.progress_bars[filename]['bar']
                pbar = self.progress_bars[filename]['bar']
                scaled_max = 10000
                if total_bytes > 0:
                    if pbar.maximum() != scaled_max:
                        pbar.setMaximum(scaled_max)
                    pbar.setValue(scaled_max)
                else:
                    if pbar.maximum() != 100:
                        pbar.setMaximum(100)
                    pbar.setValue(100)

    @Slot(str, int, int) # Use int instead of QMetaType.Type.LongLong
    def update_progress_bar(self, filename, current_bytes, total_bytes):
        """Updates the value of a specific progress bar during download."""
        # Values received should be 64-bit integers (LongLong)
        # Cast to Python ints for safety in calculations/comparisons if needed,
        # but qint64 should be handled correctly by QProgressBar.
        # Keep using the qint64 arguments directly for Qt calls.
        # Use int() only for non-Qt logic if necessary.
        # current_bytes_int = int(current_bytes) # Example if needed elsewhere
        total_bytes_int = int(total_bytes) # Keep this for the <= 0 check logic

        if filename in self.progress_bars:
            pbar_info = self.progress_bars[filename]
            pbar = pbar_info['bar']
            scaled_max = 10000

            # Handle unknown total size (total_bytes_int <= 0)
            if total_bytes_int <= 0:
                # If total size is unknown, display bytes downloaded without percentage.
                # Option 1: Indeterminate (pulsing bar) - might be confusing
                # pbar.setRange(0, 0) # Makes it indeterminate
                # Option 2: Show bytes downloaded
                if pbar.maximum() != 100: # Ensure max is 100 for this mode
                    pbar.setMaximum(100)
                pbar.setValue(0)     # Value doesn't represent percentage here
                # Use the received LongLong current_bytes for calculation, cast to int for division
                print(f'Current Bytes: {current_bytes}; Total Bytes: {total_bytes}; Total Bytes Int: {total_bytes_int}')
                size_mib = int(current_bytes) / (1024 * 1024)
                print(f'Size MiB: {size_mib}')
                pbar.setFormat(f"Downloading: {size_mib:.2f} MiB")
            else:
                # Known total size (total_bytes_int > 0) - Use scaling
                # Ensure max is the scaled maximum
                if pbar.maximum() != scaled_max:
                    logging.debug(f"update_progress_bar: Setting max size for '{filename}' to scaled {scaled_max}")
                    pbar.setMaximum(scaled_max)
                    # Ensure range is not indeterminate if size becomes known
                    # pbar.setRange(0, scaled_max) # Alternative

                # Ensure format shows downloading percentage state if it wasn't already
                current_format = pbar.format()
                # Only change format if it's not already showing percentage or complete
                if not current_format.startswith("Downloading: ") and current_format != "Complete":
                    # Check if it's the MiB format or Pending/Cached/Skipped etc.
                    if "%p%" not in current_format:
                        logging.debug(f"update_progress_bar: Changing format for '{filename}' from '{current_format}' to Downloading %.")
                        pbar.setFormat("Downloading: %p%")

                # Calculate and update scaled progress value
                # Use the received LongLong values for accurate calculation before scaling
                # Clamp current_bytes to total_bytes before scaling
                clamped_current = min(current_bytes, total_bytes) # Comparison works with LongLong
                # Calculate scaled value, ensuring total_bytes is not zero
                # Use float division to avoid overflow with large integers
                if total_bytes > 0:
                    try:
                        # Convert to float for division to prevent overflow
                        ratio = float(clamped_current) / float(total_bytes)
                        scaled_value = int(ratio * scaled_max)
                    except (OverflowError, ValueError) as e:
                        # Handle potential overflow errors with extremely large files
                        logging.warning(f"Error calculating progress ratio: {e}. Using alternative calculation.")
                        # Alternative calculation for very large files
                        # Use string-based percentage calculation to avoid overflow
                        percent_complete = 100.0 * clamped_current / total_bytes
                        scaled_value = int((percent_complete / 100.0) * scaled_max)
                else:
                    scaled_value = 0
                pbar.setValue(scaled_value) # Set scaled value

                # Mark as complete if finished (using original LongLong values for check)
                if current_bytes >= total_bytes: # Comparison works with LongLong
                    # Check if format is already Complete to avoid redundant logging/updates
                    if pbar.format() != "Complete":
                        pbar.setFormat("Complete")
                        # Ensure value is exactly maximum on completion
                        pbar.setValue(scaled_max) # Set to scaled max
                        logging.debug(f"update_progress_bar: Download marked complete for '{filename}'")
        else:
            # Handle missing progress bar case (should be less likely now)
            logging.warning(f"update_progress_bar: Progress bar for '{filename}' not found during update. Attempting to create.")
            # Create with appropriate state based on total_bytes
            initial_format = "Downloading: %p%" if total_bytes > 0 else ("Complete (0 B)" if total_bytes == 0 else "Downloading...")
            self._create_progress_bar_widget(filename, total_bytes, initial_format)
            # Try updating again immediately after adding
            if filename in self.progress_bars:
                # Re-call update_progress_bar to set the correct value/format
                self.update_progress_bar(filename, current_bytes, total_bytes)
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
            QMessageBox.warning(self, "Download Finished with Errors", message) # Use warning for partial success
        else:
            QMessageBox.information(self, "Download Complete", message) # Use info for success

        # After download attempt, mark any remaining "Pending" bars as "Skipped" or "Unknown"
        # (They weren't downloaded, cached, or errored specifically)
        logging.debug("Download process finished. Updating status of any remaining pending bars.")
        for filename, pbar_info in self.progress_bars.items():
            pbar = pbar_info['bar']
            if pbar.format() == "Pending...":
                logging.debug(f"Marking '{filename}' as Skipped post-download.")
                # Keep value at 0 or set indeterminate state? Set to Skipped format.
                pbar.setValue(0)
                pbar.setFormat("Skipped") # Indicate it wasn't processed

        # No need to update status label again here, message box is shown
        if not is_error:
            # Optionally trigger next step only on full success
            pass
        # Button re-enabled in cleanup (happens via download_thread_cleanup)

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
