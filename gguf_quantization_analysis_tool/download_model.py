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
import threading # Added for periodic save timer

from PySide6.QtCore import QThread, Signal, Slot, Qt # Import QMetaType
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class DownloadWorker(QThread):
    """
    Handles the Hugging Face model download in a separate thread
    to avoid blocking the GUI.
    """
    initial_files_signal = Signal(list) # Emits the full list of files upfront
    progress_signal = Signal(str, int, int) # filename, percentage (0-100), size_in_mib
    new_file_signal = Signal(str, int) # filename, size_in_mib
    finished_signal = Signal(str, bool) # message, is_error
    status_update = Signal(str) # Intermediate status messages
    # Signal to mark a file as cached without downloading
    file_cached_signal = Signal(str, int) # filename, size_in_mib
    # Signal to update UI with discovered subdirectories
    subdirs_discovered_signal = Signal(list) # list of subdirectory names

    def __init__(self, repo_id, local_dir, token, subdir_filter=None):
        super().__init__()
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.token = token # HF token for authorization
        self.subdir_filter = subdir_filter # Optional subdirectory filter
        self._is_running = True # Flag to signal stop
        self._current_file = None # Reference to current file being downloaded
        self._save_timer = None # Timer for periodic saves
        self._save_interval = 10 # Save every 10 seconds
        self._subdirectories = set() # Store discovered subdirectories

    def _get_repo_files(self):
        """Fetches file list and sizes from Hugging Face Hub API."""
        api_url = f"https://huggingface.co/api/models/{self.repo_id}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            repo_info = response.json()
            # Extract filenames and sizes from the 'siblings' list
            files = {}
            self._subdirectories = set()

            for file_info in repo_info.get("siblings", []):
                filename = file_info.get("rfilename")
                size = file_info.get("size") # Size might be None for LFS files not downloaded yet? API docs unclear.

                if filename:
                    # Extract subdirectory if present
                    if '/' in filename:
                        subdir = filename.split('/')[0]
                        self._subdirectories.add(subdir)

                    # Apply subdirectory filter if specified
                    if self.subdir_filter and not filename.startswith(f"{self.subdir_filter}/"):
                        # Skip files not in the specified subdirectory
                        continue

                    # Store size, default to -1 if not available (indicates unknown size)
                    files[filename] = size if size is not None else -1

            # Emit signal with discovered subdirectories
            self.status_update.emit(f"Found {len(self._subdirectories)} subdirectories in repository")

            logging.info(f"Found {len(files)} files in repository via API"
                         + (f" (filtered to subdirectory '{self.subdir_filter}')" if self.subdir_filter else ""))
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

            # Emit the list of subdirectories discovered
            self.subdirs_discovered_signal.emit(sorted(list(self._subdirectories)))

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
                return

            file_path = os.path.join(self.local_dir, filename)
            # Ensure subdirectory exists if filename includes path separators
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            except OSError as e:
                logging.error(f"Failed to create directory for {filename}: {e}")
                self.status_update.emit(f"Error: Could not create directory for {filename}. Skipping.")
                error_count += 1
                continue

            # --- Check if file exists and size matches (basic caching) ---
            if os.path.exists(file_path):
                local_size = os.path.getsize(file_path)
                # If API size is -1 (unknown), assume the local file is good
                if api_size == -1:
                    logging.info(f"File '{filename}' already exists and API size is unknown. Assuming file is good.")
                    # Use local size for display since API size is unknown
                    size_mib = int(local_size / (1024 * 1024))
                    self.file_cached_signal.emit(filename, size_mib) # Signal UI to mark as cached
                    continue # Skip to next file
                # Otherwise, use API size for check if available and > 0
                elif api_size is not None and api_size > 0 and local_size == api_size:
                    logging.info(f"File '{filename}' already exists and size matches ({api_size} bytes). Skipping download.")
                    # Convert API size to MiB for display
                    size_mib = int(api_size / (1024 * 1024))
                    self.file_cached_signal.emit(filename, size_mib) # Signal UI to mark as cached
                    continue # Skip to next file
                else:
                    logging.info(f"File '{filename}' exists but size mismatch (local: {local_size}, api: {api_size}). Re-downloading.")
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

                # Check if the partial file might already be complete by comparing with API size
                if api_size is not None and api_size > 0 and resumed_size == api_size:
                    logging.info(f"Partial file '{filename}' appears to be complete ({resumed_size} bytes). Renaming to final filename.")
                    os.rename(temp_file_path, file_path)
                    # Convert API size to MiB for display
                    size_mib = int(api_size / (1024 * 1024)) if api_size > 0 else 0
                    self.file_cached_signal.emit(filename, size_mib) # Treat as cached/complete
                    continue # Skip to next file

                # Otherwise, attempt to resume the download
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
                            # Convert API size to MiB for display
                            size_mib = int(api_size / (1024 * 1024)) if api_size > 0 else 0
                            self.file_cached_signal.emit(filename, size_mib) # Treat as cached/complete
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
                        logging.warning(f"Error parsing size headers for {filename}: {e}")
                        # Try API size if parsing failed
                        if api_size is not None and api_size > 0:
                            total_size = api_size
                            logging.warning(f"Using API size ({api_size}) due to header parsing error.")

                    if total_size == 0: # Handle zero-byte files (check after determining size)
                        logging.info(f"File '{filename}' is zero bytes. Creating empty file.")
                        # Emit signals to show completion immediately (remove casts)
                        self.new_file_signal.emit(filename, 0)
                        if not self._is_running: return
                        pathlib.Path(file_path).touch() # Create empty file directly
                        self.progress_signal.emit(filename, 0, 0)
                        download_count += 1
                        continue # Skip to next file

                    # Convert total size to MiB for display before emitting
                    size_mib = int(total_size / (1024 * 1024)) if total_size > 0 else 0
                    # Emit signal that download is starting with size in MiB
                    self.new_file_signal.emit(filename, size_mib)
                    if not self._is_running: return # Check stop flag again

                    # Initialize for download loop
                    downloaded_since_resume = 0 # Bytes downloaded in this session
                    current_bytes = resumed_size # Total bytes in file so far
                    chunk_size = 1024 * 1024 # Use larger 1MB chunks
                    last_update_time = time.time()
                    bytes_since_last_update = 0
                    update_interval_secs = 0.2 # Update at most every 0.2 seconds
                    update_interval_bytes = 1024 * 1024 # Or every 1MB

                    # Define periodic save function outside the with block
                    def periodic_save():
                        if self._current_file and not self._current_file.closed:
                            try:
                                logging.debug(f"Performing periodic flush for {filename}")
                                self._current_file.flush()
                                os.fsync(self._current_file.fileno())
                            except (IOError, OSError) as e:
                                logging.warning(f"Error during periodic flush: {e}")

                            # Schedule next save if still running
                            if self._is_running:
                                self._save_timer = threading.Timer(self._save_interval, periodic_save)
                                self._save_timer.daemon = True
                                self._save_timer.start()

                    with open(temp_file_path, file_mode) as f:
                        self._current_file = f

                        # Start the periodic save timer
                        self._save_timer = threading.Timer(self._save_interval, periodic_save)
                        self._save_timer.daemon = True
                        self._save_timer.start()

                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not self._is_running:
                                logging.info(f"Download stopped by user during transfer of {filename}.")
                                # Cancel the timer
                                if self._save_timer:
                                    self._save_timer.cancel()
                                    self._save_timer = None
                                # Flush the file to ensure partial data is saved for resume
                                f.flush()
                                os.fsync(f.fileno())
                                logging.info(f"Partial file {temp_file_path} preserved for future resume")
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
                                should_update = (time_elapsed >= update_interval_secs or bytes_since_last_update >= update_interval_bytes or (total_size > 0 and current_bytes >= total_size))

                                if should_update:
                                    # Calculate percentage and MiB for display
                                    percentage = min(100, int((current_bytes * 100) / total_size)) if total_size > 0 else 0
                                    size_mib = int(current_bytes / (1024 * 1024))

                                    # Emit progress with percentage and MiB values
                                    self.progress_signal.emit(filename, percentage, size_mib)
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
                        # Calculate final percentage (100%) and MiB for display
                        size_mib = int(final_size / (1024 * 1024))
                        # Emit final progress signal with 100% and final size in MiB
                        self.progress_signal.emit(filename, 100, size_mib)

                    # Check if the final file already exists with the same size
                    if os.path.exists(file_path) and os.path.getsize(file_path) == final_size:
                        logging.info(f"Final file '{filename}' already exists with matching size. Removing temporary file.")
                        os.remove(temp_file_path)  # Remove the temporary file
                        # Emit final progress signal
                        size_mib = int(final_size / (1024 * 1024))
                        self.progress_signal.emit(filename, 100, size_mib)
                        continue  # Skip to next file

                    # Cancel the timer when download is complete
                    if self._save_timer:
                        self._save_timer.cancel()
                        self._save_timer = None
                    self._current_file = None

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
                self.status_update.emit(f"Error downloading {filename}: {e}. Partial file preserved for resume.")
                error_count += 1
                # Keep partial file for future resume attempts
                if os.path.exists(temp_file_path):
                    logging.info(f"Preserved partial file {temp_file_path} for future resume")
            except IOError as e:
                logging.error(f"File I/O error for {filename}: {e}")
                self.status_update.emit(f"File error for {filename}: {e}. Partial file preserved for resume.")
                error_count += 1
                # Keep partial file for future resume attempts
                if os.path.exists(temp_file_path):
                    logging.info(f"Preserved partial file {temp_file_path} for future resume")
            except Exception as e:
                logging.exception(f"An unexpected error occurred during download of {filename}")
                self.status_update.emit(f"Unexpected error for {filename}: {e}. Partial file preserved for resume.")
                error_count += 1
                # Keep partial file for future resume attempts
                if os.path.exists(temp_file_path):
                    logging.info(f"Preserved partial file {temp_file_path} for future resume")

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

        # Cancel any active save timer
        if self._save_timer:
            self._save_timer.cancel()
            self._save_timer = None

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

        # Flag to prevent circular updates between text field and dropdown
        self.updating_subdir_ui = False

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

        # Subdirectory filter
        self.subdir_filter_input = QLineEdit()
        self.subdir_filter_input.setPlaceholderText("Optional: Filter by subdirectory (e.g., 'ggml' or 'gguf')")
        input_layout.addRow("Subdirectory Filter:", self.subdir_filter_input)

        # Subdirectory dropdown with fetch button
        subdir_layout = QHBoxLayout()
        self.subdir_combo = QComboBox()
        self.subdir_combo.setEnabled(False)  # Initially disabled until repo is queried
        self.subdir_combo.addItem("Loading subdirectories...")
        self.subdir_combo.currentTextChanged.connect(self.on_subdir_selected)
        self.fetch_button = QPushButton("Fetch")
        self.fetch_button.clicked.connect(self.fetch_subdirectories)
        subdir_layout.addWidget(self.subdir_combo)
        subdir_layout.addWidget(self.fetch_button)
        input_layout.addRow("Available Subdirs:", subdir_layout)

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
        subdir_filter = self.subdir_filter_input.text().strip() or None

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
        self.download_thread = DownloadWorker(repo_id, local_dir, token_to_use, subdir_filter)
        # Connect signals
        self.download_thread.status_update.connect(self.update_status)
        self.download_thread.initial_files_signal.connect(self.populate_initial_progress_bars)
        self.download_thread.new_file_signal.connect(self.add_or_update_progress_bar)
        self.download_thread.file_cached_signal.connect(self.mark_file_as_cached) # Connect cache signal
        self.download_thread.progress_signal.connect(self.update_progress_bar)
        self.download_thread.finished_signal.connect(self.download_finished)
        self.download_thread.finished.connect(self.download_thread_cleanup) # Clean up thread object
        self.download_thread.subdirs_discovered_signal.connect(self.update_subdirectory_dropdown)
        self.download_thread.start()

    def clear_progress_bars(self):
        """Removes all widgets from the progress area layout."""
        # Clear all widgets from the layout
        while self.progress_area_layout.count():
            child = self.progress_area_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear the dictionary
        self.progress_bars.clear()

    def _format_size_string(self, size_bytes):
        """Helper to format file sizes consistently."""
        if size_bytes == -1:
            return " (Unknown size)"
        elif size_bytes <= 0:
            return ""

        # Use a list of units and a loop for cleaner code
        units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
        size = float(size_bytes)
        unit_index = 0

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        return f" ({size:.2f} {units[unit_index]})"

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

    def _create_progress_bar_widget(self, filename, size_mib, initial_format):
        """Helper to create and store label/pbar widgets."""
        # Format size string directly from MiB
        size_str = f" ({size_mib:.2f} MiB)" if size_mib > 0 else ""

        # Extract just the filename part for display
        display_name = filename.split('/')[-1]
        label = QLabel(f"{display_name}{size_str}")

        pbar = QProgressBar()
        pbar.setMinimum(0)
        pbar.setMaximum(100)  # Always use 100 for percentage-based display
        pbar.setValue(0)  # Start at 0
        pbar.setTextVisible(True)
        pbar.setFormat(initial_format)

        # Create a container widget for better organization
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)
        container_layout.addWidget(label)
        container_layout.addWidget(pbar)

        # Store label and pbar for later updates
        self.progress_bars[filename] = {'bar': pbar, 'label': label, 'container': container}
        self.progress_area_layout.addWidget(container)

        logging.debug(f"Created progress bar for '{filename}' with format '{initial_format}'")

    @Slot(str, int) # Receives filename and size_in_mib
    def add_or_update_progress_bar(self, filename, size_mib):
        """
        Updates a progress bar when a file download starts (or resumes).
        Called when the new_file_signal is received.
        """
        logging.info(f"Starting download for: {filename} ({size_mib} MiB)")

        # Extract just the filename part if it contains a path (should match QtTqdm logic)
        display_name = filename.split('/')[-1]

        # Format size string directly from MiB
        size_str = f" ({size_mib:.2f} MiB)" if size_mib > 0 else ""

        if filename in self.progress_bars:
            # Update existing progress bar (expected case)
            pbar_info = self.progress_bars[filename]
            pbar = pbar_info['bar']
            label = pbar_info['label']

            # Update label with size
            label.setText(f"{display_name}{size_str}")

            # Ensure progress bar is set to use percentage (0-100)
            if pbar.maximum() != 100:
                pbar.setMaximum(100)

            # Update progress bar properties for download start
            if size_mib > 0:
                pbar.setValue(0) # Reset to start of download
                pbar.setFormat("Downloading: %p%") # Set downloading state
            elif size_mib == 0: # Specifically zero-byte
                pbar.setValue(100) # Mark as complete visually
                pbar.setFormat("Complete (0 B)")
            else: # Unknown size
                pbar.setValue(0)
                pbar.setFormat("Pending...")

            logging.debug(f"add_or_update_progress_bar: Updated bar for '{filename}' to download state.")

        else:
            # Fallback: Create new progress bar if it wasn't created initially
            # This might happen if list_repo_files failed or had timing issues
            logging.warning(f"add_or_update_progress_bar: Progress bar for '{filename}' not found. Creating now.")
            initial_format = "Downloading: %p%" if size_mib > 0 else ("Complete (0 B)" if size_mib == 0 else "Pending...")
            self._create_progress_bar_widget(filename, size_mib, initial_format)
            # Ensure the newly created bar reflects the current state (value 0 or 100 for 0 B)
            if filename in self.progress_bars:
                new_pbar = self.progress_bars[filename]['bar']
                if size_mib == 0:
                    new_pbar.setValue(new_pbar.maximum()) # Set to max (100)
                else:
                    new_pbar.setValue(0)

    @Slot(str, int) # Receives filename and size_in_mib
    def mark_file_as_cached(self, filename, size_mib):
        """Marks a progress bar as 'Cached' when the file already exists."""
        logging.debug(f"Marking '{filename}' as Cached ({size_mib} MiB) in UI.")
        if filename in self.progress_bars:
            pbar_info = self.progress_bars[filename]
            pbar = pbar_info['bar']
            label = pbar_info['label']

            # Update label with size in MiB if not already done
            display_name = filename.split('/')[-1]
            size_str = f" ({size_mib:.2f} MiB)" if size_mib > 0 else ""
            label.setText(f"{display_name}{size_str}")

            # Update progress bar state to 100%
            if pbar.maximum() != 100:
                pbar.setMaximum(100)
            pbar.setValue(100) # Set to 100%
            pbar.setFormat("Cached")
        else:
            logging.warning(f"mark_file_as_cached: Progress bar for '{filename}' not found.")
            # Optionally create it here if needed, marked as cached
            self._create_progress_bar_widget(filename, size_mib, "Cached")
            if filename in self.progress_bars:
                pbar = self.progress_bars[filename]['bar']
                # Ensure progress bar is set to use percentage (0-100)
                if pbar.maximum() != 100:
                    pbar.setMaximum(100)
                pbar.setValue(100) # Set to 100%

    @Slot(str, int, int) # Receives filename, percentage (0-100), and size_in_mib
    def update_progress_bar(self, filename, percentage, size_mib):
        """Updates the value of a specific progress bar during download."""
        if filename in self.progress_bars:
            pbar_info = self.progress_bars[filename]
            pbar = pbar_info['bar']

            # Ensure progress bar is set to use percentage (0-100)
            if pbar.maximum() != 100:
                pbar.setMaximum(100)

            # Set the value directly from the percentage we received
            pbar.setValue(percentage)

            # Update the format based on state
            if percentage < 100:
                # Show downloading with MiB
                pbar.setFormat(f"Downloading: {percentage}% ({size_mib:.2f} MiB)")
            else:
                # Mark as complete if 100%
                if pbar.format() != "Complete":
                    pbar.setFormat("Complete")
                    logging.debug(f"update_progress_bar: Download marked complete for '{filename}'")
        else:
            # Handle missing progress bar case (should be less likely now)
            logging.warning(f"update_progress_bar: Progress bar for '{filename}' not found during update. Attempting to create.")
            # Create with appropriate state based on percentage and size_mib
            initial_format = f"Downloading: {percentage}% ({size_mib:.2f} MiB)" if percentage < 100 else "Complete"
            self._create_progress_bar_widget(filename, size_mib, initial_format)
            # Set the value directly after creating
            if filename in self.progress_bars:
                pbar = self.progress_bars[filename]['bar']
                pbar.setValue(percentage)
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

    @Slot(list)
    def update_subdirectory_dropdown(self, subdirectories):
        """Updates the subdirectory dropdown with discovered directories."""
        self.updating_subdir_ui = True
        try:
            self.subdir_combo.clear()
            self.subdir_combo.setEnabled(True)

            # Add "All files" option first
            self.subdir_combo.addItem("All files")

            # Add each subdirectory
            for subdir in subdirectories:
                self.subdir_combo.addItem(subdir)

            # Select the current filter if it exists in the list
            current_filter = self.subdir_filter_input.text().strip()
            if current_filter:
                index = self.subdir_combo.findText(current_filter)
                if index >= 0:
                    self.subdir_combo.setCurrentIndex(index)
                else:
                    # If filter doesn't match any subdirectory, select "All files"
                    self.subdir_combo.setCurrentIndex(0)
            else:
                # No filter, select "All files"
                self.subdir_combo.setCurrentIndex(0)
        finally:
            self.updating_subdir_ui = False

    @Slot(str)
    def on_subdir_selected(self, subdir):
        """Handles selection from the subdirectory dropdown."""
        if self.updating_subdir_ui:
            return  # Avoid circular updates
            
        self.updating_subdir_ui = True
        try:
            if subdir == "All files":
                # Clear the filter text field
                self.subdir_filter_input.setText("")
            else:
                # Update the filter text field with selected subdirectory
                self.subdir_filter_input.setText(subdir)
        finally:
            self.updating_subdir_ui = False
            
    @Slot()
    def fetch_subdirectories(self):
        """Fetches subdirectories without starting a full download."""
        repo_id = self.repo_id_input.text().strip()
        
        # Input validation
        if not repo_id:
            QMessageBox.warning(self, "Input Error", "Please enter a Hugging Face Repo ID.")
            return
            
        # Get token (optional, for gated models)
        hf_token = self.hf_token_input.text().strip() or None
        token_to_use = hf_token or os.environ.get("HF_TOKEN")  # Check env var as fallback
        
        # Update UI
        self.update_status(f"Status: Fetching subdirectories for {repo_id}...")
        self.fetch_button.setEnabled(False)
        self.subdir_combo.clear()
        self.subdir_combo.addItem("Fetching...")
        
        # Create a worker just for fetching subdirectories
        fetch_worker = DownloadWorker(repo_id, "", token_to_use)
        fetch_worker.status_update.connect(self.update_status)
        fetch_worker.subdirs_discovered_signal.connect(self.update_subdirectory_dropdown)
        fetch_worker.finished_signal.connect(lambda msg, is_error: self.fetch_button.setEnabled(True))
        
        # Override run method to only fetch subdirectories
        def fetch_only_run():
            if not fetch_worker._is_running:
                return
                
            try:
                # Only get repo files to discover subdirectories
                fetch_worker.status_update.emit(f"Listing subdirectories in {fetch_worker.repo_id} via API...")
                repo_files = fetch_worker._get_repo_files()
                
                if not fetch_worker._is_running:
                    return
                    
                # Emit the list of subdirectories discovered
                fetch_worker.subdirs_discovered_signal.emit(sorted(list(fetch_worker._subdirectories)))
                fetch_worker.status_update.emit(f"Found {len(fetch_worker._subdirectories)} subdirectories in repository")
                fetch_worker.finished_signal.emit("Subdirectory fetch complete", False)
                
            except (ConnectionError, ValueError, RuntimeError) as e:
                logging.error(f"Failed to get subdirectories: {e}")
                if fetch_worker._is_running:
                    fetch_worker.status_update.emit(f"Error fetching subdirectories: {e}")
                    fetch_worker.finished_signal.emit(f"Error fetching subdirectories: {e}", True)
        
        # Replace the run method with our custom one
        fetch_worker.run = fetch_only_run
        fetch_worker.start()

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
