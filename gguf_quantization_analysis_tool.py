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

from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget
)
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfFolder, HfHubHTTPError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Worker Thread for Downloads ---
class DownloadWorker(QThread):
    """
    Handles the Hugging Face model download in a separate thread
    to avoid blocking the GUI.
    """
    # TODO: Add signals for detailed progress (new_file_signal, progress_signal)
    # progress_signal = Signal(str, int, int) # filename, current_bytes, total_bytes
    # new_file_signal = Signal(str, int) # filename, total_bytes
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

        self.status_update.emit(f"Starting download of {self.repo_id}...")
        try:
            # TODO: Implement progress reporting hook if possible with snapshot_download
            #       or switch to manual file listing and downloading for progress.
            #       Consider using hf_transfer for potentially better performance/progress.
            logging.info(f"Downloading {self.repo_id} to {self.local_dir}")
            path = snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.local_dir,
                local_dir_use_symlinks=False, # Avoid symlinks for simplicity across OS
                token=self.token,
                # Example: ignore pytorch_model.bin if safetensors exist
                # ignore_patterns=["*.bin"], # Add more patterns as needed
                resume_download=True,
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
        self.progress_bars = {} # Dictionary to hold progress bars {filename: QProgressBar}

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
        # self.download_thread.new_file_signal.connect(self.add_progress_bar) # TODO
        # self.download_thread.progress_signal.connect(self.update_progress_bar) # TODO
        self.download_thread.finished_signal.connect(self.download_finished)
        self.download_thread.finished.connect(self.download_thread_cleanup) # Clean up thread object
        self.download_thread.start()

    def clear_progress_bars(self):
         """Removes all widgets from the progress area layout."""
         while self.progress_area_layout.count():
            child = self.progress_area_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater() # Ensure proper widget deletion
        self.progress_bars.clear()

    # --- Placeholder Slots for Progress Updates ---
    # @Slot(str, int)
    # def add_progress_bar(self, filename, total_bytes):
    #     """Adds a new progress bar for a file being downloaded."""
    #     if filename not in self.progress_bars:
    #         label = QLabel(f"{filename} ({(total_bytes / (1024*1024)):.2f} MB)")
    #         pbar = QProgressBar()
    #         pbar.setMaximum(total_bytes)
    #         pbar.setValue(0)
    #         pbar.setTextVisible(True)
    #         self.progress_area_layout.addWidget(label)
    #         self.progress_area_layout.addWidget(pbar)
    #         self.progress_bars[filename] = pbar
    #         logging.debug(f"Added progress bar for {filename}")

    # @Slot(str, int, int)
    # def update_progress_bar(self, filename, current_bytes, total_bytes):
    #     """Updates the value of a specific progress bar."""
    #     if filename in self.progress_bars:
    #         pbar = self.progress_bars[filename]
    #         pbar.setMaximum(total_bytes) # Ensure max is correct
    #         pbar.setValue(current_bytes)
    #     else:
    #         logging.warning(f"Attempted to update progress for unknown file: {filename}")

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
            # Optionally trigger next step, e.g., loading model structure
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
