#!/usr/bin/env python3
from PySide6.QtGui import QFont
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QComboBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QCheckBox, QTabWidget,
    QTextEdit
)
from gguf.quants import quantize, dequantize
from gguf.constants import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader
import sys
import os
import argparse
import numpy as np
import logging
import time
import sqlite3
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("QuantizationAnalyzer")

# Add parent directory to path to import gguf
sys.path.insert(0, str(Path(__file__).parent.parent))


# Number of histogram buckets
HISTOGRAM_BUCKETS = 150
HISTOGRAM_RANGE = 0.03


class ErrorStats:
    """Class to track error statistics for quantization"""

    def __init__(self):
        self.num_samples = 0
        self.total_error = 0.0
        self.max_error = 0.0
        self.error_histogram = np.zeros(HISTOGRAM_BUCKETS, dtype=np.uint64)

    def update(self, original: np.ndarray, quantized: np.ndarray):
        """Update error statistics with new data"""
        logger.debug("Calculating differences between original and quantized data")
        diff = original - quantized
        squared_diff = diff * diff

        logger.debug("Summing squared differences")
        sum_squared = np.sum(squared_diff)
        self.total_error += sum_squared

        logger.debug("Finding maximum absolute error")
        max_abs_error = np.max(np.abs(diff))
        self.max_error = max(self.max_error, max_abs_error)

        self.num_samples += original.size

        logger.debug(f"Updating histogram with {original.size} samples")
        # Update histogram - this can be slow for large tensors
        # Process in batches to avoid memory issues
        batch_size = 1000000  # Process 1M elements at a time
        flattened = np.abs(diff.flatten())

        for i in range(0, len(flattened), batch_size):
            batch = flattened[i:i +batch_size]
            buckets = np.minimum((batch / HISTOGRAM_RANGE * HISTOGRAM_BUCKETS).astype(np.int64),
                                 HISTOGRAM_BUCKETS - 1)
            for b in range(HISTOGRAM_BUCKETS):
                self.error_histogram[b] += np.sum(buckets == b)

        logger.debug(f"Update completed. RMSE: {self.get_rmse():.6f}, Max Error: {self.max_error:.6f}")

    def combine(self, other: 'ErrorStats'):
        """Combine with another ErrorStats object"""
        self.num_samples += other.num_samples
        self.total_error += other.total_error
        self.max_error = max(self.max_error, other.max_error)
        self.error_histogram += other.error_histogram

    def get_rmse(self) -> float:
        """Get root mean square error"""
        if self.num_samples == 0:
            return 0.0
        return np.sqrt(self.total_error / self.num_samples)

    def get_quantile(self, q: float) -> float:
        """Find the error value at the given quantile"""
        if self.num_samples == 0:
            return 0.0

        total = np.sum(self.error_histogram)
        accum = 0

        for i in range(HISTOGRAM_BUCKETS):
            accum += self.error_histogram[i]
            if accum >= total * q:
                return (i + 1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS

        return float('inf')

    def get_median(self) -> float:
        """Get median error"""
        return self.get_quantile(0.5)

    def get_percentile95(self) -> float:
        """Get 95th percentile error"""
        return self.get_quantile(0.95)

    def get_histogram_data(self) -> Tuple[List[float], List[int]]:
        """Get histogram data as lists of bin edges and counts"""
        bin_edges = [i * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS for i in range(HISTOGRAM_BUCKETS + 1)]
        counts = self.error_histogram.tolist()
        return bin_edges, counts


class QuantizationWorker(QThread):
    """Worker thread for quantization analysis"""
    progress_updated = Signal(int, int)  # current, total
    layer_completed = Signal(str, object, str)  # layer_name, error_stats, quant_type_name
    all_completed = Signal(dict)  # all_stats

    def __init__(self, model_path: str, quant_types: List[GGMLQuantizationType],
                 include_layers: List[str], exclude_layers: List[str], db_path: str):
        super().__init__()
        self.model_path = model_path
        self.quant_types = quant_types
        self.include_layers = include_layers
        self.exclude_layers = exclude_layers
        self.reader = None
        self.stop_requested = False
        self.db_path = db_path

    def run(self):
        try:
            logger.info(f"Starting to load model from {self.model_path}")
            start_time = time.time()
            self.reader = GGUFReader(self.model_path)
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

            # Get all tensor names
            logger.info("Getting tensor names")
            tensor_names = [tensor.name for tensor in self.reader.tensors]
            logger.info(f"Found {len(tensor_names)} tensors in the model")

            filtered_names = self._filter_tensor_names(tensor_names)
            logger.info(f"After filtering, {len(filtered_names)} tensors will be processed")

            # Initialize database connection for this thread
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            logger.info(f"Connected to database at {self.db_path}")

            all_stats = {}

            for quant_type in self.quant_types:
                if self.stop_requested:
                    logger.info("Stop requested, breaking out of quantization loop")
                    break

                logger.info(f"Starting analysis for quantization type: {quant_type.name}")
                type_stats = ErrorStats()
                all_stats[quant_type] = {
                    'global': type_stats,
                    'layers': {}
                }

                # Create or update the global entry for this quantization type
                model_name = os.path.basename(self.model_path)
                timestamp = datetime.now().isoformat()

                # Check if entry exists
                cursor.execute(
                    "SELECT id FROM quantization_results WHERE model_path = ? AND quant_type = ?",
                    (self.model_path, quant_type.name)
                )
                existing = cursor.fetchone()

                if existing:
                    logger.info(f"Found existing entry for {quant_type.name}, will update")
                    result_id = existing[0]
                    # Delete existing layer results
                    cursor.execute("DELETE FROM layer_results WHERE result_id = ?", (result_id,))
                else:
                    logger.info(f"Creating new entry for {quant_type.name}")
                    # Insert initial entry with placeholder values
                    cursor.execute('''
                    INSERT INTO quantization_results
                    (model_path, model_name, quant_type, rmse, max_error, percentile_95, median, timestamp)
                    VALUES (?, ?, ?, 0, 0, 0, 0, ?)
                    ''', (self.model_path, model_name, quant_type.name, timestamp))
                    result_id = cursor.lastrowid

                conn.commit()
                logger.info(f"Database entry created/updated with ID {result_id}")

                for i, tensor_name in enumerate(filtered_names):
                    if self.stop_requested:
                        logger.info("Stop requested, breaking out of tensor processing loop")
                        break

                    logger.info(f"Processing tensor {i+1}/{len(filtered_names)}: {tensor_name}")
                    self.progress_updated.emit(i, len(filtered_names))

                    # Get tensor data
                    logger.debug(f"Getting tensor data for {tensor_name}")
                    tensor = next(t for t in self.reader.tensors if t.name == tensor_name)

                    # Skip tensors that are already quantized
                    if tensor.tensor_type != GGMLQuantizationType.F32 and tensor.tensor_type != GGMLQuantizationType.F16:
                        logger.info(f"Skipping {tensor_name} as it's already quantized: {tensor.tensor_type.name}")
                        continue

                    # Get tensor data as float32
                    logger.debug("Converting tensor data to float32")
                    start_time = time.time()
                    original_data = tensor.data.astype(np.float32)
                    logger.debug(f"Tensor shape: {original_data.shape}, conversion took {time.time() - start_time:.2f} seconds")

                    # Skip tensors with dimensions not compatible with the quantization type
                    if original_data.shape[-1] % 32 != 0:  # Most quants need multiple of 32
                        logger.info(f"Skipping {tensor_name} as its last dimension {original_data.shape[-1]} is not a multiple of 32")
                        continue

                    try:
                        # Quantize and dequantize
                        logger.debug(f"Starting quantization of {tensor_name}")
                        start_time = time.time()
                        quantized_data = quantize(original_data, quant_type)
                        logger.debug(f"Quantization completed in {time.time() - start_time:.2f} seconds")

                        logger.debug("Starting dequantization")
                        start_time = time.time()
                        dequantized_data = dequantize(quantized_data, quant_type)
                        logger.debug(f"Dequantization completed in {time.time() - start_time:.2f} seconds")

                        # Calculate error statistics
                        logger.debug("Calculating error statistics")
                        start_time = time.time()
                        layer_stats = ErrorStats()
                        layer_stats.update(original_data, dequantized_data)
                        logger.debug(f"Error calculation completed in {time.time() - start_time:.2f} seconds")

                        # Update global stats
                        type_stats.combine(layer_stats)

                        # Store layer stats
                        all_stats[quant_type]['layers'][tensor_name] = layer_stats

                        # Save layer stats to database
                        try:
                            cursor.execute('''
                            INSERT INTO layer_results
                            (result_id, layer_name, rmse, max_error, percentile_95, median)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ''', (
                                result_id,
                                tensor_name,
                                layer_stats.get_rmse(),
                                f"{layer_stats.max_error:.8f}",
                                layer_stats.get_percentile95(),
                                layer_stats.get_median()
                            ))
                            conn.commit()
                            logger.debug(f"Saved layer stats for {tensor_name} to database")
                        except Exception as db_err:
                            logger.error(f"Error saving layer stats to database: {db_err}", exc_info=True)

                        # Emit signal for layer completion
                        self.layer_completed.emit(f"{quant_type.name}::{tensor_name}", layer_stats, quant_type.name)
                        logger.info(f"Completed processing {tensor_name}, RMSE: {layer_stats.get_rmse():.6f}, Max Error: {layer_stats.max_error:.6f}")

                    except Exception as e:
                        logger.error(f"Error processing {tensor_name} with {quant_type.name}: {e}", exc_info=True)

                # Update the global stats for this quantization type
                try:
                    cursor.execute('''
                    UPDATE quantization_results
                    SET rmse = ?, max_error = ?, percentile_95 = ?, median = ?, timestamp = ?
                    WHERE id = ?
                    ''', (
                        type_stats.get_rmse(),
                        f"{type_stats.max_error:.8f}",
                        type_stats.get_percentile95(),
                        type_stats.get_median(),
                        timestamp,
                        result_id
                    ))
                    conn.commit()
                    logger.info(f"Updated global stats for {quant_type.name} in database")
                except Exception as db_err:
                    logger.error(f"Error updating global stats in database: {db_err}", exc_info=True)

            # Close database connection
            conn.close()
            logger.info("Database connection closed")

            logger.info("Analysis completed, emitting results")
            self.all_completed.emit(all_stats)

        except Exception as e:
            logger.error(f"Error in worker thread: {e}", exc_info=True)

    def _filter_tensor_names(self, tensor_names: List[str]) -> List[str]:
        """Filter tensor names based on include/exclude patterns"""
        result = []

        for name in tensor_names:
            # Skip if in exclude list
            if any(exclude in name for exclude in self.exclude_layers):
                continue

            # Include if in include list or include list is empty
            if not self.include_layers or any(include in name for include in self.include_layers):
                result.append(name)

        return result

    def stop(self):
        self.stop_requested = True


class QuantizationAnalyzer(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GGUF Quantization Analyzer")
        self.setMinimumSize(1000, 700)

        self.model_path = ""
        self.worker = None
        self.all_stats = {}
        self.db_path = os.path.join(os.path.dirname(__file__), "quantization_stats.db")

        # Initialize database
        self._init_database()

        self._setup_ui()

    def _setup_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_path_label = QLabel("No model selected")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_model)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_path_label, 1)
        model_layout.addWidget(browse_button)

        # Quantization type selection
        quant_layout = QHBoxLayout()
        quant_label = QLabel("Quantization Types:")
        self.quant_combo = QComboBox()

        # Add all quantization types
        for quant_type in GGMLQuantizationType:
            self.quant_combo.addItem(quant_type.name, quant_type)

        quant_layout.addWidget(quant_label)
        quant_layout.addWidget(self.quant_combo, 1)

        # Layer filtering
        filter_layout = QHBoxLayout()
        include_label = QLabel("Include Layers:")
        self.include_edit = QTextEdit()
        self.include_edit.setMaximumHeight(60)
        self.include_edit.setPlaceholderText("Enter patterns to include, one per line")

        exclude_label = QLabel("Exclude Layers:")
        self.exclude_edit = QTextEdit()
        self.exclude_edit.setMaximumHeight(60)
        self.exclude_edit.setPlaceholderText("Enter patterns to exclude, one per line")

        filter_layout.addWidget(include_label)
        filter_layout.addWidget(self.include_edit, 1)
        filter_layout.addWidget(exclude_label)
        filter_layout.addWidget(self.exclude_edit, 1)

        # Options
        options_layout = QHBoxLayout()
        self.per_layer_checkbox = QCheckBox("Show per-layer statistics")
        self.per_layer_checkbox.setChecked(True)  # Enable by default
        self.histogram_checkbox = QCheckBox("Show error histogram")
        self.histogram_checkbox.setChecked(True)  # Enable by default

        options_layout.addWidget(self.per_layer_checkbox)
        options_layout.addWidget(self.histogram_checkbox)
        options_layout.addStretch(1)

        # Analyze button
        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze Quantization")
        self.analyze_button.clicked.connect(self._start_analysis)
        self.analyze_button.setEnabled(False)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_analysis)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch(1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m layers processed")

        # Results tabs
        self.results_tabs = QTabWidget()

        # Summary tab
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels(["Type", "RMSE", "Max Error", "95th Percentile", "Median"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Layers tab
        self.layers_table = QTableWidget()
        self.layers_table.setColumnCount(5)
        self.layers_table.setHorizontalHeaderLabels(["Layer", "RMSE", "Max Error", "95th Percentile", "Median"])
        self.layers_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Histogram tab
        self.histogram_text = QTextEdit()
        self.histogram_text.setReadOnly(True)
        self.histogram_text.setFont(QFont("Monospace"))

        # Add tabs
        self.results_tabs.addTab(self.summary_table, "Summary")
        self.results_tabs.addTab(self.layers_table, "Layers")
        self.results_tabs.addTab(self.histogram_text, "Histogram")

        # Add all layouts to main layout
        main_layout.addLayout(model_layout)
        main_layout.addLayout(quant_layout)
        main_layout.addLayout(filter_layout)
        main_layout.addLayout(options_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.results_tabs, 1)

        self.setCentralWidget(main_widget)

    def _browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Models (*.gguf);;All Files (*)"
        )

        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(os.path.basename(file_path))
            self.analyze_button.setEnabled(True)

            # Check if we have existing results and load them
            self._load_existing_results()

    def _start_analysis(self):
        if not self.model_path:
            return

        # Check if we already have results for this model and quantization type
        selected_quant = self.quant_combo.currentData()
        if self._check_existing_results(self.model_path, selected_quant.name):
            # Ask user if they want to overwrite
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                'Overwrite Existing Results',
                f'Analysis results already exist for {os.path.basename(self.model_path)} with {selected_quant.name}. Overwrite?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

        logger.info(f"Starting analysis for model: {self.model_path}")

        # Clear previous results
        self.summary_table.setRowCount(0)
        self.layers_table.setRowCount(0)
        self.histogram_text.clear()

        # Get selected quantization types
        quant_types = [selected_quant]
        logger.info(f"Selected quantization type: {selected_quant.name}")

        # Get include/exclude patterns
        include_patterns = [line.strip() for line in self.include_edit.toPlainText().split('\n') if line.strip()]
        exclude_patterns = [line.strip() for line in self.exclude_edit.toPlainText().split('\n') if line.strip()]

        logger.info(f"Include patterns: {include_patterns}")
        logger.info(f"Exclude patterns: {exclude_patterns}")

        # Show the summary tab initially
        self.results_tabs.setCurrentIndex(0)

        # Create and start worker thread
        self.worker = QuantizationWorker(
            self.model_path,
            quant_types,
            include_patterns,
            exclude_patterns,
            self.db_path
        )

        self.worker.progress_updated.connect(self._update_progress)
        self.worker.layer_completed.connect(self._layer_completed)
        self.worker.all_completed.connect(self._analysis_completed)

        self.analyze_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # Initialize the summary table with a placeholder row
        self.summary_table.setRowCount(1)
        self.summary_table.setItem(0, 0, QTableWidgetItem(selected_quant.name))
        self.summary_table.setItem(0, 1, QTableWidgetItem("Calculating..."))
        self.summary_table.setItem(0, 2, QTableWidgetItem("Calculating..."))
        self.summary_table.setItem(0, 3, QTableWidgetItem("Calculating..."))
        self.summary_table.setItem(0, 4, QTableWidgetItem("Calculating..."))

        logger.info("Starting worker thread")
        self.worker.start()

    def _stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.analyze_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def _update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

        # Force UI update
        QApplication.processEvents()

    def _layer_completed(self, layer_name, stats, quant_type_name=None):
        # Always update the table, but only show it if checkbox is checked
        row = self.layers_table.rowCount()
        self.layers_table.insertRow(row)

        self.layers_table.setItem(row, 0, QTableWidgetItem(layer_name))
        self.layers_table.setItem(row, 1, QTableWidgetItem(f"{stats.get_rmse():.8f}"))
        self.layers_table.setItem(row, 2, QTableWidgetItem(f"{stats.max_error:.8f}"))
        self.layers_table.setItem(row, 3, QTableWidgetItem(f"{stats.get_percentile95():.4f}"))
        self.layers_table.setItem(row, 4, QTableWidgetItem(f"{stats.get_median():.4f}"))

        # Force the UI to update
        QApplication.processEvents()

        # Hide the layers tab if not checked
        if not self.per_layer_checkbox.isChecked():
            self.results_tabs.setTabVisible(1, False)
        else:
            self.results_tabs.setTabVisible(1, True)

    def _analysis_completed(self, all_stats):
        self.all_stats = all_stats
        self.analyze_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Update summary table
        self.summary_table.setRowCount(0)

        for quant_type, stats_dict in all_stats.items():
            global_stats = stats_dict['global']

            row = self.summary_table.rowCount()
            self.summary_table.insertRow(row)

            self.summary_table.setItem(row, 0, QTableWidgetItem(quant_type.name))
            self.summary_table.setItem(row, 1, QTableWidgetItem(f"{global_stats.get_rmse():.8f}"))
            self.summary_table.setItem(row, 2, QTableWidgetItem(f"{global_stats.max_error:.8f}"))
            self.summary_table.setItem(row, 3, QTableWidgetItem(f"{global_stats.get_percentile95():.4f}"))
            self.summary_table.setItem(row, 4, QTableWidgetItem(f"{global_stats.get_median():.4f}"))

        # Make sure we're showing the summary tab
        self.results_tabs.setCurrentIndex(0)

        # Update histogram if enabled
        if self.histogram_checkbox.isChecked():
            self._update_histogram()

        logger.info("Analysis completed and UI updated")

    def _update_histogram(self):
        self.histogram_text.clear()

        for quant_type, stats_dict in self.all_stats.items():
            global_stats = stats_dict['global']

            self.histogram_text.append(f"Error distribution for {quant_type.name}:")
            self.histogram_text.append("-" * 50)

            bin_edges, counts = global_stats.get_histogram_data()

            for i in range(HISTOGRAM_BUCKETS):
                lower = bin_edges[i]
                upper = bin_edges[i +1]
                if i == HISTOGRAM_BUCKETS - 1:
                    self.histogram_text.append(f"[{lower:.4f}, inf): {counts[i]}")
                else:
                    self.histogram_text.append(f"[{lower:.4f}, {upper:.4f}): {counts[i]}")

            self.histogram_text.append("\n")

    def _init_database(self):
        """Initialize the SQLite database for storing quantization results"""
        try:
            # Only create the database if it doesn't exist
            create_tables = not os.path.exists(self.db_path)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if create_tables:
                logger.info(f"Creating new database at {self.db_path}")
                # Create tables
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS quantization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_path TEXT,
                    model_name TEXT,
                    quant_type TEXT,
                    rmse NUMERIC,
                    max_error NUMERIC,
                    percentile_95 NUMERIC,
                    median NUMERIC,
                    timestamp TEXT,
                    UNIQUE(model_path, quant_type)
                )
                ''')

                cursor.execute('''
                CREATE TABLE IF NOT EXISTS layer_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id INTEGER,
                    layer_name TEXT,
                    rmse NUMERIC,
                    max_error NUMERIC,
                    percentile_95 NUMERIC,
                    median NUMERIC,
                    FOREIGN KEY(result_id) REFERENCES quantization_results(id)
                )
                ''')
            else:
                logger.info(f"Using existing database at {self.db_path}")

                # Verify tables exist and have correct schema
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quantization_results'")
                if not cursor.fetchone():
                    logger.warning("quantization_results table not found, creating it")
                    cursor.execute('''
                    CREATE TABLE quantization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_path TEXT,
                        model_name TEXT,
                        quant_type TEXT,
                        rmse NUMERIC,
                        max_error NUMERIC,
                        percentile_95 NUMERIC,
                        median NUMERIC,
                        timestamp TEXT,
                        UNIQUE(model_path, quant_type)
                    )
                    ''')

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='layer_results'")
                if not cursor.fetchone():
                    logger.warning("layer_results table not found, creating it")
                    cursor.execute('''
                    CREATE TABLE layer_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        result_id INTEGER,
                        layer_name TEXT,
                        rmse NUMERIC,
                        max_error NUMERIC,
                        percentile_95 NUMERIC,
                        median NUMERIC,
                        FOREIGN KEY(result_id) REFERENCES quantization_results(id)
                    )
                    ''')

            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)

    def _check_existing_results(self, model_path, quant_type):
        """Check if results already exist for this model and quantization type"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id FROM quantization_results WHERE model_path = ? AND quant_type = ?",
                (model_path, quant_type)
            )

            result = cursor.fetchone()
            conn.close()

            return result is not None
        except Exception as e:
            logger.error(f"Error checking existing results: {e}", exc_info=True)
            return False

    # Removed _save_results_to_db method as it's now handled in the worker thread

    def _load_existing_results(self):
        """Load existing results from the database if available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all quantization types for this model
            cursor.execute(
                "SELECT quant_type, rmse, max_error, percentile_95, median FROM quantization_results WHERE model_path = ?",
                (self.model_path,)
            )

            results = cursor.fetchall()

            if results:
                # Clear existing results
                self.summary_table.setRowCount(0)

                # Add results to summary table
                for result in results:
                    quant_type, rmse, max_error, percentile_95, median = result

                    row = self.summary_table.rowCount()
                    self.summary_table.insertRow(row)

                    # Convert values to float if they're not already
                    try:
                        rmse = float(rmse)
                        max_error = float(max_error)
                        percentile_95 = float(percentile_95)
                        median = float(median)
                    except (TypeError, ValueError):
                        logger.warning(f"Type conversion issue with values: {rmse}, {max_error}, {percentile_95}, {median}")

                    self.summary_table.setItem(row, 0, QTableWidgetItem(str(quant_type)))
                    self.summary_table.setItem(row, 1, QTableWidgetItem(f"{rmse:.8f}"))
                    self.summary_table.setItem(row, 2, QTableWidgetItem(f"{max_error:.8f}"))
                    self.summary_table.setItem(row, 3, QTableWidgetItem(f"{percentile_95:.4f}"))
                    self.summary_table.setItem(row, 4, QTableWidgetItem(f"{median:.4f}"))

                # Set the combo box to the first quantization type found
                first_quant_type = str(results[0][0])
                index = self.quant_combo.findText(first_quant_type)
                if index >= 0:
                    self.quant_combo.setCurrentIndex(index)

                logger.info(f"Loaded {len(results)} existing results for {os.path.basename(self.model_path)}")

            conn.close()
        except Exception as e:
            logger.error(f"Error loading existing results: {e}", exc_info=True)


def parse_args():
    parser = argparse.ArgumentParser(description="GGUF Quantization Analyzer")
    parser.add_argument("-m", "--model", type=str, help="Path to GGUF model file")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting GGUF Quantization Analyzer")

    app = QApplication(sys.argv)
    window = QuantizationAnalyzer()

    # If model path provided via command line
    if args.model:
        logger.info(f"Model path provided via command line: {args.model}")
        window.model_path = args.model
        window.model_path_label.setText(os.path.basename(args.model))
        window.analyze_button.setEnabled(True)
        window._load_existing_results()

    window.show()
    logger.info("Application window displayed")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
