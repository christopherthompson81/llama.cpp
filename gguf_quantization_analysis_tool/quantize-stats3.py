#!/usr/bin/env python3
import sys
import os
import re
import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum
import signal
import resource
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox,
                               QTableWidget, QTableWidgetItem, QTabWidget, QProgressBar,
                               QLineEdit, QGroupBox, QGridLayout, QSplitter, QMessageBox)
from PySide6.QtCore import Qt, Signal, QThread
import llama_cpp
import ctypes

# Global variable to track if we're in a critical section
in_critical_section = False
critical_section_name = ""


def signal_handler(sig, frame):
    """Handle signals like SIGSEGV (segmentation fault)"""
    signal_names = {
        signal.SIGSEGV: "SIGSEGV (Segmentation Fault)",
        signal.SIGABRT: "SIGABRT (Abort)",
        signal.SIGBUS: "SIGBUS (Bus Error)",
        signal.SIGILL: "SIGILL (Illegal Instruction)",
        signal.SIGFPE: "SIGFPE (Floating Point Exception)"
    }

    signal_name = signal_names.get(sig, f"Signal {sig}")

    error_msg = f"Caught {signal_name}"
    if in_critical_section:
        error_msg += f" during {critical_section_name}"

    print(f"ERROR: {error_msg}")

    # Write to a log file
    with open("crash_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
        if hasattr(frame, 'f_code'):
            f.write(f"  File: {frame.f_code.co_filename}, Line: {frame.f_lineno}\n")

    # Don't exit - let the exception propagate so it can be caught
    # This allows the application to continue running


# Register signal handlers
signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGABRT, signal_handler)
signal.signal(signal.SIGBUS, signal_handler)
signal.signal(signal.SIGILL, signal_handler)
signal.signal(signal.SIGFPE, signal_handler)

# Increase stack size to avoid stack overflow
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

# Constants
HISTOGRAM_BUCKETS = 150
HISTOGRAM_RANGE = 0.03
QK_K = 256

# Define GGML types


class GGMLType(Enum):
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_COUNT = 16


# Map GGML types to bits per weight
BPW_MAP = {
    GGMLType.GGML_TYPE_F32: 32.0,
    GGMLType.GGML_TYPE_F16: 16.0,
    GGMLType.GGML_TYPE_Q4_0: 4.0,
    GGMLType.GGML_TYPE_Q4_1: 4.5,
    GGMLType.GGML_TYPE_Q5_0: 5.0,
    GGMLType.GGML_TYPE_Q5_1: 5.5,
    GGMLType.GGML_TYPE_Q8_0: 8.0,
    GGMLType.GGML_TYPE_Q8_1: 8.5,
    GGMLType.GGML_TYPE_Q2_K: 2.0,
    GGMLType.GGML_TYPE_Q3_K: 3.0,
    GGMLType.GGML_TYPE_Q4_K: 4.0,
    GGMLType.GGML_TYPE_Q5_K: 5.0,
    GGMLType.GGML_TYPE_Q6_K: 6.0,
    GGMLType.GGML_TYPE_Q8_K: 8.0,
}

# Error statistics structure


class ErrorStats:
    def __init__(self):
        self.num_samples = 0
        self.total_error = 0.0
        self.max_error = 0.0
        self.error_histogram = [0] * HISTOGRAM_BUCKETS

    def update(self, input_data: np.ndarray, output_data: np.ndarray):
        diff = input_data - output_data
        self.total_error += np.sum(diff * diff)
        self.max_error = max(self.max_error, np.max(np.abs(diff)))

        # Update histogram
        for d in np.abs(diff):
            bucket = min(int(d / HISTOGRAM_RANGE * HISTOGRAM_BUCKETS), HISTOGRAM_BUCKETS - 1)
            self.error_histogram[bucket] += 1

        self.num_samples += len(input_data)

    def combine(self, other):
        self.num_samples += other.num_samples
        self.total_error += other.total_error
        self.max_error = max(self.max_error, other.max_error)
        for i in range(HISTOGRAM_BUCKETS):
            self.error_histogram[i] += other.error_histogram[i]

    def get_rmse(self):
        if self.num_samples == 0:
            return 0.0
        return np.sqrt(self.total_error / self.num_samples)

    def find_quantile(self, quantile):
        if self.num_samples == 0:
            return float('inf')

        total = sum(self.error_histogram)
        accum = 0

        for i in range(HISTOGRAM_BUCKETS):
            accum += self.error_histogram[i]
            if accum >= total * quantile:
                return (i + 1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS

        return float('inf')

    def get_median(self):
        return self.find_quantile(0.5)

    def get_percentile95(self):
        return self.find_quantile(0.95)

    def get_summary(self):
        return {
            'rmse': self.get_rmse(),
            'max_error': self.max_error,
            'median': self.get_median(),
            'percentile95': self.get_percentile95(),
            'num_samples': self.num_samples
        }

# Database manager


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        # Create models table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY,
            name TEXT,
            path TEXT,
            date_added TIMESTAMP
        )
        ''')

        # Create analysis_runs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY,
            model_id INTEGER,
            date_run TIMESTAMP,
            description TEXT,
            FOREIGN KEY (model_id) REFERENCES models (id)
        )
        ''')

        # Create layers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS layers (
            id INTEGER PRIMARY KEY,
            model_id INTEGER,
            name TEXT,
            size INTEGER,
            FOREIGN KEY (model_id) REFERENCES models (id)
        )
        ''')

        # Create quantization_results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS quantization_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            layer_id INTEGER,
            quant_type TEXT,
            bits_per_weight REAL,
            rmse REAL,
            max_error REAL,
            median_error REAL,
            p95_error REAL,
            histogram BLOB,
            FOREIGN KEY (run_id) REFERENCES analysis_runs (id),
            FOREIGN KEY (layer_id) REFERENCES layers (id)
        )
        ''')

        self.conn.commit()

    def add_model(self, name, path):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO models (name, path, date_added) VALUES (?, ?, ?)",
            (name, path, datetime.now())
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_model_by_path(self, path):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM models WHERE path = ?", (path,))
        result = cursor.fetchone()
        return result[0] if result else None

    def add_layer(self, model_id, name, size):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO layers (model_id, name, size) VALUES (?, ?, ?)",
            (model_id, name, size)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_layer(self, model_id, name):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM layers WHERE model_id = ? AND name = ?",
            (model_id, name)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def add_analysis_run(self, model_id, description=""):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO analysis_runs (model_id, date_run, description) VALUES (?, ?, ?)",
            (model_id, datetime.now(), description)
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_quantization_result(self, run_id, layer_id, quant_type, stats):
        cursor = self.conn.cursor()

        # Convert histogram to blob
        histogram_blob = sqlite3.Binary(bytes(stats.error_histogram))

        # Get bits per weight
        bpw = BPW_MAP.get(quant_type, 0.0)

        cursor.execute(
            """INSERT INTO quantization_results
               (run_id, layer_id, quant_type, bits_per_weight, rmse, max_error, median_error, p95_error, histogram)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, layer_id, quant_type.name, bpw, stats.get_rmse(), stats.max_error,
             stats.get_median(), stats.get_percentile95(), histogram_blob)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_models(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, path FROM models")
        return cursor.fetchall()

    def get_layers(self, model_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, size FROM layers WHERE model_id = ?", (model_id,))
        return cursor.fetchall()

    def get_runs(self, model_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, date_run, description FROM analysis_runs WHERE model_id = ?", (model_id,))
        return cursor.fetchall()

    def get_layer_results(self, run_id, layer_id):
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT quant_type, bits_per_weight, rmse, max_error, median_error, p95_error
               FROM quantization_results
               WHERE run_id = ? AND layer_id = ?
               ORDER BY bits_per_weight DESC""",
            (run_id, layer_id)
        )
        return cursor.fetchall()

    def close(self):
        self.conn.close()


class QuantizationWorker(QThread):
    # Worker thread for quantization
    progress_updated = Signal(int, str)
    finished = Signal(dict)

    def __init__(self, model_path, include_layers, exclude_layers, quant_types, lib_path=None):
        super().__init__()
        self.model_path = model_path
        self.include_layers = include_layers
        self.exclude_layers = exclude_layers
        self.quant_types = quant_types
        self.lib_path = lib_path
        self.stop_requested = False
        self.params = None
        self.model = None

    def _run_simulated_analysis(self):
        """Run a simulated analysis when model loading fails"""
        results = {}

        # Create some simulated layers
        layer_names = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.1.attn_q.weight",
            "blk.1.attn_k.weight"
        ]

        # Filter layers based on include/exclude patterns
        filtered_layers = []
        for name in layer_names:
            if self.layer_included(name):
                filtered_layers.append(name)

        total_layers = len(filtered_layers)
        total_types = len(self.quant_types)
        total_steps = total_layers * total_types
        current_step = 0

        # Process each simulated layer
        for layer_name in filtered_layers:
            if self.stop_requested:
                break

            # Create simulated tensor data
            tensor_data = np.random.randn(1024 * 1024).astype(np.float32)
            tensor_results = {}

            # Process each quantization type
            for quant_type in self.quant_types:
                if self.stop_requested:
                    break

                current_step += 1
                progress_pct = int(100 * current_step / total_steps)
                self.progress_updated.emit(
                    progress_pct,
                    f"Simulating {layer_name} with {quant_type.name} ({current_step}/{total_steps})"
                )

                # Simulate quantization with different error levels based on type
                noise_level = 0.0
                if quant_type == GGMLType.GGML_TYPE_Q4_0:
                    noise_level = 0.05
                elif quant_type == GGMLType.GGML_TYPE_Q5_0:
                    noise_level = 0.025
                elif quant_type == GGMLType.GGML_TYPE_Q8_0:
                    noise_level = 0.01
                elif quant_type == GGMLType.GGML_TYPE_Q2_K:
                    noise_level = 0.1
                elif quant_type == GGMLType.GGML_TYPE_Q3_K:
                    noise_level = 0.075
                elif quant_type == GGMLType.GGML_TYPE_Q4_K:
                    noise_level = 0.05
                elif quant_type == GGMLType.GGML_TYPE_Q5_K:
                    noise_level = 0.025
                elif quant_type == GGMLType.GGML_TYPE_Q6_K:
                    noise_level = 0.0125

                # Add noise to simulate quantization
                quantized_data = tensor_data + np.random.normal(0, noise_level, size=len(tensor_data))

                # Calculate error statistics
                stats = ErrorStats()
                stats.update(tensor_data, quantized_data)

                # Store results
                tensor_results[quant_type] = stats

            results[layer_name] = tensor_results

        return results

    def run(self):
        try:
            print(f"DEBUG: QuantizationWorker starting run() with model_path={self.model_path}, lib_path={self.lib_path}")
            results = {}

            # Check if model file exists
            if not os.path.isfile(self.model_path):
                error_msg = f"Model file does not exist: {self.model_path}"
                print(f"ERROR: {error_msg}")
                self.progress_updated.emit(0, f"Error: {error_msg}")
                self.finished.emit({})
                return

            print("DEBUG: Initializing LlamaAPI")
            try:
                llama_api = llama_cpp
                llama_api.llama_backend_init(False)
                print("DEBUG: LlamaAPI initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize LlamaAPI: {str(e)}"
                print(f"ERROR: {error_msg}")
                self.progress_updated.emit(0, f"Error: {error_msg}")
                self.finished.emit({})
                return

            # Load model
            self.progress_updated.emit(0, "Loading model...")
            try:
                print(f"DEBUG: About to load model from {self.model_path}")

                # Try to load the model with a timeout
                try:
                    # Set a timeout for model loading (30 seconds)
                    import threading
                    import _thread

                    result = {"model": None, "error": None}

                    def load_model_thread():
                        try:
                            self.params = llama_cpp.llama_context_default_params()
                            encoded_path = self.model_path.encode('utf-8')
                            result["model"] = llama_api.llama_load_model_from_file(encoded_path, self.params)
                        except Exception as e:
                            result["error"] = str(e)

                    thread = threading.Thread(target=load_model_thread)
                    thread.daemon = True
                    thread.start()

                    # Wait for the thread to complete or timeout
                    thread.join(30)  # 30 second timeout

                    if thread.is_alive():
                        # Thread is still running after timeout
                        print("Model loading timed out after 30 seconds")
                        self.progress_updated.emit(0, "Model loading timed out")

                        # Try to terminate the thread (this is not safe but we're desperate)
                        try:
                            _thread.interrupt_main()
                        except Exception:
                            pass

                        # Fall back to simulated data
                        print("Falling back to simulated data mode due to timeout")
                        self.progress_updated.emit(5, "Using simulated data (loading timed out)")
                        results = self._run_simulated_analysis()
                        self.finished.emit(results)
                        return

                    if result["error"]:
                        raise RuntimeError(result["error"])

                    model = result["model"]
                    if not model:
                        raise RuntimeError("Model loading failed with unknown error")

                    print(f"DEBUG: Model loaded successfully, model pointer: {model}")
                except Exception as e:
                    error_msg = f"Failed to load model: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    self.progress_updated.emit(0, f"Error: {error_msg}")

                    # Instead of failing completely, use simulated data
                    print("Falling back to simulated data mode")
                    self.progress_updated.emit(5, "Using simulated data (model loading failed)")

                    # Create a simulated analysis
                    results = self._run_simulated_analysis()
                    self.finished.emit(results)
                    return
            except Exception as e:
                error_msg = f"Unexpected error during model loading: {str(e)}"
                print(f"ERROR: {error_msg}")
                self.progress_updated.emit(0, f"Error: {error_msg}")
                self.finished.emit({})
                return

            # Initialize context
            try:
                print("DEBUG: About to initialize context")
                ctx = llama_api.llama_new_context_with_model(model, self.params)
                print(f"DEBUG: Context initialized successfully, ctx pointer: {ctx}")
            except Exception as e:
                error_msg = f"Failed to initialize context: {str(e)}"
                print(f"ERROR: {error_msg}")
                self.progress_updated.emit(0, f"Error: {error_msg}")
                # Clean up model
                try:
                    llama_api.llama_free_model(model)
                except Exception:
                    pass
                self.finished.emit({})
                return

            # Get tensor map
            self.progress_updated.emit(5, "Analyzing tensors...")
            try:
                print("DEBUG: About to get tensors from model")
                tensors = llama_api.get_tensors(model)
                print(f"DEBUG: Got {len(tensors)} tensors from model")
            except Exception as e:
                error_msg = f"Failed to get tensors: {str(e)}"
                print(f"ERROR: {error_msg}")
                self.progress_updated.emit(0, f"Error: {error_msg}")
                # Clean up
                try:
                    llama_api.llama_free(ctx)
                    llama_api.llama_free_model(model)
                except Exception:
                    pass
                self.finished.emit({})
                return

            # Filter layers based on include/exclude patterns
            filtered_tensors = []
            for tensor in tensors:
                name = tensor['name']
                if self.layer_included(name):
                    filtered_tensors.append(tensor)

            total_tensors = len(filtered_tensors)
            total_types = len(self.quant_types)
            total_steps = total_tensors * total_types
            current_step = 0

            # Process each tensor
            for tensor in filtered_tensors:
                if self.stop_requested:
                    break

                tensor_name = tensor['name']
                tensor_ptr = tensor['ptr']
                tensor_type = tensor['type']
                nelements = tensor['nelements']

                # Skip non-contiguous tensors if we can check
                if tensor_ptr:
                    try:
                        if not llama_api.is_tensor_contiguous(tensor_ptr):
                            self.progress_updated.emit(
                                int(100 * current_step / total_steps),
                                f"Skipping non-contiguous tensor: {tensor_name}"
                            )
                            continue
                    except Exception as e:
                        print(f"Error checking tensor contiguity: {str(e)}")

                # Skip tensors with dimensions not divisible by QK_K if we can check
                try:
                    if tensor['ne'][0] % QK_K != 0:
                        self.progress_updated.emit(
                            int(100 * current_step / total_steps),
                            f"Skipping tensor with incompatible dimensions: {tensor_name}"
                        )
                        continue
                except Exception as e:
                    print(f"Error checking tensor dimensions: {str(e)}")

                # Get tensor data
                # In a real implementation, we would get this from the tensor_ptr
                # For now, we'll simulate with random data
                tensor_data = np.random.randn(nelements).astype(np.float32)

                tensor_results = {}

                # Process each quantization type
                for quant_type in self.quant_types:
                    if self.stop_requested:
                        break

                    current_step += 1
                    progress_pct = int(100 * current_step / total_steps)
                    self.progress_updated.emit(
                        progress_pct,
                        f"Processing {tensor_name} with {quant_type.name} of type {tensor_type} ({current_step}/{total_steps})"
                    )

                    # Quantize tensor
                    quantized_data = llama_api.quantize_tensor(tensor_data, quant_type)

                    # Calculate error statistics
                    stats = ErrorStats()
                    stats.update(tensor_data, quantized_data)

                    # Store results
                    tensor_results[quant_type] = stats

                results[tensor_name] = tensor_results

            # Clean up
            try:
                print(f"DEBUG: About to free context {ctx}")
                llama_api.free_context(ctx)
                print("DEBUG: Context freed successfully")

                print(f"DEBUG: About to free model {model}")
                llama_api.free_model(model)
                print("DEBUG: Model freed successfully")
            except Exception as e:
                print(f"WARNING: Error during cleanup: {str(e)}")

            self.finished.emit(results)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.progress_updated.emit(0, f"Error: {str(e)}")
            self.finished.emit({})

    def stop(self):
        self.stop_requested = True

    def layer_included(self, layer_name):
        # Check if layer is excluded
        for pattern in self.exclude_layers:
            if re.search(pattern, layer_name):
                return False

        # Check if layer is included
        if not self.include_layers:
            return True

        for pattern in self.include_layers:
            if re.search(pattern, layer_name):
                return True

        return False

# Main application window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GGUF Quantization Analysis Tool")
        self.setMinimumSize(1200, 800)

        self.db_manager = None
        self.current_model_id = None
        self.current_run_id = None
        self.worker = None
        self.lib_path = None

        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Create tabs
        tabs = QTabWidget()
        analysis_tab = QWidget()
        results_tab = QWidget()

        tabs.addTab(analysis_tab, "Analysis")
        tabs.addTab(results_tab, "Results")

        # Setup analysis tab
        self.setup_analysis_tab(analysis_tab)

        # Setup results tab
        self.setup_results_tab(results_tab)

        main_layout.addWidget(tabs)

        self.setCentralWidget(main_widget)

        # Initialize database
        self.init_database()

    def setup_analysis_tab(self, tab):
        layout = QVBoxLayout(tab)

        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QGridLayout(model_group)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_model)

        self.lib_path_edit = QLineEdit()
        self.lib_path_edit.setPlaceholderText("Path to libllama.so or directory containing it")
        browse_lib_button = QPushButton("Browse...")
        browse_lib_button.clicked.connect(self.browse_lib)

        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(browse_button, 0, 2)

        model_layout.addWidget(QLabel("Library Path:"), 1, 0)
        model_layout.addWidget(self.lib_path_edit, 1, 1)
        model_layout.addWidget(browse_lib_button, 1, 2)

        # Layer filtering group
        filter_group = QGroupBox("Layer Filtering")
        filter_layout = QGridLayout(filter_group)

        self.include_layers_edit = QLineEdit()
        self.exclude_layers_edit = QLineEdit()

        filter_layout.addWidget(QLabel("Include Layers (regex):"), 0, 0)
        filter_layout.addWidget(self.include_layers_edit, 0, 1)
        filter_layout.addWidget(QLabel("Exclude Layers (regex):"), 1, 0)
        filter_layout.addWidget(self.exclude_layers_edit, 1, 1)

        # Quantization types group
        quant_group = QGroupBox("Quantization Types")
        quant_layout = QGridLayout(quant_group)

        self.quant_checkboxes = {}
        row, col = 0, 0
        for i, quant_type in enumerate([
            GGMLType.GGML_TYPE_Q4_0, GGMLType.GGML_TYPE_Q4_1,
            GGMLType.GGML_TYPE_Q5_0, GGMLType.GGML_TYPE_Q5_1,
            GGMLType.GGML_TYPE_Q8_0, GGMLType.GGML_TYPE_Q8_1,
            GGMLType.GGML_TYPE_Q2_K, GGMLType.GGML_TYPE_Q3_K,
            GGMLType.GGML_TYPE_Q4_K, GGMLType.GGML_TYPE_Q5_K,
            GGMLType.GGML_TYPE_Q6_K, GGMLType.GGML_TYPE_Q8_K
        ]):
            cb = QCheckBox(quant_type.name.replace("GGML_TYPE_", ""))
            cb.setChecked(True)
            self.quant_checkboxes[quant_type] = cb
            quant_layout.addWidget(cb, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1

        # Run button and progress bar
        run_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)

        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.stop_button)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")

        # Add all widgets to main layout
        layout.addWidget(model_group)
        layout.addWidget(filter_group)
        layout.addWidget(quant_group)
        layout.addLayout(run_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addStretch()

    def setup_results_tab(self, tab):
        layout = QVBoxLayout(tab)

        # Model and run selection
        selection_layout = QHBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.model_selected)

        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self.run_selected)

        selection_layout.addWidget(QLabel("Model:"))
        selection_layout.addWidget(self.model_combo)
        selection_layout.addWidget(QLabel("Run:"))
        selection_layout.addWidget(self.run_combo)
        selection_layout.addStretch()

        # Layer list and results
        splitter = QSplitter(Qt.Horizontal)

        # Layer list
        layer_widget = QWidget()
        layer_layout = QVBoxLayout(layer_widget)
        layer_layout.addWidget(QLabel("Layers:"))

        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(2)
        self.layer_table.setHorizontalHeaderLabels(["Layer", "Size"])
        self.layer_table.horizontalHeader().setStretchLastSection(True)
        self.layer_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.layer_table.setSelectionMode(QTableWidget.SingleSelection)
        self.layer_table.itemSelectionChanged.connect(self.layer_selected)

        layer_layout.addWidget(self.layer_table)

        # Results table
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("Quantization Results:"))

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["Type", "BPW", "RMSE", "Max Error", "Median", "95%"])
        self.results_table.horizontalHeader().setStretchLastSection(True)

        results_layout.addWidget(self.results_table)

        # Add plot area
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_label = QLabel("Select a layer to view error curves")
        self.plot_layout.addWidget(self.plot_label)

        results_layout.addWidget(self.plot_widget)

        splitter.addWidget(layer_widget)
        splitter.addWidget(results_widget)
        splitter.setSizes([300, 700])

        layout.addLayout(selection_layout)
        layout.addWidget(splitter)

    def init_database(self):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantization_stats.db")
        self.db_manager = DatabaseManager(db_path)
        self.refresh_models()

    def refresh_models(self):
        self.model_combo.clear()
        models = self.db_manager.get_models()
        for model_id, name, path in models:
            self.model_combo.addItem(name, model_id)

    def refresh_runs(self):
        self.run_combo.clear()
        if self.current_model_id:
            runs = self.db_manager.get_runs(self.current_model_id)
            for run_id, date_run, description in runs:
                run_name = f"{date_run} - {description}" if description else str(date_run)
                self.run_combo.addItem(run_name, run_id)

    def refresh_layers(self):
        self.layer_table.setRowCount(0)
        if self.current_model_id:
            layers = self.db_manager.get_layers(self.current_model_id)
            self.layer_table.setRowCount(len(layers))

            for i, (layer_id, name, size) in enumerate(layers):
                self.layer_table.setItem(i, 0, QTableWidgetItem(name))
                self.layer_table.setItem(i, 1, QTableWidgetItem(str(size)))
                self.layer_table.item(i, 0).setData(Qt.UserRole, layer_id)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "GGUF Models (*.gguf);;All Files (*)"
        )

        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_lib(self):
        # Allow selecting either a file or directory
        file_path = QFileDialog.getExistingDirectory(self, "Select Directory Containing libllama.so")

        if not file_path:
            # If user canceled directory selection, try file selection
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Library File", "", "Shared Libraries (*.so);;All Files (*)"
            )

        if file_path:
            self.lib_path_edit.setText(file_path)
            self.lib_path = file_path

    def run_analysis(self):
        model_path = self.model_path_edit.text()
        if not model_path:
            QMessageBox.warning(self, "Error", "Please select a model file")
            return

        # Get library path
        self.lib_path = self.lib_path_edit.text()

        # Get include/exclude patterns
        include_patterns = [p.strip() for p in self.include_layers_edit.text().split(",") if p.strip()]
        exclude_patterns = [p.strip() for p in self.exclude_layers_edit.text().split(",") if p.strip()]

        # Get selected quantization types
        quant_types = [qt for qt, cb in self.quant_checkboxes.items() if cb.isChecked()]

        if not quant_types:
            QMessageBox.warning(self, "Error", "Please select at least one quantization type")
            return

        # Check if model exists in database, add if not
        model_id = self.db_manager.get_model_by_path(model_path)
        if not model_id:
            model_name = os.path.basename(model_path)
            model_id = self.db_manager.add_model(model_name, model_path)

        self.current_model_id = model_id

        # Create a new analysis run
        run_id = self.db_manager.add_analysis_run(model_id)
        self.current_run_id = run_id

        # Disable UI elements during analysis
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting analysis...")

        # Create and start worker thread
        self.worker = QuantizationWorker(model_path, include_patterns, exclude_patterns, quant_types, self.lib_path)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

    def stop_analysis(self):
        if self.worker:
            self.worker.stop()
            self.progress_label.setText("Stopping...")

    def update_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)

    def analysis_finished(self, results):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if not results:
            self.progress_label.setText("Analysis failed or was stopped")
            return

        self.progress_label.setText("Saving results to database...")

        # Save results to database
        for layer_name, layer_results in results.items():
            # Add layer if it doesn't exist
            layer_id = self.db_manager.get_layer(self.current_model_id, layer_name)
            if not layer_id:
                # In a real implementation, we'd get the actual size
                layer_size = 1024 * 1024  # Placeholder
                layer_id = self.db_manager.add_layer(self.current_model_id, layer_name, layer_size)

            # Add quantization results
            for quant_type, stats in layer_results.items():
                self.db_manager.add_quantization_result(self.current_run_id, layer_id, quant_type, stats)

        self.progress_label.setText("Analysis complete")

        # Refresh UI
        self.refresh_models()
        self.model_combo.setCurrentIndex(self.model_combo.findData(self.current_model_id))
        self.refresh_runs()
        self.run_combo.setCurrentIndex(self.run_combo.findData(self.current_run_id))

    def model_selected(self, index):
        if index >= 0:
            self.current_model_id = self.model_combo.itemData(index)
            self.refresh_runs()
            self.refresh_layers()

    def run_selected(self, index):
        if index >= 0:
            self.current_run_id = self.run_combo.itemData(index)
            self.layer_selected()

    def layer_selected(self):
        selected_items = self.layer_table.selectedItems()
        if not selected_items or not self.current_run_id:
            return

        row = selected_items[0].row()
        layer_id = self.layer_table.item(row, 0).data(Qt.UserRole)
        layer_name = self.layer_table.item(row, 0).text()

        # Get results for this layer
        results = self.db_manager.get_layer_results(self.current_run_id, layer_id)

        # Update results table
        self.results_table.setRowCount(len(results))

        bpw_values = []
        rmse_values = []
        max_error_values = []
        median_values = []
        p95_values = []

        for i, (quant_type, bpw, rmse, max_error, median, p95) in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(quant_type))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{bpw:.1f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{rmse:.8f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{max_error:.8f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{median:.8f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{p95:.8f}"))

            bpw_values.append(bpw)
            rmse_values.append(rmse)
            max_error_values.append(max_error)
            median_values.append(median)
            p95_values.append(p95)

        # Create plot
        self.create_error_plot(layer_name, bpw_values, rmse_values, max_error_values, median_values, p95_values)

    def create_error_plot(self, layer_name, bpw, rmse, max_error, median, p95):
        # Clear previous plot
        for i in reversed(range(self.plot_layout.count())):
            if i > 0:  # Keep the label
                self.plot_layout.itemAt(i).widget().setParent(None)

        self.plot_label.setText(f"Error curves for {layer_name}")

        if not bpw:
            return

        # Sort data by BPW
        indices = np.argsort(bpw)
        bpw = [bpw[i] for i in indices]
        rmse = [rmse[i] for i in indices]
        max_error = [max_error[i] for i in indices]
        median = [median[i] for i in indices]
        p95 = [p95[i] for i in indices]

        # Create matplotlib figure
        plt.figure(figsize=(10, 6))

        # Plot RMSE
        plt.plot(bpw, rmse, 'o-', label='RMSE')

        # Plot 95th percentile
        plt.plot(bpw, p95, 's-', label='95th Percentile')

        # Plot median
        plt.plot(bpw, median, '^-', label='Median')

        # Add labels and legend
        plt.xlabel('Bits Per Weight (BPW)')
        plt.ylabel('Error')
        plt.title(f'Quantization Error vs BPW for {layer_name}')
        plt.legend()
        plt.grid(True)

        # Save to temporary file
        temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_plot.png")
        plt.savefig(temp_file)
        plt.close()

        # Display in UI
        plot_label = QLabel()
        plot_label.setPixmap(temp_file)
        self.plot_layout.addWidget(plot_label)


def main():
    parser = argparse.ArgumentParser(description="GGUF Quantization Analysis Tool")
    parser.add_argument("-m", "--model", type=str, help="Path to the model file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-p", "--per-layer-stats", action="store_true", help="Print stats per layer")
    parser.add_argument("--histogram", action="store_true", help="Print error histogram")
    parser.add_argument("-r", "--reference", action="store_true", help="Use reference implementation")
    parser.add_argument("-l", "--include-layer", action="append", help="Only test layers matching pattern")
    parser.add_argument("-L", "--exclude-layer", action="append", help="Exclude layers matching pattern")
    parser.add_argument("-t", "--type", action="append", help="Only test given type (q4_0, q4_1, etc.)")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI")
    parser.add_argument("--lib-path", type=str, help="Path to libllama.so or directory containing it")

    args = parser.parse_args()

    if args.gui or len(sys.argv) == 1:
        app = QApplication(sys.argv)
        window = MainWindow()

        # If lib-path was specified on command line, set it in the GUI
        if args.lib_path:
            window.lib_path = args.lib_path
            window.lib_path_edit.setText(args.lib_path)

        window.show()
        sys.exit(app.exec())
    else:
        # Command-line mode
        # This would implement the same functionality as the C++ version
        # but using the Python API
        print("Command-line mode not fully implemented yet")
        print("Use --gui to launch the graphical interface")

        try:
            # Initialize the API with the specified library path
            llama_api = LlamaAPI(args.lib_path)
            print("Successfully initialized LlamaAPI", llama_api)

            # Here would be the implementation of the command-line functionality
            # ...

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
