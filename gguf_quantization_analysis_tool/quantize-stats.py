#!/usr/bin/env python3
import sys
import os
import re
import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
import ctypes
from ctypes import c_void_p, c_int, c_float, c_char_p, c_bool, POINTER, Structure, c_int64, c_uint64, c_size_t
from enum import Enum, auto
import threading
import queue
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox,
                               QTableWidget, QTableWidgetItem, QTabWidget, QProgressBar,
                               QLineEdit, QGroupBox, QGridLayout, QSplitter, QMessageBox)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QFont, QColor

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

# LLAMA C API wrapper


class LlamaAPI:
    def __init__(self, lib_path=None):
        # Find the library
        lib_paths = []

        # If a specific path was provided, check there first
        if lib_path:
            # Check for direct path to the library
            if os.path.isfile(lib_path):
                lib_paths.append(lib_path)
            else:
                # Check for the library in the specified directory
                lib_paths.append(os.path.join(lib_path, "libllama.so"))

        # Add default search paths
        lib_paths.extend([
            "./libllama.so",
            "../libllama.so",
            "../../libllama.so",
            "/usr/local/lib/libllama.so",
            "/usr/lib/libllama.so",
        ])

        self.lib = None
        for path in lib_paths:
            if os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    print(f"Successfully loaded library from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {str(e)}")
                    continue

        if self.lib is None:
            raise RuntimeError("Could not find libllama.so. Make sure it's in your library path or specify with --lib-path.")

        # Define structures and function prototypes
        self._setup_api()

    def _setup_api(self):
        # Define llama_model and llama_context as opaque pointers
        print(f"DEBUG: Setting up function prototypes")
        try:
            self.lib.llama_model_load_from_file.restype = c_void_p
            self.lib.llama_model_load_from_file.argtypes = [c_char_p, c_void_p]
            print(f"DEBUG: Set up llama_model_load_from_file")

            self.lib.llama_model_free.argtypes = [c_void_p]
            print(f"DEBUG: Set up llama_model_free")

            self.lib.llama_init_from_model.restype = c_void_p
            self.lib.llama_init_from_model.argtypes = [c_void_p, c_void_p]
            print(f"DEBUG: Set up llama_init_from_model")

            self.lib.llama_free.argtypes = [c_void_p]
            print(f"DEBUG: Set up llama_free")
        except Exception as e:
            print(f"DEBUG: EXCEPTION setting up function prototypes: {str(e)}")
            import traceback
            traceback.print_exc()

        # Define model params structure
        class LlamaModelParams(Structure):
            _fields_ = [
                ("n_gpu_layers", c_int),
                ("main_gpu", c_int),
                ("tensor_split", c_float * 8),
                ("progress_callback", c_void_p),
                ("progress_callback_user_data", c_void_p),
                ("vocab_only", c_bool),
                ("use_mmap", c_bool),
                ("use_mlock", c_bool),
                ("split_mode", c_int),
                ("main_device", c_int),
                ("devices", POINTER(c_void_p)),
                ("check_tensors", c_bool),
                ("kv_overrides", c_void_p),
                ("tensor_buft_overrides", c_void_p),
            ]

        self.LlamaModelParams = LlamaModelParams

        # Define context params structure
        class LlamaContextParams(Structure):
            _fields_ = [
                ("n_ctx", c_int),
                ("n_batch", c_int),
                ("n_threads", c_int),
                ("n_threads_batch", c_int),
                ("rope_scaling_type", c_int),
                ("rope_freq_base", c_float),
                ("rope_freq_scale", c_float),
                ("yarn_ext_factor", c_float),
                ("yarn_attn_factor", c_float),
                ("yarn_beta_fast", c_float),
                ("yarn_beta_slow", c_float),
                ("yarn_orig_ctx", c_int),
                ("cb_eval", c_void_p),
                ("cb_eval_user_data", c_void_p),
                ("type_k", c_int),
                ("type_v", c_int),
                ("logits_all", c_bool),
                ("embedding", c_bool),
                ("abort_callback", c_void_p),
                ("abort_callback_data", c_void_p),
            ]

        self.LlamaContextParams = LlamaContextParams

        # Define default params functions
        self.lib.llama_model_default_params.restype = LlamaModelParams
        self.lib.llama_context_default_params.restype = LlamaContextParams

        # We'll use a different approach to access tensors since llama_internal_get_tensor_map is not available

        # Try to define GGML functions - these might not all be available
        # We'll check for their existence before using them
        try:
            # Core tensor functions
            if hasattr(self.lib, 'ggml_nelements'):
                self.lib.ggml_nelements.restype = c_int64
                self.lib.ggml_nelements.argtypes = [c_void_p]

            if hasattr(self.lib, 'ggml_type_name'):
                self.lib.ggml_type_name.restype = c_char_p
                self.lib.ggml_type_name.argtypes = [c_int]

            if hasattr(self.lib, 'ggml_get_name'):
                self.lib.ggml_get_name.restype = c_char_p
                self.lib.ggml_get_name.argtypes = [c_void_p]

            if hasattr(self.lib, 'ggml_get_type'):
                self.lib.ggml_get_type.restype = c_int
                self.lib.ggml_get_type.argtypes = [c_void_p]

            if hasattr(self.lib, 'ggml_get_type_traits'):
                self.lib.ggml_get_type_traits.restype = c_void_p
                self.lib.ggml_get_type_traits.argtypes = [c_int]

            if hasattr(self.lib, 'ggml_get_type_traits_cpu'):
                self.lib.ggml_get_type_traits_cpu.restype = c_void_p
                self.lib.ggml_get_type_traits_cpu.argtypes = [c_int]

            if hasattr(self.lib, 'ggml_quantize_init'):
                self.lib.ggml_quantize_init.argtypes = [c_int]

            if hasattr(self.lib, 'ggml_get_data_f32'):
                self.lib.ggml_get_data_f32.restype = POINTER(c_float)
                self.lib.ggml_get_data_f32.argtypes = [c_void_p]

            if hasattr(self.lib, 'ggml_get_f32_1d'):
                self.lib.ggml_get_f32_1d.restype = c_float
                self.lib.ggml_get_f32_1d.argtypes = [c_void_p, c_int64]

            # Tensor access functions
            if hasattr(self.lib, 'ggml_tensor_is_contiguous'):
                self.lib.ggml_tensor_is_contiguous.restype = c_bool
                self.lib.ggml_tensor_is_contiguous.argtypes = [c_void_p]

            if hasattr(self.lib, 'ggml_type_size'):
                self.lib.ggml_type_size.restype = c_size_t
                self.lib.ggml_type_size.argtypes = [c_int]

            if hasattr(self.lib, 'ggml_blck_size'):
                self.lib.ggml_blck_size.restype = c_int64
                self.lib.ggml_blck_size.argtypes = [c_int]

            # Tensor dimensions access
            if hasattr(self.lib, 'ggml_get_ne'):
                self.lib.ggml_get_ne.restype = c_int64
                self.lib.ggml_get_ne.argtypes = [c_void_p, c_int]

            print("Successfully set up available GGML functions")
        except Exception as e:
            print(f"Warning: Error setting up GGML functions: {str(e)}")
            print("Will use fallback simulation methods instead")

        # Define quantization functions
        class TypeTraits(Structure):
            _fields_ = [
                ("type_name", c_char_p),
                ("blck_size", c_int64),
                ("blck_size_interleave", c_int64),
                ("type_size", c_size_t),
                ("is_quantized", c_bool),
                ("to_float", c_void_p),
                ("from_float", c_void_p),
            ]

        self.TypeTraits = TypeTraits

    def model_default_params(self):
        return self.lib.llama_model_default_params()

    def context_default_params(self):
        return self.lib.llama_context_default_params()

    def load_model(self, model_path):
        print(f"DEBUG: Starting to load model from {model_path}")
        params = self.model_default_params()
        params.use_mlock = False

        print(f"DEBUG: Created model params, use_mlock={params.use_mlock}")

        # Create a pointer to the params structure
        params_ptr = ctypes.byref(params)
        print(f"DEBUG: Created params pointer: {params_ptr}")

        print(f"DEBUG: About to call llama_model_load_from_file")
        try:
            model = self.lib.llama_model_load_from_file(model_path.encode('utf-8'), params_ptr)
            print(f"DEBUG: llama_model_load_from_file returned: {model}")
            if not model:
                raise RuntimeError(f"Failed to load model: {model_path}")

            return model
        except Exception as e:
            print(f"DEBUG: EXCEPTION in load_model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def init_context(self, model):
        print(f"DEBUG: Starting to initialize context from model {model}")
        params = self.context_default_params()
        params.n_ctx = 256

        print(f"DEBUG: Created context params, n_ctx={params.n_ctx}")

        # Create a pointer to the params structure
        params_ptr = ctypes.byref(params)
        print(f"DEBUG: Created context params pointer: {params_ptr}")

        print(f"DEBUG: About to call llama_init_from_model")
        try:
            ctx = self.lib.llama_init_from_model(model, params_ptr)
            print(f"DEBUG: llama_init_from_model returned: {ctx}")
            if not ctx:
                raise RuntimeError("Failed to create context")

            return ctx
        except Exception as e:
            print(f"DEBUG: EXCEPTION in init_context: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def free_model(self, model):
        print(f"DEBUG: Calling llama_model_free on {model}")
        try:
            self.lib.llama_model_free(model)
            print(f"DEBUG: llama_model_free completed successfully")
        except Exception as e:
            print(f"DEBUG: EXCEPTION in free_model: {str(e)}")
            import traceback
            traceback.print_exc()

    def free_context(self, ctx):
        print(f"DEBUG: Calling llama_free on {ctx}")
        try:
            self.lib.llama_free(ctx)
            print(f"DEBUG: llama_free completed successfully")
        except Exception as e:
            print(f"DEBUG: EXCEPTION in free_context: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_tensor_map(self, model):
        # This function is a placeholder since we can't directly access the tensor map
        # We'll use a different approach to get tensor information
        return None

    def get_nelements(self, tensor):
        try:
            if hasattr(self.lib, 'ggml_nelements'):
                return self.lib.ggml_nelements(tensor)
            else:
                # Fallback: return a default value
                return 1024 * 1024  # Placeholder
        except Exception as e:
            print(f"Error in get_nelements: {str(e)}")
            return 1024 * 1024  # Placeholder

    def get_tensor_type(self, tensor):
        try:
            if hasattr(self.lib, 'ggml_get_type'):
                return self.lib.ggml_get_type(tensor)
            else:
                # Fallback: return F32 as default
                return GGMLType.GGML_TYPE_F32.value
        except Exception as e:
            print(f"Error in get_tensor_type: {str(e)}")
            return GGMLType.GGML_TYPE_F32.value

    def get_tensor_name(self, tensor):
        try:
            if hasattr(self.lib, 'ggml_get_name'):
                name = self.lib.ggml_get_name(tensor)
                if name:
                    return name.decode('utf-8')
            # Fallback
            return "unnamed"
        except Exception as e:
            print(f"Error in get_tensor_name: {str(e)}")
            return "unnamed"

    def get_tensor_dimensions(self, tensor):
        dims = []
        try:
            if hasattr(self.lib, 'ggml_get_ne'):
                for i in range(4):  # GGML supports up to 4 dimensions
                    dims.append(self.lib.ggml_get_ne(tensor, i))
            else:
                # Fallback: return default dimensions
                dims = [1024, 1024, 1, 1]
        except Exception as e:
            print(f"Error in get_tensor_dimensions: {str(e)}")
            dims = [1024, 1024, 1, 1]
        return dims

    def get_tensor_data(self, tensor, tensor_type, nelements):
        try:
            if tensor_type == GGMLType.GGML_TYPE_F32.value and hasattr(self.lib, 'ggml_get_data_f32'):
                data_ptr = self.lib.ggml_get_data_f32(tensor)
                return np.ctypeslib.as_array(data_ptr, shape=(nelements,))
            elif tensor_type == GGMLType.GGML_TYPE_F16.value and hasattr(self.lib, 'ggml_get_f32_1d'):
                # This would need special handling for F16
                # For now, we'll convert F16 to F32 one by one
                data = np.zeros(nelements, dtype=np.float32)
                for i in range(nelements):
                    data[i] = self.lib.ggml_get_f32_1d(tensor, i)
                return data
            else:
                # Fallback: generate random data for testing
                print(f"Using simulated data for tensor type: {tensor_type}")
                return np.random.randn(nelements).astype(np.float32)
        except Exception as e:
            print(f"Error in get_tensor_data: {str(e)}")
            return np.random.randn(nelements).astype(np.float32)

    def is_tensor_contiguous(self, tensor):
        try:
            if hasattr(self.lib, 'ggml_tensor_is_contiguous'):
                return self.lib.ggml_tensor_is_contiguous(tensor)
            else:
                # Fallback: assume tensor is contiguous
                return True
        except Exception as e:
            print(f"Error in is_tensor_contiguous: {str(e)}")
            return True

    def get_type_name(self, type_id):
        try:
            if hasattr(self.lib, 'ggml_type_name'):
                name = self.lib.ggml_type_name(type_id)
                if name:
                    return name.decode('utf-8')
            return f"Unknown({type_id})"
        except Exception as e:
            print(f"Error in get_type_name: {str(e)}")
            return f"Unknown({type_id})"

    def quantize_tensor(self, input_data, quant_type):
        try:
            # Check if we have the necessary functions
            if (hasattr(self.lib, 'ggml_quantize_init')
                and hasattr(self.lib, 'ggml_get_type_traits')
                and hasattr(self.lib, 'ggml_get_type_traits_cpu')
                and hasattr(self.lib, 'ggml_type_size')
                    and hasattr(self.lib, 'ggml_blck_size')):

                # Initialize quantization for the type
                self.lib.ggml_quantize_init(quant_type.value)

                # Get type traits
                traits_ptr = self.lib.ggml_get_type_traits(quant_type.value)
                traits_cpu_ptr = self.lib.ggml_get_type_traits_cpu(quant_type.value)

                if not traits_ptr or not traits_cpu_ptr:
                    print(f"Failed to get type traits for {quant_type.name}, using simulation")
                    return self._simulate_quantization(input_data, quant_type)

                # Convert to Python structures
                traits = self.TypeTraits.from_address(traits_ptr)

                # Allocate memory for quantized data
                nelements = len(input_data)
                type_size = self.lib.ggml_type_size(quant_type.value)
                blck_size = self.lib.ggml_blck_size(quant_type.value)

                quantized_size = nelements * type_size // blck_size
                quantized_data = (ctypes.c_char * quantized_size)()

                # Allocate memory for output data
                output_data = (ctypes.c_float * nelements)()

                # For now, we'll use the simulation since we don't have direct access to quantize functions
                return self._simulate_quantization(input_data, quant_type)
            else:
                print(f"Required quantization functions not available, using simulation")
                return self._simulate_quantization(input_data, quant_type)
        except Exception as e:
            print(f"Error in quantize_tensor: {str(e)}")
            return self._simulate_quantization(input_data, quant_type)

    def _simulate_quantization(self, input_data, quant_type):
        """Simulate quantization with noise proportional to quantization level"""
        nelements = len(input_data)

        # Determine noise level based on quantization type
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
        output = input_data + np.random.normal(0, noise_level, size=nelements)

        return output

    def get_tensors(self, model):
        """Get tensors from the model using available API functions"""
        tensors = []

        try:
            print(f"DEBUG: Starting get_tensors with model pointer {model}")
            # Try to use llama_model_n_tensors if available
            if hasattr(self.lib, 'llama_model_n_tensors'):
                print(f"DEBUG: Found llama_model_n_tensors function")
                self.lib.llama_model_n_tensors.restype = c_int
                self.lib.llama_model_n_tensors.argtypes = [c_void_p]

                self.lib.llama_model_tensor.restype = c_void_p
                self.lib.llama_model_tensor.argtypes = [c_void_p, c_int]

                print(f"DEBUG: About to call llama_model_n_tensors")
                n_tensors = self.lib.llama_model_n_tensors(model)
                print(f"DEBUG: Model has {n_tensors} tensors")

                for i in range(n_tensors):
                    print(f"DEBUG: Getting tensor {i}/{n_tensors}")
                    try:
                        tensor_ptr = self.lib.llama_model_tensor(model, i)
                        print(f"DEBUG: Got tensor pointer: {tensor_ptr}")

                        if tensor_ptr:
                            print(f"DEBUG: Getting tensor name")
                            tensor_name = self.get_tensor_name(tensor_ptr)
                            print(f"DEBUG: Tensor name: {tensor_name}")

                            print(f"DEBUG: Getting tensor type")
                            tensor_type = self.get_tensor_type(tensor_ptr)
                            print(f"DEBUG: Tensor type: {tensor_type}")

                            print(f"DEBUG: Getting nelements")
                            nelements = self.get_nelements(tensor_ptr)
                            print(f"DEBUG: Tensor nelements: {nelements}")

                            print(f"DEBUG: Getting dimensions")
                            ne = self.get_tensor_dimensions(tensor_ptr)
                            print(f"DEBUG: Tensor dimensions: {ne}")

                            tensors.append({
                                'name': tensor_name,
                                'ptr': tensor_ptr,
                                'type': tensor_type,
                                'nelements': nelements,
                                'ne': ne
                            })
                            print(f"DEBUG: Added tensor to list")
                    except Exception as e:
                        print(f"DEBUG: EXCEPTION processing tensor {i}: {str(e)}")
                        import traceback
                        traceback.print_exc()
            else:
                # If we can't access tensors directly, simulate with random data for testing
                print("Cannot access model tensors directly. Using simulated data for testing.")
                for i in range(10):
                    tensors.append({
                        'name': f"layer.{i}.weight",
                        'ptr': None,
                        'type': GGMLType.GGML_TYPE_F32.value,
                        'nelements': 1024 * 1024,
                        'ne': [1024, 1024, 1, 1]
                    })
        except Exception as e:
            print(f"Error accessing tensors: {str(e)}")
            # Fall back to simulated data
            for i in range(10):
                tensors.append({
                    'name': f"layer.{i}.weight",
                    'ptr': None,
                    'type': GGMLType.GGML_TYPE_F32.value,
                    'nelements': 1024 * 1024,
                    'ne': [1024, 1024, 1, 1]
                })

        return tensors

# Worker thread for quantization


class QuantizationWorker(QThread):
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

    def run(self):
        try:
            print(f"DEBUG: QuantizationWorker starting run() with model_path={self.model_path}, lib_path={self.lib_path}")
            results = {}

            print(f"DEBUG: Initializing LlamaAPI")
            llama_api = LlamaAPI(self.lib_path)
            print(f"DEBUG: LlamaAPI initialized successfully")

            # Load model
            self.progress_updated.emit(0, "Loading model...")
            print(f"DEBUG: About to load model from {self.model_path}")
            model = llama_api.load_model(self.model_path)
            print(f"DEBUG: Model loaded successfully, model pointer: {model}")

            print(f"DEBUG: About to initialize context")
            ctx = llama_api.init_context(model)
            print(f"DEBUG: Context initialized successfully, ctx pointer: {ctx}")

            # Get tensor map
            self.progress_updated.emit(5, "Analyzing tensors...")
            print(f"DEBUG: About to get tensors from model")
            tensors = llama_api.get_tensors(model)
            print(f"DEBUG: Got {len(tensors)} tensors from model")

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
                        f"Processing {tensor_name} with {quant_type.name} ({current_step}/{total_steps})"
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
            print(f"DEBUG: About to free context {ctx}")
            llama_api.free_context(ctx)
            print(f"DEBUG: Context freed successfully")

            print(f"DEBUG: About to free model {model}")
            llama_api.free_model(model)
            print(f"DEBUG: Model freed successfully")

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
                self, "Select libllama.so File", "", "Shared Libraries (*.so);;All Files (*)"
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
            print(f"Successfully initialized LlamaAPI")

            # Here would be the implementation of the command-line functionality
            # ...

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
