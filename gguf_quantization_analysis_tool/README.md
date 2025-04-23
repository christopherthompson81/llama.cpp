# GGUF Quantization Analysis Tool

This tool helps analyze the impact of different quantization methods on GGUF model tensors. It allows you to:

1. Analyze how different quantization types affect model weights
2. Track error metrics (RMSE, max error, median, 95th percentile)
3. Visualize error curves as bits-per-weight (BPW) decreases
4. Store results in a SQLite database for later analysis
5. Identify the minimum BPW before quality degrades significantly

## Requirements

- Python 3.8+
- PySide6
- NumPy
- Matplotlib
- Access to libllama.so (from llama.cpp)

## Installation

```bash
pip install pyside6 numpy matplotlib
```

Make sure `libllama.so` is in your library path or in one of the following locations:
- ./libllama.so
- ../libllama.so
- ../../libllama.so
- /usr/local/lib/libllama.so
- /usr/lib/libllama.so

## Usage

### GUI Mode

```bash
python quantize-stats.py --gui
```

Or simply:

```bash
python quantize-stats.py
```

### Command Line Mode

```bash
python quantize-stats.py -m /path/to/model.gguf [options]
```

Options:
- `-m, --model`: Path to the model file
- `-v, --verbose`: Verbose output
- `-p, --per-layer-stats`: Print stats per layer
- `--histogram`: Print error histogram
- `-r, --reference`: Use reference implementation
- `-l, --include-layer`: Only test layers matching pattern
- `-L, --exclude-layer`: Exclude layers matching pattern
- `-t, --type`: Only test given type (q4_0, q4_1, etc.)

## Database

Results are stored in `quantization_stats.db` in the same directory as the script. This SQLite database contains:

- Models
- Analysis runs
- Layers
- Quantization results with error metrics

## How It Works

1. The tool loads a GGUF model using the llama.cpp library
2. It extracts tensor data from the model
3. For each tensor, it applies different quantization methods
4. It calculates error metrics by comparing original and quantized values
5. Results are stored in the database and displayed in the GUI

## Finding Optimal Quantization

The error curves help identify the "elbow point" where reducing BPW further would cause disproportionate quality loss. This can guide you in choosing the optimal quantization method for each tensor in your model.
