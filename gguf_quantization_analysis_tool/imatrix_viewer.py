#!/usr/bin/env python3
import sys
import os
import struct
import numpy as np
# import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    # QComboBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QTabWidget)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatrixCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatrixCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.current_matrix = None
        self.current_name = None

    def plot_matrix(self, values, name):
        self.axes.clear()
        self.current_matrix = values
        self.current_name = name

        # Reshape if needed for visualization
        if len(values.shape) == 1:
            # For 1D data, plot as a line
            self.axes.plot(values)
            self.axes.set_title(f"Activation values for {name}")
            self.axes.set_xlabel("Neuron index")
            self.axes.set_ylabel("Importance value")
        else:
            # For 2D data (e.g., MoE experts), plot as heatmap
            im = self.axes.imshow(values, aspect='auto', cmap='viridis')
            self.axes.set_title(f"Activation values for {name}")
            self.axes.set_xlabel("Neuron index")
            self.axes.set_ylabel("Expert index")
            self.fig.colorbar(im)

        self.fig.tight_layout()
        self.draw()


class IMatrixViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Importance Matrix Viewer")
        self.setGeometry(100, 100, 1200, 800)

        self.imatrix_data = None
        self.input_filename = None
        self.num_calls = 0

        self.setup_ui()

    def setup_ui(self):
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top controls
        top_layout = QHBoxLayout()
        self.load_button = QPushButton("Load iMatrix File")
        self.load_button.clicked.connect(self.load_imatrix)
        self.file_label = QLabel("No file loaded")

        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.file_label)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)

        # Splitter for tree and visualization
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - tree view of matrices
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Layer Name", "Shape"])
        self.tree.itemClicked.connect(self.on_item_selected)
        splitter.addWidget(self.tree)

        # Right panel - visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        # Tabs for different visualizations
        self.tabs = QTabWidget()

        # Matrix visualization tab
        matrix_tab = QWidget()
        matrix_layout = QVBoxLayout(matrix_tab)
        self.matrix_canvas = MatrixCanvas(matrix_tab, width=5, height=4)
        matrix_layout.addWidget(self.matrix_canvas)

        # Stats tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_label = QLabel("No statistics available")
        stats_layout.addWidget(self.stats_label)

        self.tabs.addTab(matrix_tab, "Visualization")
        self.tabs.addTab(stats_tab, "Statistics")

        viz_layout.addWidget(self.tabs)
        splitter.addWidget(viz_widget)

        # Set initial sizes
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

    def load_imatrix(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open iMatrix File", "", "iMatrix Files (*.dat);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.imatrix_data = self.read_imatrix_file(filename)
            self.file_label.setText(f"Loaded: {os.path.basename(filename)}")
            self.populate_tree()
            self.statusBar().showMessage(f"Loaded {len(self.imatrix_data)} matrix entries")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {str(e)}")

    def read_imatrix_file(self, filename):
        with open(filename, 'rb') as f:
            # Read number of entries
            n_entries = struct.unpack('i', f.read(4))[0]

            data = {}

            # Read each entry
            for _ in range(n_entries):
                # Read name
                name_len = struct.unpack('i', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')

                # Read ncall
                ncall = struct.unpack('i', f.read(4))[0]

                # Read values
                nval = struct.unpack('i', f.read(4))[0]
                values = np.frombuffer(f.read(nval * 4), dtype=np.float32)

                # Store the data
                data[name] = {
                    'ncall': ncall,
                    'values': values
                }

            # Read the number of calls the matrix was computed with
            self.num_calls = struct.unpack('i', f.read(4))[0]

            # Read the input filename
            input_filename_len = struct.unpack('i', f.read(4))[0]
            self.input_filename = f.read(input_filename_len).decode('utf-8')

            return data

    def populate_tree(self):
        self.tree.clear()

        # Group by block
        blocks = {}
        others = []

        for name, data in self.imatrix_data.items():
            if name.startswith('blk.'):
                parts = name.split('.')
                if len(parts) >= 2:
                    block_num = parts[1]
                    if block_num not in blocks:
                        blocks[block_num] = []
                    blocks[block_num].append((name, data))
            else:
                others.append((name, data))

        # Add blocks to tree
        for block_num in sorted(blocks.keys(), key=lambda x: int(x)):
            block_item = QTreeWidgetItem(self.tree)
            block_item.setText(0, f"Block {block_num}")

            for name, data in sorted(blocks[block_num]):
                item = QTreeWidgetItem(block_item)
                item.setText(0, name)
                item.setText(1, f"{data['values'].shape}")
                item.setData(0, Qt.UserRole, name)

        # Add other entries
        if others:
            other_item = QTreeWidgetItem(self.tree)
            other_item.setText(0, "Other")

            for name, data in sorted(others):
                item = QTreeWidgetItem(other_item)
                item.setText(0, name)
                item.setText(1, f"{data['values'].shape}")
                item.setData(0, Qt.UserRole, name)

        self.tree.expandAll()

    def on_item_selected(self, item, column):
        name = item.data(0, Qt.UserRole)
        if not name or name not in self.imatrix_data:
            return

        data = self.imatrix_data[name]
        values = data['values']

        # Update visualization
        self.matrix_canvas.plot_matrix(values, name)

        # Update stats
        stats_text = (
            f"Layer: {name}\n"
            f"Shape: {values.shape}\n"
            f"Number of calls: {data['ncall']}\n"
            f"Min value: {values.min():.6f}\n"
            f"Max value: {values.max():.6f}\n"
            f"Mean value: {values.mean():.6f}\n"
            f"Std dev: {values.std():.6f}\n"
        )
        self.stats_label.setText(stats_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IMatrixViewer()
    window.show()
    sys.exit(app.exec())
