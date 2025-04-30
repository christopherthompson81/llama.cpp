#include "ggml.h"
#include "llama.h"
#include "common.h"

#include "../src/llama-model.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <regex>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

// Qt includes
#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QFileDialog>
#include <QCheckBox>
#include <QProgressBar>
#include <QComboBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QTabWidget>
#include <QGroupBox>
#include <QSpinBox>
#include <QTextEdit>
#include <QMessageBox>
#include <QChart>
#include <QChartView>
#include <QBarSeries>
#include <QBarSet>
#include <QBarCategoryAxis>
#include <QValueAxis>
#include <QDateTime>
#include <QThread>
#include <QListWidget>

// SQLite
#include <sqlite3.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr size_t HISTOGRAM_BUCKETS = 150;
constexpr double HISTOGRAM_RANGE = 0.03;

struct error_stats {
    size_t num_samples;
    double total_error;
    double max_error;
    uint64_t error_histogram[HISTOGRAM_BUCKETS];

    error_stats() : num_samples(0), total_error(0), max_error(0) {
        memset(error_histogram, 0, sizeof(error_histogram));
    }
};

// Database management
class DatabaseManager {
private:
    sqlite3* db;
    bool initialized;

    bool execute_query(const std::string& query) {
        char* error_message = nullptr;
        int rc = sqlite3_exec(db, query.c_str(), nullptr, nullptr, &error_message);
        
        if (rc != SQLITE_OK) {
            fprintf(stderr, "SQL error: %s\n", error_message);
            sqlite3_free(error_message);
            return false;
        }
        return true;
    }

public:
    DatabaseManager() : db(nullptr), initialized(false) {}

    ~DatabaseManager() {
        if (db) {
            sqlite3_close(db);
        }
    }

    bool init(const std::string& db_path) {
        int rc = sqlite3_open(db_path.c_str(), &db);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
            return false;
        }

        // Create tables if they don't exist
        const char* create_runs_table = 
            "CREATE TABLE IF NOT EXISTS runs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "model_path TEXT,"
            "timestamp TEXT,"
            "description TEXT"
            ");";

        const char* create_stats_table = 
            "CREATE TABLE IF NOT EXISTS quantization_stats ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "run_id INTEGER,"
            "layer_name TEXT,"
            "quant_type TEXT,"
            "rmse REAL,"
            "max_error REAL,"
            "median_error REAL,"
            "p95_error REAL,"
            "num_samples INTEGER,"
            "FOREIGN KEY(run_id) REFERENCES runs(id)"
            ");";

        const char* create_histogram_table = 
            "CREATE TABLE IF NOT EXISTS error_histograms ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "stats_id INTEGER,"
            "bucket_index INTEGER,"
            "bucket_value INTEGER,"
            "FOREIGN KEY(stats_id) REFERENCES quantization_stats(id)"
            ");";

        if (!execute_query(create_runs_table) || 
            !execute_query(create_stats_table) || 
            !execute_query(create_histogram_table)) {
            return false;
        }

        initialized = true;
        return true;
    }

    int64_t start_new_run(const std::string& model_path, const std::string& description) {
        if (!initialized) return -1;

        // Get current timestamp
        auto now = QDateTime::currentDateTime();
        std::string timestamp = now.toString(Qt::ISODate).toStdString();

        // Prepare the SQL statement
        sqlite3_stmt* stmt;
        const char* query = "INSERT INTO runs (model_path, timestamp, description) VALUES (?, ?, ?);";
        
        int rc = sqlite3_prepare_v2(db, query, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
            return -1;
        }
        
        sqlite3_bind_text(stmt, 1, model_path.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, timestamp.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, description.c_str(), -1, SQLITE_STATIC);
        
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Failed to insert run: %s\n", sqlite3_errmsg(db));
            sqlite3_finalize(stmt);
            return -1;
        }
        
        int64_t run_id = sqlite3_last_insert_rowid(db);
        sqlite3_finalize(stmt);
        
        return run_id;
    }

    int64_t save_stats(int64_t run_id, const std::string& layer_name, const std::string& quant_type, 
                      const error_stats& stats, double median, double p95) {
        if (!initialized || run_id < 0) return -1;

        double rmse = sqrt(stats.total_error / (double) stats.num_samples);
        
        // Insert stats
        sqlite3_stmt* stmt;
        const char* query = "INSERT INTO quantization_stats "
                           "(run_id, layer_name, quant_type, rmse, max_error, median_error, p95_error, num_samples) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
        
        int rc = sqlite3_prepare_v2(db, query, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
            return -1;
        }
        
        sqlite3_bind_int64(stmt, 1, run_id);
        sqlite3_bind_text(stmt, 2, layer_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, quant_type.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, rmse);
        sqlite3_bind_double(stmt, 5, stats.max_error);
        sqlite3_bind_double(stmt, 6, median);
        sqlite3_bind_double(stmt, 7, p95);
        sqlite3_bind_int64(stmt, 8, stats.num_samples);
        
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Failed to insert stats: %s\n", sqlite3_errmsg(db));
            sqlite3_finalize(stmt);
            return -1;
        }
        
        int64_t stats_id = sqlite3_last_insert_rowid(db);
        sqlite3_finalize(stmt);
        
        // Insert histogram data
        const char* hist_query = "INSERT INTO error_histograms (stats_id, bucket_index, bucket_value) VALUES (?, ?, ?);";
        
        rc = sqlite3_prepare_v2(db, hist_query, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "Failed to prepare histogram statement: %s\n", sqlite3_errmsg(db));
            return stats_id;
        }
        
        for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
            sqlite3_bind_int64(stmt, 1, stats_id);
            sqlite3_bind_int(stmt, 2, i);
            sqlite3_bind_int64(stmt, 3, stats.error_histogram[i]);
            
            rc = sqlite3_step(stmt);
            if (rc != SQLITE_DONE) {
                fprintf(stderr, "Failed to insert histogram data: %s\n", sqlite3_errmsg(db));
                break;
            }
            
            sqlite3_reset(stmt);
        }
        
        sqlite3_finalize(stmt);
        return stats_id;
    }

    bool is_initialized() const {
        return initialized;
    }
};

// Utility functions copied from test-quantize-stats.cpp
static bool tensor_is_contiguous(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static void update_error_stats(int64_t nelements, const float * input, const float * output, error_stats & stats) {
    for (int64_t i = 0; i < nelements; i++) {
        double diff = input[i] - output[i];
        stats.total_error += diff * diff;
        stats.max_error = fmax(fabs(diff), stats.max_error);
        stats.error_histogram[std::max(std::min((size_t) floor(fabs(diff) / HISTOGRAM_RANGE * HISTOGRAM_BUCKETS), HISTOGRAM_BUCKETS-1), (size_t) 0)]++;
    }
    stats.num_samples += nelements;
}

static void combine_error_stats(error_stats & into, const error_stats & from) {
    into.num_samples += from.num_samples;
    into.total_error += from.total_error;
    if (from.max_error > into.max_error) into.max_error = from.max_error;
    for (size_t i=0; i<HISTOGRAM_BUCKETS; ++i) into.error_histogram[i] += from.error_histogram[i];
}

static double find_quantile(const error_stats & stats, double quantile) {
    double sum = std::accumulate(std::begin(stats.error_histogram), std::end(stats.error_histogram), 0.0);

    double accum = 0;
    for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
        accum += stats.error_histogram[i];
        if (accum >= sum*quantile) {
            return (i+1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
        }
    }
    return INFINITY;
}

static void test_roundtrip_on_chunk(
    const ggml_tensor * layer, int64_t offset, int64_t chunk_size, const ggml_type_traits & qfns, const ggml_type_traits_cpu & qfns_cpu, bool use_reference,
    float * input_scratch, char * quantized_scratch, float * output_scratch, error_stats & stats
) {
    if (layer->type == GGML_TYPE_F16) {
        for (int i = 0; i < chunk_size; i++) {
            input_scratch[i] = ggml_get_f32_1d(layer, i + offset);
        }
    } else {
        input_scratch = ggml_get_data_f32(layer) + offset;
    }

    if (use_reference) {
        qfns.from_float_ref(input_scratch, quantized_scratch, chunk_size);
    } else {
        qfns_cpu.from_float(input_scratch, quantized_scratch, chunk_size);
    }
    qfns.to_float(quantized_scratch, output_scratch, chunk_size);

    update_error_stats(chunk_size, input_scratch, output_scratch, stats);
}

// Forward declarations for Qt classes
class ModelLoader;
class AnalysisWorker;

// Main application window
class QuantizationAnalysisWindow : public QMainWindow {
    Q_OBJECT

private:
    // Database
    DatabaseManager db_manager;
    int64_t current_run_id = -1;

    // UI components
    QLineEdit* modelPathEdit;
    QLineEdit* dbPathEdit;
    QCheckBox* verboseCheckbox;
    QCheckBox* perLayerStatsCheckbox;
    QCheckBox* referenceCheckbox;
    QLineEdit* includeLayersEdit;
    QLineEdit* excludeLayersEdit;
    QListWidget* quantTypeList;
    QSpinBox* threadsSpinBox;
    QPushButton* analyzeButton;
    QProgressBar* progressBar;
    QTextEdit* logOutput;
    QTabWidget* resultsTabs;
    QTableWidget* resultsTable;
    QChartView* histogramView;

    // Model and analysis state
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::vector<std::string> selected_types;
    bool analysis_running = false;
    
    // Worker thread
    QThread* workerThread = nullptr;

public:
    QuantizationAnalysisWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Quantization Analysis Tool");
        setMinimumSize(800, 600);

        // Create central widget and layout
        QWidget* centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);
        QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);

        // Create settings group
        QGroupBox* settingsGroup = new QGroupBox("Analysis Settings");
        QVBoxLayout* settingsLayout = new QVBoxLayout(settingsGroup);

        // Model path selection
        QHBoxLayout* modelPathLayout = new QHBoxLayout();
        modelPathLayout->addWidget(new QLabel("Model Path:"));
        modelPathEdit = new QLineEdit();
        modelPathLayout->addWidget(modelPathEdit);
        QPushButton* browseButton = new QPushButton("Browse");
        modelPathLayout->addWidget(browseButton);
        settingsLayout->addLayout(modelPathLayout);

        // Database path selection
        QHBoxLayout* dbPathLayout = new QHBoxLayout();
        dbPathLayout->addWidget(new QLabel("Database Path:"));
        dbPathEdit = new QLineEdit("quantization_stats.db");
        dbPathLayout->addWidget(dbPathEdit);
        QPushButton* browseDatabaseButton = new QPushButton("Browse");
        dbPathLayout->addWidget(browseDatabaseButton);
        settingsLayout->addLayout(dbPathLayout);

        // Options
        QHBoxLayout* optionsLayout = new QHBoxLayout();
        verboseCheckbox = new QCheckBox("Verbose");
        perLayerStatsCheckbox = new QCheckBox("Per-layer Stats");
        referenceCheckbox = new QCheckBox("Use Reference Implementation");
        optionsLayout->addWidget(verboseCheckbox);
        optionsLayout->addWidget(perLayerStatsCheckbox);
        optionsLayout->addWidget(referenceCheckbox);
        settingsLayout->addLayout(optionsLayout);

        // Layer filters
        QHBoxLayout* layerFilterLayout = new QHBoxLayout();
        layerFilterLayout->addWidget(new QLabel("Include Layers (regex):"));
        includeLayersEdit = new QLineEdit();
        layerFilterLayout->addWidget(includeLayersEdit);
        layerFilterLayout->addWidget(new QLabel("Exclude Layers (regex):"));
        excludeLayersEdit = new QLineEdit();
        layerFilterLayout->addWidget(excludeLayersEdit);
        settingsLayout->addLayout(layerFilterLayout);

        // Quantization type selection
        QHBoxLayout* quantTypeLayout = new QHBoxLayout();
        quantTypeLayout->addWidget(new QLabel("Quantization Types:"));
        quantTypeList = new QListWidget();
        quantTypeList->setSelectionMode(QAbstractItemView::MultiSelection);
        
        // Add all quantization types
        for (int i = 0; i < GGML_TYPE_COUNT; i++) {
            const char* type_name = ggml_type_name((ggml_type)i);
            if (type_name) {
                quantTypeList->addItem(type_name);
            }
        }
        
        quantTypeLayout->addWidget(quantTypeList);
        
        // Thread count
        QVBoxLayout* threadLayout = new QVBoxLayout();
        threadLayout->addWidget(new QLabel("Threads:"));
        threadsSpinBox = new QSpinBox();
        threadsSpinBox->setMinimum(0);
        threadsSpinBox->setMaximum(64);
        threadsSpinBox->setValue(0);
        threadsSpinBox->setSpecialValueText("Auto");
        threadLayout->addWidget(threadsSpinBox);
        threadLayout->addStretch();
        
        quantTypeLayout->addLayout(threadLayout);
        settingsLayout->addLayout(quantTypeLayout);

        // Add settings group to main layout
        mainLayout->addWidget(settingsGroup);

        // Progress bar
        progressBar = new QProgressBar();
        progressBar->setRange(0, 100);
        progressBar->setValue(0);
        mainLayout->addWidget(progressBar);

        // Analyze button
        analyzeButton = new QPushButton("Start Analysis");
        mainLayout->addWidget(analyzeButton);

        // Results tabs
        resultsTabs = new QTabWidget();
        
        // Log output tab
        logOutput = new QTextEdit();
        logOutput->setReadOnly(true);
        resultsTabs->addTab(logOutput, "Log");
        
        // Results table tab
        resultsTable = new QTableWidget();
        resultsTable->setColumnCount(7);
        resultsTable->setHorizontalHeaderLabels({"Layer", "Type", "RMSE", "Max Error", "Median", "95%", "Samples"});
        resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        resultsTabs->addTab(resultsTable, "Results");
        
        // Histogram tab
        histogramView = new QChartView();
        histogramView->setRenderHint(QPainter::Antialiasing);
        resultsTabs->addTab(histogramView, "Histogram");
        
        mainLayout->addWidget(resultsTabs);

        // Connect signals and slots
        connect(browseButton, &QPushButton::clicked, this, &QuantizationAnalysisWindow::browseModelFile);
        connect(browseDatabaseButton, &QPushButton::clicked, this, &QuantizationAnalysisWindow::browseDatabaseFile);
        connect(analyzeButton, &QPushButton::clicked, this, &QuantizationAnalysisWindow::startAnalysis);

        // Initialize database
        if (!db_manager.init(dbPathEdit->text().toStdString())) {
            QMessageBox::warning(this, "Database Error", "Failed to initialize the database.");
        }
    }

    ~QuantizationAnalysisWindow() {
        cleanup();
    }

private slots:
    void browseModelFile() {
        QString filePath = QFileDialog::getOpenFileName(this, "Select Model File", "", "Model Files (*.bin *.gguf);;All Files (*)");
        if (!filePath.isEmpty()) {
            modelPathEdit->setText(filePath);
        }
    }

    void browseDatabaseFile() {
        QString filePath = QFileDialog::getSaveFileName(this, "Select Database File", "", "SQLite Files (*.db);;All Files (*)");
        if (!filePath.isEmpty()) {
            dbPathEdit->setText(filePath);
            
            // Re-initialize database with new path
            if (!db_manager.init(filePath.toStdString())) {
                QMessageBox::warning(this, "Database Error", "Failed to initialize the database.");
            }
        }
    }

    void startAnalysis() {
        if (analysis_running) {
            QMessageBox::information(this, "Analysis in Progress", "An analysis is already running.");
            return;
        }

        // Clear previous results
        logOutput->clear();
        resultsTable->setRowCount(0);
        
        // Get parameters
        std::string model_path = modelPathEdit->text().toStdString();
        if (model_path.empty()) {
            QMessageBox::warning(this, "Error", "Please select a model file.");
            return;
        }

        // Initialize database if needed
        if (!db_manager.is_initialized()) {
            if (!db_manager.init(dbPathEdit->text().toStdString())) {
                QMessageBox::warning(this, "Database Error", "Failed to initialize the database.");
                return;
            }
        }

        // Start a new run in the database
        current_run_id = db_manager.start_new_run(model_path, "Qt GUI Analysis");
        if (current_run_id < 0) {
            QMessageBox::warning(this, "Database Error", "Failed to create a new run in the database.");
            return;
        }

        // Load the model
        log("Loading model: " + model_path);
        progressBar->setValue(0);
        
        // Disable UI during analysis
        setUIEnabled(false);
        analysis_running = true;

        // Get selected quantization types
        selected_types.clear();
        for (int i = 0; i < quantTypeList->count(); i++) {
            QListWidgetItem* item = quantTypeList->item(i);
            if (item->isSelected()) {
                selected_types.push_back(item->text().toStdString());
            }
        }
        
        // If no types selected, use all types
        if (selected_types.empty()) {
            for (int i = 0; i < GGML_TYPE_COUNT; i++) {
                const char* type_name = ggml_type_name((ggml_type)i);
                if (type_name) {
                    selected_types.push_back(type_name);
                }
            }
        }

        // Load model in a separate thread
        QThread* loaderThread = new QThread;
        ModelLoader* loader = new ModelLoader(model_path);
        loader->moveToThread(loaderThread);
        
        connect(loaderThread, &QThread::started, loader, &ModelLoader::loadModel);
        connect(loader, &ModelLoader::modelLoaded, this, &QuantizationAnalysisWindow::onModelLoaded);
        connect(loader, &ModelLoader::loadError, this, &QuantizationAnalysisWindow::onModelLoadError);
        connect(loader, &ModelLoader::finished, loaderThread, &QThread::quit);
        connect(loader, &ModelLoader::finished, loader, &QObject::deleteLater);
        connect(loaderThread, &QThread::finished, loaderThread, &QObject::deleteLater);
        
        loaderThread->start();
    }

    void onModelLoaded(llama_model* loadedModel, llama_context* loadedCtx) {
        model = loadedModel;
        ctx = loadedCtx;
        
        log("Model loaded successfully.");
        progressBar->setValue(10);
        
        // Start analysis in a worker thread
        QThread* analysisThread = new QThread;
        AnalysisWorker* worker = new AnalysisWorker(
            model, 
            verboseCheckbox->isChecked(),
            perLayerStatsCheckbox->isChecked(),
            referenceCheckbox->isChecked(),
            includeLayersEdit->text().toStdString(),
            excludeLayersEdit->text().toStdString(),
            selected_types,
            threadsSpinBox->value()
        );
        
        worker->moveToThread(analysisThread);
        
        connect(analysisThread, &QThread::started, worker, &AnalysisWorker::runAnalysis);
        connect(worker, &AnalysisWorker::progressUpdate, this, &QuantizationAnalysisWindow::updateProgress);
        connect(worker, &AnalysisWorker::logMessage, this, &QuantizationAnalysisWindow::log);
        connect(worker, &AnalysisWorker::resultReady, this, &QuantizationAnalysisWindow::addResult);
        connect(worker, &AnalysisWorker::analysisComplete, this, &QuantizationAnalysisWindow::onAnalysisComplete);
        connect(worker, &AnalysisWorker::finished, analysisThread, &QThread::quit);
        connect(worker, &AnalysisWorker::finished, worker, &QObject::deleteLater);
        connect(analysisThread, &QThread::finished, analysisThread, &QObject::deleteLater);
        
        analysisThread->start();
        workerThread = analysisThread;
    }

    void onModelLoadError(const QString& error) {
        log("Error loading model: " + error.toStdString());
        QMessageBox::critical(this, "Model Load Error", error);
        setUIEnabled(true);
        analysis_running = false;
    }

    void updateProgress(int progress) {
        progressBar->setValue(progress);
    }

    void log(const std::string& message) {
        logOutput->append(QString::fromStdString(message));
    }

    void addResult(const std::string& layer, const std::string& type, double rmse, 
                  double max_error, double median, double p95, size_t samples, 
                  const error_stats& stats) {
        // Add to results table
        int row = resultsTable->rowCount();
        resultsTable->insertRow(row);
        
        resultsTable->setItem(row, 0, new QTableWidgetItem(QString::fromStdString(layer)));
        resultsTable->setItem(row, 1, new QTableWidgetItem(QString::fromStdString(type)));
        resultsTable->setItem(row, 2, new QTableWidgetItem(QString::number(rmse, 'g', 8)));
        resultsTable->setItem(row, 3, new QTableWidgetItem(QString::number(max_error, 'g', 8)));
        resultsTable->setItem(row, 4, new QTableWidgetItem(QString::number(median, 'g', 8)));
        resultsTable->setItem(row, 5, new QTableWidgetItem(QString::number(p95, 'g', 8)));
        resultsTable->setItem(row, 6, new QTableWidgetItem(QString::number(samples)));
        
        // Save to database
        db_manager.save_stats(current_run_id, layer, type, stats, median, p95);
        
        // Update histogram if this is a global result (not per-layer)
        if (layer.find("::") == std::string::npos) {
            updateHistogram(type, stats);
        }
    }

    void updateHistogram(const std::string& type, const error_stats& stats) {
        // Create a new chart
        QChart* chart = new QChart();
        chart->setTitle(QString("Error Distribution for %1").arg(QString::fromStdString(type)));
        
        // Create a bar set for the histogram
        QBarSet* barSet = new QBarSet("Error Count");
        
        // Find the max value for scaling
        uint64_t maxValue = 0;
        for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
            if (stats.error_histogram[i] > maxValue) {
                maxValue = stats.error_histogram[i];
            }
        }
        
        // Add data to the bar set (use log scale for better visualization)
        QStringList categories;
        for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
            double value = stats.error_histogram[i] > 0 ? log10(stats.error_histogram[i]) : 0;
            *barSet << value;
            
            // Create category labels (only show some for readability)
            if (i % 10 == 0 || i == HISTOGRAM_BUCKETS - 1) {
                double error = (i+0.5) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
                categories << QString::number(error, 'g', 3);
            } else {
                categories << "";
            }
        }
        
        // Create the bar series
        QBarSeries* series = new QBarSeries();
        series->append(barSet);
        chart->addSeries(series);
        
        // Set up the axes
        QBarCategoryAxis* axisX = new QBarCategoryAxis();
        axisX->append(categories);
        chart->addAxis(axisX, Qt::AlignBottom);
        series->attachAxis(axisX);
        
        QValueAxis* axisY = new QValueAxis();
        axisY->setLabelFormat("%.1f");
        axisY->setTitleText("log10(count)");
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisY);
        
        // Set the chart in the view
        histogramView->setChart(chart);
        histogramView->setRenderHint(QPainter::Antialiasing);
        
        // Switch to the histogram tab
        resultsTabs->setCurrentWidget(histogramView);
    }

    void onAnalysisComplete() {
        log("Analysis complete.");
        progressBar->setValue(100);
        setUIEnabled(true);
        analysis_running = false;
        
        // Clean up model and context
        cleanup();
    }

    void setUIEnabled(bool enabled) {
        modelPathEdit->setEnabled(enabled);
        dbPathEdit->setEnabled(enabled);
        verboseCheckbox->setEnabled(enabled);
        perLayerStatsCheckbox->setEnabled(enabled);
        referenceCheckbox->setEnabled(enabled);
        includeLayersEdit->setEnabled(enabled);
        excludeLayersEdit->setEnabled(enabled);
        quantTypeList->setEnabled(enabled);
        threadsSpinBox->setEnabled(enabled);
        analyzeButton->setEnabled(enabled);
    }

    void cleanup() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
    }
};

// Model loader class
class ModelLoader : public QObject {
    Q_OBJECT
    
private:
    std::string model_path;
    
public:
    ModelLoader(const std::string& path) : model_path(path) {}
    
public slots:
    void loadModel() {
        llama_model* model = nullptr;
        llama_context* ctx = nullptr;
        
        try {
            auto mparams = llama_model_default_params();
            mparams.use_mlock = false;
            
            model = llama_model_load_from_file(model_path.c_str(), mparams);
            
            if (model == nullptr) {
                emit loadError(QString("Failed to load model '%1'").arg(QString::fromStdString(model_path)));
                emit finished();
                return;
            }
            
            auto cparams = llama_context_default_params();
            cparams.n_ctx = 256;
            
            ctx = llama_init_from_model(model, cparams);
            
            if (ctx == nullptr) {
                llama_model_free(model);
                emit loadError(QString("Failed to create context with model '%1'").arg(QString::fromStdString(model_path)));
                emit finished();
                return;
            }
            
            emit modelLoaded(model, ctx);
        } catch (const std::exception& e) {
            if (ctx) llama_free(ctx);
            if (model) llama_model_free(model);
            emit loadError(QString("Exception during model loading: %1").arg(e.what()));
        }
        
        emit finished();
    }
    
signals:
    void modelLoaded(llama_model* model, llama_context* ctx);
    void loadError(const QString& error);
    void finished();
};

// Analysis worker class
class AnalysisWorker : public QObject {
    Q_OBJECT
    
private:
    llama_model* model;
    bool verbose;
    bool per_layer_stats;
    bool reference;
    std::string include_layers_pattern;
    std::string exclude_layers_pattern;
    std::vector<std::string> quant_types;
    int max_threads;
    
    std::vector<std::regex> include_regexes;
    std::vector<std::regex> exclude_regexes;
    
public:
    AnalysisWorker(llama_model* m, bool v, bool pls, bool ref, 
                  const std::string& include, const std::string& exclude,
                  const std::vector<std::string>& types, int threads)
        : model(m), verbose(v), per_layer_stats(pls), reference(ref),
          include_layers_pattern(include), exclude_layers_pattern(exclude),
          quant_types(types), max_threads(threads) {
        
        // Compile regexes
        if (!include_layers_pattern.empty()) {
            include_regexes.push_back(std::regex(include_layers_pattern));
        }
        
        if (!exclude_layers_pattern.empty()) {
            exclude_regexes.push_back(std::regex(exclude_layers_pattern));
        }
    }
    
    bool layer_included(const std::string& layer) {
        for (const auto& regex : exclude_regexes) {
            if (std::regex_search(layer, regex)) {
                return false;
            }
        }
        
        if (include_regexes.empty()) {
            return true;
        }
        
        for (const auto& regex : include_regexes) {
            if (std::regex_search(layer, regex)) {
                return true;
            }
        }
        
        return false;
    }
    
public slots:
    void runAnalysis() {
        emit logMessage("Starting analysis...");
        
        const auto& tensors = llama_internal_get_tensor_map(model);
        
        // Check layer tensors
        int included_layers = 0;
        int64_t max_nelements = 0;
        bool is_f16 = false;
        
        for (const auto& kv_tensor : tensors) {
            if (!layer_included(kv_tensor.first)) {
                continue;
            }
            
            if (verbose) {
                emit logMessage(kv_tensor.first + ": type " + 
                               std::string(ggml_type_name(kv_tensor.second->type)) + 
                               ", size " + std::to_string(ggml_nelements(kv_tensor.second)));
            }
            
            if (kv_tensor.second->type == GGML_TYPE_F16) {
                is_f16 = true;
            } else if (kv_tensor.second->type != GGML_TYPE_F32) {
                emit logMessage("Error: Quantization should be tested with a float model, "
                               "this model contains already quantized layers (" + 
                               kv_tensor.first + " is type " + 
                               std::to_string(kv_tensor.second->type) + ")");
                emit analysisComplete();
                emit finished();
                return;
            }
            
            included_layers++;
            max_nelements = std::max(max_nelements, ggml_nelements(kv_tensor.second));
        }
        
        if (is_f16) {
            emit logMessage("Note: source model is f16");
        }
        
        emit logMessage("Testing " + std::to_string(included_layers) + 
                       " layers with max size " + std::to_string(max_nelements));
        
        // Allocate scratch space
        std::vector<float> input_scratch;
        std::vector<char> quantized_scratch;
        std::vector<float> output_scratch;
        
        int progress = 15;
        emit progressUpdate(progress);
        
        // Loop through quantization types
        int type_count = 0;
        for (const auto& type_name : quant_types) {
            ggml_type type = GGML_TYPE_COUNT;
            
            // Find the type by name
            for (int i = 0; i < GGML_TYPE_COUNT; i++) {
                const char* name = ggml_type_name((ggml_type)i);
                if (name && type_name == name) {
                    type = (ggml_type)i;
                    break;
                }
            }
            
            if (type == GGML_TYPE_COUNT) {
                continue;
            }
            
            const auto* qfns = ggml_get_type_traits(type);
            const auto* qfns_cpu = ggml_get_type_traits_cpu(type);
            
            if (!qfns_cpu->from_float || !qfns->to_float) {
                continue;
            }
            
            if (verbose) {
                emit logMessage("Testing " + type_name + "...");
            }
            
            ggml_quantize_init(type);
            
            error_stats global_stats;
            
            int layer_count = 0;
            for (const auto& kv_tensor : tensors) {
                if (!layer_included(kv_tensor.first)) {
                    continue;
                }
                
                if (verbose) {
                    emit logMessage("  " + kv_tensor.first + "...");
                }
                
                std::string layer_name = type_name + "::" + kv_tensor.first;
                
                // Test roundtrip on this layer
                if (!tensor_is_contiguous(kv_tensor.second)) {
                    emit logMessage("Warning: Skipping non-contiguous tensor " + kv_tensor.first);
                    continue;
                }
                
                error_stats layer_stats;
                uint64_t nelements = ggml_nelements(kv_tensor.second);
                
                float* input_scratch_ptr = nullptr;
                if (kv_tensor.second->type == GGML_TYPE_F16) {
                    if (input_scratch.size() < nelements) input_scratch.resize(nelements);
                    input_scratch_ptr = input_scratch.data();
                    
                    for (int i = 0; i < nelements; i++) {
                        input_scratch_ptr[i] = ggml_get_f32_1d(kv_tensor.second, i);
                    }
                } else {
                    input_scratch_ptr = ggml_get_data_f32(kv_tensor.second);
                }
                
                if (quantized_scratch.size() < 4*nelements) quantized_scratch.resize(4*nelements);
                if (output_scratch.size() < nelements) output_scratch.resize(nelements);
                
                int threads_to_use = max_threads;
                if (threads_to_use < 1) threads_to_use = std::thread::hardware_concurrency();
                
                int chunk_size = 32*512;
                int num_chunks = (nelements + chunk_size - 1)/chunk_size;
                
                if (num_chunks < 2 || threads_to_use < 2) {
                    // Single-threaded processing
                    if (reference) {
                        qfns->from_float_ref(input_scratch_ptr, quantized_scratch.data(), nelements);
                    } else {
                        qfns_cpu->from_float(input_scratch_ptr, quantized_scratch.data(), nelements);
                    }
                    qfns->to_float(quantized_scratch.data(), output_scratch.data(), nelements);
                    
                    update_error_stats(nelements, input_scratch_ptr, output_scratch.data(), layer_stats);
                } else {
                    // Multi-threaded processing
                    std::mutex mutex;
                    uint64_t counter = 0;
                    std::vector<std::thread> workers(threads_to_use - 1);
                    
                    auto compute = [&]() {
                        error_stats local_stats;
                        while (true) {
                            uint64_t offset;
                            {
                                std::lock_guard<std::mutex> lock(mutex);
                                offset = counter;
                                counter += chunk_size;
                                if (offset >= nelements) {
                                    combine_error_stats(layer_stats, local_stats);
                                    break;
                                }
                            }
                            
                            uint64_t chunk = std::min((uint64_t)chunk_size, nelements - offset);
                            
                            float* input = kv_tensor.second->type == GGML_TYPE_F16 ? 
                                          input_scratch_ptr + offset : 
                                          ggml_get_data_f32(kv_tensor.second) + offset;
                            
                            char* quantized = quantized_scratch.data() + 4*offset;
                            float* output = output_scratch.data() + offset;
                            
                            if (reference) {
                                qfns->from_float_ref(input, quantized, chunk);
                            } else {
                                qfns_cpu->from_float(input, quantized, chunk);
                            }
                            qfns->to_float(quantized, output, chunk);
                            
                            update_error_stats(chunk, input, output, local_stats);
                        }
                    };
                    
                    for (auto& worker : workers) {
                        worker = std::thread(compute);
                    }
                    
                    compute();
                    
                    for (auto& worker : workers) {
                        worker.join();
                    }
                }
                
                double rmse = sqrt(layer_stats.total_error / (double)layer_stats.num_samples);
                double median = find_quantile(layer_stats, 0.5);
                double p95 = find_quantile(layer_stats, 0.95);
                
                if (per_layer_stats) {
                    emit resultReady(kv_tensor.first, type_name, rmse, layer_stats.max_error, 
                                    median, p95, layer_stats.num_samples, layer_stats);
                }
                
                combine_error_stats(global_stats, layer_stats);
                
                // Update progress
                layer_count++;
                int layer_progress = 15 + (75 * type_count / quant_types.size()) + 
                                    (75 / quant_types.size() * layer_count / included_layers);
                emit progressUpdate(layer_progress);
            }
            
            // Report global stats for this type
            double rmse = sqrt(global_stats.total_error / (double)global_stats.num_samples);
            double median = find_quantile(global_stats, 0.5);
            double p95 = find_quantile(global_stats, 0.95);
            
            emit resultReady(type_name, type_name, rmse, global_stats.max_error, 
                            median, p95, global_stats.num_samples, global_stats);
            
            type_count++;
        }
        
        emit progressUpdate(100);
        emit analysisComplete();
        emit finished();
    }
    
signals:
    void progressUpdate(int progress);
    void logMessage(const std::string& message);
    void resultReady(const std::string& layer, const std::string& type, 
                    double rmse, double max_error, double median, 
                    double p95, size_t samples, const error_stats& stats);
    void analysisComplete();
    void finished();
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    
    ggml_time_init();
    
    QuantizationAnalysisWindow window;
    window.show();
    
    return app.exec();
}

#include "quantization_analysis.moc"
