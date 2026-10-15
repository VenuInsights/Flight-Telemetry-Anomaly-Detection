# Engine Sensor Anomaly Detection Using Machine Learning

## Overview

This project explores simulated turbofan engine sensor data using Python, exploratory data analysis, signal smoothing, and unsupervised machine learning. It uses NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset (FD001) to investigate sensor patterns and identify potentially unusual observations.

The project implements Isolation Forest, Local Outlier Factor (LOF), and a statistical Z-score baseline to compare their responses to artificially injected sensor anomalies. It demonstrates a basic workflow for sensor-data processing, anomaly detection, and model evaluation relevant to predictive maintenance and condition monitoring.

**Note:** The dataset represents simulated turbofan engine degradation, not actual aircraft flight telemetry.

## Objectives

- Explore and analyze engine sensor measurements.
- Visualize sensor readings across operating cycles.
- Apply moving-average filtering to smooth sensor data.
- Implement Isolation Forest for unsupervised anomaly detection.
- Investigate potentially anomalous records across engines.
- Generate synthetic sensor anomalies to test model behavior.
- Compare multiple anomaly-detection approaches.
- Evaluate detection rates and false-positive rates using a held-out set of records.

## Technologies Used

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook, Visual Studio Code
- **Version Control:** Git and GitHub

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset — FD001**

The dataset contains simulated engine operating cycles, operating settings, and sensor measurements.

The `train_FD001.txt` file contains 20,631 records and 26 columns in the expected dataset format:

- 1 engine identifier
- 1 operating-cycle value
- 3 operating settings
- 21 sensor measurements

Dataset information: [NASA C-MAPSS dataset on NASA's open-data portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

The dataset file is not included in this repository. Download it separately and place `train_FD001.txt` inside `data/raw/`.

## Project Workflow

1. Load and inspect the simulated engine sensor dataset.
2. Perform exploratory data analysis to understand the dataset structure and sensor distributions.
3. Visualize engine operating cycles and selected sensor readings.
4. Apply moving-average filtering to smooth sensor measurements.
5. Train an Isolation Forest model using sensor readings and operating settings.
6. Examine records flagged as potential anomalies.
7. Create synthetic anomalies by modifying selected sensor readings.
8. Compare Isolation Forest, Local Outlier Factor, and a Z-score baseline.
9. Evaluate model behavior using held-out records and synthetic anomalies.
10. Save evaluation results and visualize model performance.

## Anomaly Detection Methods

### 1. Isolation Forest

Isolation Forest is an unsupervised anomaly-detection algorithm that identifies observations that are easier to isolate from the rest of the data.

The implementation uses sensor measurements and operating settings as input features. Its contamination parameter is configured experimentally to control the expected proportion of outliers.

Predictions are interpreted as:

- `1`: Normal observation
- `-1`: Potential anomaly

### 2. Local Outlier Factor (LOF)

Local Outlier Factor identifies observations whose local density differs from that of their neighboring observations. A novelty-detection configuration is used for predictions on held-out and modified records.

### 3. Z-Score Baseline

The Z-score baseline measures how far a sensor reading is from the training-data mean in units of standard deviation.

For this experiment, the method examines `sensor_2` and flags readings whose absolute Z-score exceeds 3.

This is a simple statistical baseline rather than a trained multivariate machine-learning model.

## Model Evaluation

The models were compared using 100 synthetic anomalies created by increasing `sensor_2` by five standard deviations. The holdout experiment separated training records from held-out records and fitted the feature scaler on training data only.

### Holdout Evaluation Results

| Model | Detected Anomalies | Missed Anomalies | False Positives | Detection Rate | False-Positive Rate |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 2 | 98 | 76 | 2.00% | 1.84% |
| Local Outlier Factor | 100 | 0 | 82 | 100.00% | 1.99% |
| Z-score baseline | 98 | 2 | 7 | 98.00% | 0.17% |

**Interpretation:** In this specific synthetic experiment, the Z-score baseline detected 98 of the 100 injected anomalies and produced the fewest false positives among the three approaches. Local Outlier Factor detected all injected anomalies but also flagged more normal records. Isolation Forest detected relatively few of the injected anomalies with the tested configuration.

These results are specific to the experiment and do not establish that one method is universally superior.

### Evaluation Metrics

- **Detection rate (recall):** The proportion of injected anomalies correctly identified.
- **False-positive rate:** The proportion of normal test records incorrectly flagged as anomalies.
- **False positives:** Normal records predicted as anomalies.
- **Missed anomalies:** Injected anomalies predicted as normal.

The reported metrics evaluate performance against known synthetic modifications, not independently verified engine failures.

## Limitations

- The dataset contains simulated turbofan engine degradation data rather than actual flight telemetry.
- The injected anomalies are artificial and may not represent realistic sensor failures or engine faults.
- The Z-score baseline directly examines the same sensor that is modified during testing, giving it an advantage in this particular experiment.
- The model comparison does not establish performance on real-world aerospace faults.
- The Isolation Forest contamination parameter is an experimental assumption, not a verified fault rate.
- Additional validation with independently labelled fault data, more realistic fault scenarios, and appropriate train-test separation is required before drawing operational conclusions.
- The project does not currently implement GPS/IMU flight telemetry analysis, radar target tracking, or a production-ready aircraft monitoring system.

## Project Structure

```text
Flight-Telemetry-Anomaly-Detection/
├── data/
│   ├── raw/
│   │   └── train_FD001.txt
│   └── processed/
│       ├── synthetic_evaluation.csv
│       ├── scaled_anomaly_comparison.csv
│       ├── model_comparison.csv
│       └── holdout_model_comparison.csv
├── notebooks/
│   ├── 01_explore_telemetry.ipynb
│   ├── 02_evaluate_anomaly_detection.ipynb
│   ├── 03_compare_anomaly_models.ipynb
│   └── 04_holdout_evaluation.ipynb
├── src/
├── requirements.txt
└── README.md
```

The listed CSV files are generated by the notebooks. They may not exist until the corresponding notebook cells have been executed.

## Installation and Usage

### Prerequisites

- Python 3.10 or later
- Visual Studio Code
- Jupyter support in VS Code

### 1. Clone the repository

```bash
git clone https://github.com/VenuInsights/Flight-Telemetry-Anomaly-Detection.git
cd Flight-Telemetry-Anomaly-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in the VS Code terminal on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download the C-MAPSS FD001 training data from the dataset source and place `train_FD001.txt` in:

```text
data/raw/train_FD001.txt
```

Ensure the downloaded file matches the expected whitespace-separated FD001 format.

### 5. Run the notebooks

Open the repository in VS Code and run the notebooks in order:

1. `01_explore_telemetry.ipynb`
2. `02_evaluate_anomaly_detection.ipynb`
3. `03_compare_anomaly_models.ipynb`
4. `04_holdout_evaluation.ipynb`

Run each notebook's cells from top to bottom. The notebooks depend on the dataset being present in the specified location.

## Future Improvements

- Investigate model parameters and feature engineering using a more rigorous evaluation protocol.
- Test multiple types of synthetic anomalies, including spikes, drift, and abrupt changes.
- Evaluate performance on appropriately labelled fault scenarios.
- Add more detailed sensor trend and operating-cycle analysis.
- Build an interactive dashboard to explore sensor readings, anomaly flags, and evaluation metrics.
- Explore time-series methods and realistic sensor-failure scenarios.
- Investigate actual flight telemetry datasets containing relevant GPS, IMU, or other avionics sensor measurements.

## Author

Developed as a machine-learning and sensor-data analysis project to explore anomaly detection techniques relevant to predictive maintenance and aerospace applications.