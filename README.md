# Engine Sensor Anomaly Detection Using Machine Learning

An interactive machine-learning project for exploring simulated turbofan engine sensor data, detecting potentially unusual observations, and comparing anomaly-detection methods.

**GitHub Repository:** [Flight-Telemetry-Anomaly-Detection](https://github.com/VenuInsights/Flight-Telemetry-Anomaly-Detection)

**Live Dashboard:** [Open the Streamlit Dashboard](https://flight-telemetry-anomaly-detection-11.streamlit.app/)

## Overview

This project explores simulated turbofan engine sensor data using Python, exploratory data analysis, signal smoothing, and unsupervised machine learning. It uses NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset (FD001) to investigate sensor patterns and identify potentially unusual observations.

The project implements Isolation Forest, Local Outlier Factor (LOF), and a statistical Z-score baseline to compare their responses to artificially injected sensor anomalies. An interactive Streamlit dashboard allows users to select engines and sensors, inspect sensor trends, filter records, review model evaluation results, and export records as CSV files.

The project demonstrates a basic workflow for sensor-data processing, anomaly detection, visualization, and experimental model evaluation relevant to predictive maintenance and condition monitoring.

**Important:** The dataset contains simulated turbofan engine degradation data, not actual aircraft flight telemetry. Model-generated anomaly flags are not confirmed engine faults.

## Objectives

- Explore and analyze engine sensor measurements.
- Visualize sensor readings across operating cycles.
- Apply moving-average filtering to smooth sensor data.
- Implement Isolation Forest for unsupervised anomaly detection.
- Investigate potentially anomalous records across engines.
- Generate synthetic sensor anomalies to test model behavior.
- Compare multiple anomaly-detection approaches.
- Evaluate detection rates and false-positive rates using held-out records.
- Build an interactive dashboard for sensor exploration and anomaly inspection.
- Export selected sensor records and potential anomalies for further analysis.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Streamlit
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook, Visual Studio Code
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset — FD001**

The dataset contains simulated engine operating cycles, operating settings, and sensor measurements.

The `train_FD001.txt` file contains 20,631 records and 26 columns in the expected dataset format:

- 1 engine identifier
- 1 operating-cycle value
- 3 operating settings
- 21 sensor measurements

Dataset information: [NASA C-MAPSS dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

The raw dataset is not included in this repository. Download the FD001 training file separately and place `train_FD001.txt` inside `data/raw/` when running the project locally.

For the deployed dashboard, upload the dataset through the file uploader if the application asks for it.

## Key Features

### 1. Exploratory Data Analysis

- Inspect dataset dimensions and sensor measurements.
- Explore sensor distributions and engine operating cycles.
- Visualize selected sensor trends.
- Apply moving-average smoothing to investigate changes in readings.

### 2. Machine Learning-Based Anomaly Detection

- Apply Isolation Forest to sensor measurements and operating settings.
- Configure the model's expected outlier proportion.
- Flag observations as normal or potentially anomalous.
- Investigate anomaly scores and sensor trends.

### 3. Model Comparison

Compare three approaches:

- Isolation Forest
- Local Outlier Factor (LOF)
- Z-score statistical baseline

The comparison uses synthetic sensor anomalies and a held-out evaluation experiment. Results are interpreted within the limitations of the test design.

### 4. Interactive Streamlit Dashboard

- Select an engine and sensor.
- Adjust the Isolation Forest contamination parameter.
- Visualize sensor readings over operating cycles.
- Highlight potential anomalies on the sensor trend chart.
- Compare a selected engine's average sensor reading with the dataset statistics.
- Filter records to show all, normal, or potentially anomalous observations.
- Download displayed records or all flagged records for the selected engine.
- View saved model evaluation results when the evaluation CSV is available.

**Dashboard note:** If the raw dataset is not bundled with the deployment, the dashboard asks the visitor to upload `train_FD001.txt`. The evaluation comparison table also requires its saved CSV file to be present in the deployed project.

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
11. Use the Streamlit dashboard to explore sensor readings and export records.

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

This is a statistical baseline rather than a trained multivariate machine-learning model.

## Model Evaluation

The models were compared using 100 synthetic anomalies created by increasing `sensor_2` by five standard deviations. The holdout experiment separated training records from held-out records and fitted the feature scaler on training data only.

### Holdout Evaluation Results

| Model | Detected Anomalies | Missed Anomalies | False Positives | Detection Rate | False-Positive Rate |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 2 | 98 | 76 | 2.00% | 1.84% |
| Local Outlier Factor | 100 | 0 | 82 | 100.00% | 1.99% |
| Z-score baseline | 98 | 2 | 7 | 98.00% | 0.17% |

### Interpretation

In this particular synthetic experiment:

- **Isolation Forest** detected 2 of the 100 injected anomalies.
- **LOF** detected all 100 injected anomalies but also flagged more normal records.
- **Z-score** detected 98 injected anomalies and produced the fewest false positives among the three tested methods.

The Z-score baseline directly examines `sensor_2`, the sensor modified to create the synthetic anomalies. This gives it an advantage in this specific test.

These findings do not establish that any method is universally superior or that the models can detect actual aircraft engine failures.

### Evaluation Metrics

- **Detection rate (recall):** The proportion of injected anomalies correctly identified.
- **False-positive rate:** The proportion of normal test records incorrectly flagged as anomalies.
- **False positives:** Normal records predicted as anomalies.
- **Missed anomalies:** Injected anomalies predicted as normal.

The metrics measure agreement with known synthetic modifications, not independently verified engine failures.

## Limitations

- The dataset contains simulated turbofan engine degradation data rather than actual flight telemetry.
- Synthetic anomalies may not represent realistic sensor failures or engine faults.
- The Z-score baseline examines the same sensor that is modified during testing, giving it an advantage in this experiment.
- The reported comparison does not establish performance on real-world aerospace faults.
- The Isolation Forest contamination parameter is an experimental assumption, not a verified fault rate.
- The interactive dashboard fits Isolation Forest on the dataset being explored. Its live anomaly counts are descriptive and are not independent model evaluation results.
- The saved evaluation results are separate from the dashboard's live anomaly flags.
- Additional validation using independently labelled fault data and more realistic fault scenarios is required before drawing operational conclusions.
- The project does not currently implement GPS/IMU flight telemetry analysis, radar target tracking, or a production-ready aircraft monitoring system.

## Project Structure

```text
Flight-Telemetry-Anomaly-Detection/
├── app.py
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

The tree shows the intended project structure. Some processed CSV files are generated by the notebooks and may not exist until the relevant cells have been executed. The raw dataset is downloaded separately and should not be assumed to be present in the repository.

## Installation and Local Usage

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

Activate the environment in the VS Code integrated terminal on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure `requirements.txt` includes the packages required by the application and notebooks, such as:

```text
streamlit
pandas
numpy
matplotlib
scikit-learn
jupyter
```

Keep any other packages your notebooks require.

### 4. Download the dataset

Download the FD001 training file from the dataset source and place it here:

```text
data/raw/train_FD001.txt
```

Ensure the file follows the expected whitespace-separated FD001 format.

### 5. Run the Streamlit dashboard

From the project root, run:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

### 6. Run the notebooks

Open the repository in VS Code and execute the notebooks in order:

1. `notebooks/01_explore_telemetry.ipynb`
2. `notebooks/02_evaluate_anomaly_detection.ipynb`
3. `notebooks/03_compare_anomaly_models.ipynb`
4. `notebooks/04_holdout_evaluation.ipynb`

Run each notebook's cells from top to bottom. The notebooks depend on the raw dataset being available at the specified path.

## Deployment

The dashboard is deployed using Streamlit Community Cloud.

1. Push the project code and `requirements.txt` to GitHub.
2. Connect the GitHub repository through [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the repository's `main` branch and set the app file path to `app.py`.
4. Deploy the application.
5. Open the generated Streamlit URL and upload the FD001 training file if prompted.

The deployed app requires access to the dataset to perform live anomaly detection. The evaluation comparison table requires `data/processed/holdout_model_comparison.csv` to be available in the deployed environment.

## Future Improvements

- Investigate model parameters and feature engineering with a more rigorous evaluation protocol.
- Test additional synthetic anomaly types, including spikes, drift, and abrupt changes.
- Evaluate performance on appropriately labelled fault scenarios.
- Add more detailed sensor trend and operating-cycle analysis.
- Explore time-series methods and realistic sensor-failure scenarios.
- Investigate actual flight telemetry datasets containing GPS, IMU, or other relevant avionics measurements.
- Improve the dashboard's dataset-loading workflow for easier demonstrations.
- Explore additional validation methods before making claims about operational fault detection.

## Author

Developed as a machine-learning and sensor-data analysis project to explore anomaly-detection techniques relevant to predictive maintenance and aerospace applications.
