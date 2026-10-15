# Engine Sensor Anomaly Detection Using Machine Learning

## Overview

This project explores engine sensor data using Python, data analysis, signal smoothing, and unsupervised machine learning. It uses NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset to investigate sensor patterns and identify potentially unusual observations.

The project demonstrates a basic workflow for sensor-data processing and anomaly detection that is relevant to predictive maintenance and condition monitoring.

## Objectives

- Explore and analyze engine sensor measurements.
- Visualize sensor readings across operating cycles.
- Apply moving-average filtering to smooth sensor data.
- Implement Isolation Forest for unsupervised anomaly detection.
- Investigate potentially anomalous records across engines.
- Test model behavior using artificially modified sensor readings.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset — FD001**

The dataset contains simulated engine operating cycles, operating settings, and sensor measurements.

The `train_FD001.txt` file contains 20,631 rows and 26 columns in the expected dataset format.

Dataset information: https://www.nasa.gov/

## Project Workflow

1. Load and inspect the engine sensor dataset.
2. Perform exploratory data analysis.
3. Visualize operating cycles and sensor distributions.
4. Apply moving-average filtering to selected sensor readings.
5. Train an Isolation Forest model using sensor readings and operating settings.
6. Analyze records flagged as potential anomalies.
7. Test the model using synthetic modifications to selected sensor readings.
8. Create visual summaries of the results.

## Anomaly Detection

Isolation Forest is used to identify observations that differ from patterns learned from the input data.

The model labels observations as normal or potential anomalies. A configurable contamination parameter controls the expected proportion of outliers.

The synthetic anomaly experiment modifies selected sensor values and checks how the trained model responds.

## Limitations

- The dataset represents simulated engine degradation rather than actual flight telemetry.
- The experiment does not establish anomaly detection performance on real-world aerospace faults.
- Synthetic modifications are artificial and may not represent realistic failure patterns.
- The contamination parameter is an experimental assumption, not a verified fault rate.
- Additional validation using appropriately labelled data and more realistic fault scenarios would be required before operational use.

## Project Structure

```text
Flight-Telemetry-Anomaly-Detection/
├── data/
│   ├── raw/
│   │   └── train_FD001.txt
│   └── processed/
├── notebooks/
│   └── 01_explore_telemetry.ipynb
├── src/
├── requirements.txt
└── README.md
```

## How to Run

1. Clone or download this repository.
2. Install Python and the required dependencies.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open `notebooks/01_explore_telemetry.ipynb` in VS Code or Jupyter.
5. Ensure the dataset is located at `data/raw/train_FD001.txt`.
6. Run the notebook cells in order.

## Future Improvements

- Evaluate the model using carefully designed synthetic fault scenarios.
- Compare Isolation Forest with other anomaly detection methods.
- Add more detailed sensor trend analysis.
- Build an interactive dashboard for exploring anomaly results.
- Explore labelled datasets and more realistic sensor-failure scenarios.