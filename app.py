import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Engine Sensor Monitoring",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Engine Sensor Anomaly Dashboard")
st.caption(
    "Explore simulated turbofan engine sensor data "
    "and investigate potentially unusual observations."
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train_FD001.txt"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------


@st.cache_data
def load_data(file_bytes):
    columns = [
        "engine_id",
        "cycle",
        "operating_setting_1",
        "operating_setting_2",
        "operating_setting_3",
    ]

    columns += [f"sensor_{i}" for i in range(1, 22)]

    data = pd.read_csv(
        io.BytesIO(file_bytes),
        sep=r"\s+",
        header=None,
        names=columns,
    )

    return data




if DATA_PATH.exists():
    df = load_data(DATA_PATH.read_bytes())
else:
    st.warning(
        "The dataset is not bundled with this deployment. "
        "Upload the NASA C-MAPSS FD001 training file to continue."
    )

    uploaded_file = st.file_uploader(
        "Upload train_FD001.txt",
        type=["txt"],
        help=(
            "Upload the original FD001 training file. "
            "It should contain 26 whitespace-separated columns."
        ),
    )

    if uploaded_file is None:
        st.stop()

    try:
        df = load_data(uploaded_file.getvalue())
    except Exception as error:
        st.error(f"Could not read the uploaded dataset: {error}")
        st.stop()

if df.shape[1] != 26:
    st.error(
        f"Expected 26 columns, but found {df.shape[1]}. "
        "Please upload the original train_FD001.txt file."
    )
    st.stop()


sensor_columns = [
    f"sensor_{i}" for i in range(1, 22)
]

operating_columns = [
    "operating_setting_1",
    "operating_setting_2",
    "operating_setting_3",
]

feature_columns = sensor_columns + operating_columns


# --------------------------------------------------
# 3. Sidebar controls
# --------------------------------------------------

st.sidebar.header("Dashboard Controls")

engine_ids = sorted(df["engine_id"].unique())

selected_engine = st.sidebar.selectbox(
    "Select engine",
    engine_ids,
)

selected_sensor = st.sidebar.selectbox(
    "Select sensor",
    sensor_columns,
)

contamination = st.sidebar.slider(
    "Expected outlier proportion",
    min_value=0.01,
    max_value=0.10,
    value=0.02,
    step=0.01,
    help=(
        "Controls the expected proportion of outliers "
        "identified by Isolation Forest. It is not the "
        "verified percentage of actual engine faults."
    ),
)

st.sidebar.caption(
    "Changing the engine or sensor updates the dashboard. "
    "Changing the outlier proportion triggers a new model fit."
)


# --------------------------------------------------
# 4. Detect potential anomalies
# --------------------------------------------------

@st.cache_data
def detect_anomalies(data, contamination_value):
    feature_names = [
        column
        for column in data.columns
        if column.startswith("sensor_")
        or column.startswith("operating_setting_")
    ]

    features = data[feature_names].copy()

    # Remove constant features
    features = features.loc[:, features.nunique() > 1]

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination_value,
        random_state=42,
    )

    predictions = model.fit_predict(features)
    scores = model.decision_function(features)

    results = data.copy()
    results["anomaly_prediction"] = predictions
    results["anomaly_score"] = scores
    results["is_anomaly"] = predictions == -1

    return results


with st.spinner("Analyzing sensor records..."):
    results = detect_anomalies(df, contamination)


# Get records for the selected engine
engine_data = (
    results[results["engine_id"] == selected_engine]
    .sort_values("cycle")
    .copy()
)

total_records = len(results)
engine_count = results["engine_id"].nunique()
anomaly_count = int(results["is_anomaly"].sum())

anomaly_percentage = (
    anomaly_count / total_records * 100
    if total_records > 0
    else 0
)

engine_anomaly_count = int(
    engine_data["is_anomaly"].sum()
)


# --------------------------------------------------
# 5. Dataset overview
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{total_records:,}",
)

col2.metric(
    "Engines",
    f"{engine_count:,}",
)

col3.metric(
    "Potential Anomalies",
    f"{anomaly_count:,}",
)

col4.metric(
    "Flagged Records",
    f"{anomaly_percentage:.2f}%",
)

st.info(
    "An anomaly is a record flagged by the model, not a confirmed "
    "engine fault. The model is fitted on the full dataset being "
    "explored, so these figures are descriptive rather than "
    "independent model evaluation results."
)


# --------------------------------------------------
# 6. Sensor trend chart
# --------------------------------------------------

st.subheader(
    f"{selected_sensor} — Engine {selected_engine}"
)

normal_points = engine_data[
    ~engine_data["is_anomaly"]
]

anomaly_points = engine_data[
    engine_data["is_anomaly"]
]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    engine_data["cycle"],
    engine_data[selected_sensor],
    label="Sensor reading",
    linewidth=1.5,
)

if not normal_points.empty:
    ax.scatter(
        normal_points["cycle"],
        normal_points[selected_sensor],
        label="Model-classified normal",
        s=16,
        alpha=0.55,
    )

if not anomaly_points.empty:
    ax.scatter(
        anomaly_points["cycle"],
        anomaly_points[selected_sensor],
        label="Potential anomaly",
        marker="x",
        s=65,
    )

ax.set_xlabel("Operating Cycle")
ax.set_ylabel("Sensor Reading")
ax.set_title(
    f"{selected_sensor} over time — Engine {selected_engine}"
)
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Cross markers identify records flagged by Isolation Forest. "
    "A flagged reading is an investigation candidate, not proof "
    "of a mechanical failure."
)


# --------------------------------------------------
# 7. Selected sensor statistical summary
# --------------------------------------------------

st.subheader("Selected Sensor: Statistical Summary")

sensor_values = df[selected_sensor]

dataset_mean = sensor_values.mean()
dataset_std = sensor_values.std()
engine_mean = engine_data[selected_sensor].mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dataset Mean",
    f"{dataset_mean:.3f}",
)

col2.metric(
    "Dataset Standard Deviation",
    f"{dataset_std:.3f}",
)

col3.metric(
    "Selected Engine Mean",
    f"{engine_mean:.3f}",
)

if dataset_std > 0:
    deviation = (
        engine_mean - dataset_mean
    ) / dataset_std

    st.write(
        f"The selected engine's mean reading is "
        f"{deviation:.2f} standard deviations from "
        "the overall dataset mean for this sensor."
    )
else:
    st.info(
        "This sensor has no variation in the dataset, "
        "so a standardized deviation cannot be calculated."
    )

st.caption(
    "This is a descriptive comparison. A difference from the "
    "dataset mean does not, by itself, establish a fault."
)


# --------------------------------------------------
# 8. Filter and inspect sensor records
# --------------------------------------------------

st.subheader("Sensor Records")

record_filter = st.selectbox(
    "Choose records to display",
    [
        "All records",
        "Potential anomalies only",
        "Normal records only",
    ],
)

if record_filter == "Potential anomalies only":
    displayed_data = engine_data[
        engine_data["is_anomaly"]
    ].copy()

elif record_filter == "Normal records only":
    displayed_data = engine_data[
        ~engine_data["is_anomaly"]
    ].copy()

else:
    displayed_data = engine_data.copy()


display_columns = [
    "engine_id",
    "cycle",
    selected_sensor,
    "anomaly_prediction",
    "anomaly_score",
]

st.write(
    f"Records displayed: {len(displayed_data):,} "
    f"out of {len(engine_data):,} records for Engine "
    f"{selected_engine}."
)

if displayed_data.empty:
    st.info(
        "No records match this filter for the selected engine "
        "and current model settings."
    )
else:
    st.dataframe(
        displayed_data[display_columns].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

display_csv = displayed_data[
    display_columns
].to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download displayed records as CSV",
    data=display_csv,
    file_name=(
        f"engine_{selected_engine}_"
        f"{record_filter.lower().replace(' ', '_')}.csv"
    ),
    mime="text/csv",
    key="download_displayed_records",
)


# --------------------------------------------------
# 9. Saved model evaluation results
# --------------------------------------------------

st.subheader("Saved Model Evaluation")

evaluation_path = (
    PROCESSED_DIR / "holdout_model_comparison.csv"
)

if evaluation_path.exists():
    evaluation_df = pd.read_csv(evaluation_path)

    st.caption(
        "These results come from your earlier synthetic holdout "
        "experiment. They are separate from the live anomaly "
        "flags displayed above. Synthetic anomalies are not "
        "equivalent to real engine faults."
    )

    st.dataframe(
        evaluation_df,
        use_container_width=True,
        hide_index=True,
    )

    required_columns = {
        "model",
        "detection_rate_percent",
        "false_positive_rate_percent",
    }

    if required_columns.issubset(evaluation_df.columns):
        chart_df = evaluation_df.set_index("model")[
            [
                "detection_rate_percent",
                "false_positive_rate_percent",
            ]
        ]

        st.write("Model comparison: synthetic holdout experiment")

        st.bar_chart(chart_df)

    else:
        st.warning(
            "The evaluation file exists, but its column names "
            "do not match the expected format for the comparison "
            "chart. The table is still available above."
        )

else:
    st.warning(
        "Evaluation file not found. Run "
        "notebooks/04_holdout_evaluation.ipynb to generate it."
    )


# --------------------------------------------------
# 10. Export all flagged records for the selected engine
# --------------------------------------------------

st.subheader("Export Potential Anomalies")

flagged_engine_data = engine_data[
    engine_data["is_anomaly"]
].copy()

st.write(
    f"Engine {selected_engine} has "
    f"{len(flagged_engine_data):,} potentially anomalous "
    "records under the current model settings."
)

if flagged_engine_data.empty:
    st.info(
        "No records were flagged for this engine. "
        "Try another engine or review the model settings."
    )
else:
    anomaly_csv = flagged_engine_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download all flagged records for this engine",
        data=anomaly_csv,
        file_name=f"engine_{selected_engine}_anomalies.csv",
        mime="text/csv",
        key="download_all_engine_anomalies",
    )


# --------------------------------------------------
# 11. Project notes
# --------------------------------------------------

st.divider()

st.caption(
    "Dataset: NASA C-MAPSS FD001 — simulated turbofan "
    "engine degradation data. This dashboard does not analyze "
    "real aircraft flight telemetry. Results are experimental "
    "and should not be used for operational maintenance decisions."
)
