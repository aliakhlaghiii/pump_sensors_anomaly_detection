from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from detector import SENSORS, features, read_sensor_csv, sample_readings, score, sensor_window_means

st.set_page_config(page_title="Pump Signal Monitor", page_icon="📈", layout="wide")


@st.cache_resource
def load_model():
    with np.load(Path(__file__).with_name("pump_model.npz"), allow_pickle=False) as artifact:
        return dict(artifact)


@st.cache_resource
def load_sensor_reference():
    with np.load(Path(__file__).with_name("sensor_reference.npz"), allow_pickle=False) as reference:
        return {key: pd.Series(reference[key], index=SENSORS) for key in ("lower", "upper")}


@st.cache_data
def analyze(frame):
    model = load_model()
    _, values, missing = features(frame, model["lower"], model["upper"], model["medians"])
    scores = pd.Series(score(values, model), index=missing.index).where(missing.le(0.2))
    return sensor_window_means(frame), scores


def chart(signal, outside, scores, flags, threshold, lower, upper):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=signal.index, y=signal, name=signal.name,
                             line={"color": "#00897b"}, hovertemplate="%{y:.6g}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=signal.index[outside], y=signal[outside], mode="markers",
                             name="Sensor outside reference", hovertemplate="%{y:.6g}",
                             marker={"color": "#e05a3f", "size": 9}), row=1, col=1)
    reference_lines = [(lower, "Lower limit", "bottom right"), (upper, "Upper limit", "top right")]
    if lower == upper:
        reference_lines = [(lower, "Lower and upper limit", "top right")]
    for limit, label, position in reference_lines:
        if pd.notna(limit):
            fig.add_hline(y=limit, line_dash="dash", line_color="#8796a9", row=1, col=1,
                          annotation_text=f"{label}: {limit:.8g}", annotation_position=position,
                          annotation_font_size=12, annotation_bgcolor="rgba(110, 125, 145, 0.18)")
    fig.add_trace(go.Scatter(x=scores.index, y=scores, name="PCA score",
                             line={"color": "#457bba"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=scores.index[flags], y=scores[flags], mode="markers",
                             name="Pump above PCA threshold", marker={"color": "#e05a3f", "size": 9}), row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#8796a9", row=2, col=1,
                  annotation_text="Threshold")
    fig.update_yaxes(title_text="Sensor 15-min mean", row=1, col=1)
    fig.update_yaxes(title_text="Pump PCA score", row=2, col=1)
    fig.update_layout(height=520, margin={"l": 12, "r": 24, "t": 12, "b": 12},
                      hovermode="x unified", legend={"orientation": "h", "y": -0.18})
    return fig


st.title("Pump Signal Monitor")
st.write("Explore unusual patterns in pump sensor readings with a trained PCA detector.")
mode = st.radio("Data source", ["Try sample data", "Upload CSV"], horizontal=True)
if mode == "Try sample data":
    frame = sample_readings(load_model())
    st.caption("Simulated readings with two injected deviations. The detector was trained on public pump sensor data.")
    st.download_button("Download sample CSV", frame.to_csv().encode(), "pump_sample.csv", "text/csv")
else:
    st.caption("Use one-minute readings from a compatible sensor setup. Download the sample for the required columns.")
    with st.expander("CSV format"):
        st.code("timestamp," + ",".join(SENSORS), language=None)
        st.write("Use at least 60 consecutive minute readings. Timestamps must have no timezone offset. "
                 "Four full clock-aligned 15-minute windows are required.")
    uploaded = st.file_uploader("Sensor readings", type="csv")
    if uploaded is None:
        st.stop()
    try:
        frame = read_sensor_csv(uploaded)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(str(error))
        st.stop()

try:
    readings, scores = analyze(frame)
except ValueError as error:
    st.error(str(error))
    st.stop()

threshold = float(load_model()["threshold"])
flags = scores.gt(threshold)
scored, flagged, skipped = st.columns(3)
scored.metric("Scored windows", f"{scores.count():,}")
flagged.metric("Pump flagged windows", f"{flags.sum():,}")
skipped.metric("Skipped windows", f"{scores.isna().sum():,}")
sensor = st.selectbox("Sensor to display", SENSORS)
signal = readings[sensor]
reference = load_sensor_reference()
lower, upper = reference["lower"][sensor], reference["upper"][sensor]
below, above = signal.lt(lower), signal.gt(upper)
outside = below | above
st.subheader(f"{sensor} anomalies")
if pd.notna(lower) and pd.notna(upper):
    st.write(f"**{outside.sum():,} out of {signal.count():,} measured windows outside reference**"
             f" · Reference: {lower:.6g}–{upper:.6g}")
else:
    st.write("No reference range is available for this sensor.")
st.caption(f"Top chart: selected sensor and its reference limits. Bottom chart: PCA score across all sensors. "
           f"{signal.isna().sum():,} sensor windows have fewer than 12 usable readings.")
st.plotly_chart(chart(signal, outside, scores, flags, threshold, lower, upper), width="stretch")

if outside.any():
    events = signal[outside].rename("15-minute mean").to_frame()
    events["Threshold crossed"] = np.where(below[outside], lower, upper)
    events["Distance beyond threshold"] = (events["15-minute mean"] - events["Threshold crossed"]).abs()
    events["Direction"] = np.where(below[outside], "Below reference", "Above reference")
    events.index = events.index.strftime("%Y-%m-%d %H:%M")
    st.dataframe(events.rename_axis("Window start").reset_index(), hide_index=True)

st.subheader("Inspect a window")
choices = list(scores.index)
first_flag = int(np.flatnonzero(flags)[0]) if flags.any() else 0
selected = st.selectbox("Window start", choices, index=first_flag, format_func=lambda time: time.strftime("%Y-%m-%d %H:%M"))
value = scores.loc[selected]
if pd.isna(value):
    st.write("Skipped because more than 20% of sensors were missing or filtered in a contributing window.")
else:
    status = "Flagged" if flags.loc[selected] else "Within threshold"
    st.write(f"**{status}** · Score {value:.3g} · Threshold {threshold:.3g}")
st.dataframe(readings.loc[selected].rename("15-minute mean").rename_axis("Sensor").reset_index(),
             hide_index=True)
st.caption("Scores measure how unusual a pattern is relative to the training data. "
           "A flagged window is a signal for review. The model does not estimate failure probability.")
