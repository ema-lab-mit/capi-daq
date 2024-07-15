import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time
import queue
import threading
import json
from confluent_kafka import Consumer, KafkaException
from scipy.stats import norm

KAFKA_TOPIC = "daq_data"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# Kafka consumer configuration
consumer_conf = {
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
    "group.id": "streamlit_group",
    "auto.offset.reset": "latest",
}

consumer = Consumer(consumer_conf)
consumer.subscribe([KAFKA_TOPIC])

# Queue to store streaming data
data_queue = queue.Queue()


def kafka_consumer_thread():
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                st.error(f"Kafka error: {msg.error()}")
                break
        data = json.loads(msg.value().decode("utf-8"))
        data_queue.put(data)


consumer_thread = threading.Thread(target=kafka_consumer_thread, daemon=True)
consumer_thread.start()


class TaggerVisualizer:
    def __init__(self):
        self.rolling_window = 10
        self.channel_names = {-1: "Trigger", 0: "A", 1: "B", 2: "C", 3: "D"}
        self.data = pd.DataFrame()

    def update_data(self, new_data):
        self.data = pd.concat([self.data, pd.DataFrame([new_data])], ignore_index=True)
        self.data["timestamp"] = pd.to_numeric(self.data["timestamp"], errors="coerce")
        self.data["synced_time"] = pd.to_datetime(
            self.data["synced_time"], errors="coerce"
        )

    def tof_histogram(self):
        try:
            tofs = self.data["timestamp"].diff().dropna() * 1e-6
            if len(tofs) <= 1:
                return go.Figure()

            mu, std = norm.fit(tofs)
            xmin, xmax = tofs.min(), tofs.max()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    y=tofs,
                    nbinsy=100,
                    name="TOF Data",
                    marker_color="blue",
                    opacity=0.7,
                    orientation="h",
                )
            )

            fig.add_trace(
                go.Scatter(
                    y=x,
                    x=-p * len(tofs) * (xmax - xmin) / 100,
                    mode="lines",
                    name="Gaussian Fit",
                    line=dict(color="red"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    y=[mu],
                    x=[0],
                    mode="markers",
                    marker=dict(color="green", size=10, symbol="x"),
                    name=f"Mean: {mu:.2f} μs",
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=[mu - std, mu + std],
                    x=[0, 0],
                    mode="markers",
                    marker=dict(color="orange", size=10, symbol="line-ew"),
                    name=f"Standard Deviation: {std:.2f} μs",
                )
            )

            fig.update_layout(
                title="Time-of-Flight Distribution",
                yaxis_title="Time of Flight (μs)",
                xaxis_title="Events",
                showlegend=True,
            )
            return fig
        except Exception as e:
            st.error(f"Error creating TOF histogram: {e}")
            return go.Figure()

    def events_per_bunch(self):
        try:
            rolled_df = (
                data.groupby("bunch").n_events.sum().rolling(self.rolling_window).sum()
            )
            bunches = rolled_df.index
            num_events = rolled_df.values

            fig = px.line(
                x=bunches,
                y=num_events,
                title="Events per Bunch (Rolling Average)",
                labels={"x": "Bunch Number", "y": "Number of Events"},
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Bunch Number",
                yaxis_title="Number of Events",
            )
            return fig
        except Exception as e:
            st.error(f"Error creating events per bunch plot: {e}")
            return go.Figure()

    def events_over_time(self):
        try:
            events_over_time = self.data.set_index("synced_time").resample("1S").size()

            rolling_mean = events_over_time.rolling(window=60).mean()
            rolling_std = events_over_time.rolling(window=60).std()
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=events_over_time.index,
                    y=events_over_time.values,
                    mode="lines",
                    name="Events per Second",
                    line=dict(color="blue"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=rolling_mean.index,
                    y=rolling_mean,
                    mode="lines",
                    name="Rolling Mean (60s)",
                    line=dict(color="orange"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=upper_band.index,
                    y=upper_band,
                    mode="lines",
                    name="Upper Bollinger Band",
                    line=dict(color="green"),
                    fill="tonexty",
                    fillcolor="rgba(0, 255, 0, 0.2)",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=lower_band.index,
                    y=lower_band,
                    mode="lines",
                    name="Lower Bollinger Band",
                    line=dict(color="red"),
                    fill="tonexty",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                )
            )

            fig.update_layout(
                title="Events Over Time with Bollinger Bands",
                xaxis_title="Time",
                yaxis_title="Events per Second",
            )
            return fig
        except Exception as e:
            st.error(f"Error creating events over time plot: {e}")
            return go.Figure()

    def events_vs_wavenumber(self):
        try:
            wavenumber_df = self.data.dropna(subset=["wavenumber"])
            if wavenumber_df.empty:
                return go.Figure()

            fig = px.scatter(
                x=wavenumber_df["wavenumber"],
                y=wavenumber_df["n_events"],
                title="Events vs Wavenumber",
                labels={"x": "Wavenumber", "y": "Number of Events"},
            )
            fig.update_yaxes(range=[0, wavenumber_df["n_events"].max()])
            return fig
        except Exception as e:
            st.error(f"Error creating events vs wavenumber plot: {e}")
            return go.Figure()

    def delta_t_vs_wavenumber(self):
        try:
            delta_t = self.data["timestamp"].diff().dropna()
            delta_t = delta_t / 1e9  # convert from ns to seconds
            wavenumber_df = self.data.dropna(subset=["wavenumber"])
            if wavenumber_df.empty or delta_t.empty:
                return go.Figure()
            heatmap_data = pd.DataFrame(
                {"delta_t": delta_t, "wavenumber": wavenumber_df["wavenumber"][1:]}
            )

            fig = px.density_heatmap(
                heatmap_data,
                x="wavenumber",
                y="delta_t",
                title="Delta t vs Wavenumber",
                labels={"x": "Wavenumber", "y": "Delta t (s)"},
            )
            return fig
        except Exception as e:
            st.error(f"Error creating delta t vs wavenumber heatmap: {e}")
            return go.Figure()


def main():
    st.set_page_config(page_title="Tagger Data Visualizer", layout="wide")

    viz = TaggerVisualizer()
    status_indicator = st.sidebar.empty()

    st.sidebar.header("Settings")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 0.5, 10.0, 0.5, 0.5)
    rolling_window = st.sidebar.slider("Rolling Window", 1, 100, 10)
    viz.rolling_window = rolling_window

    st.sidebar.subheader("Events over Time Range")
    use_time_range = st.sidebar.checkbox("Use custom time range", value=False)
    if use_time_range:
        col1, col2 = st.sidebar.columns(2)
        start_time = col1.time_input("Start Time", datetime.now().time())
        end_time = col2.time_input(
            "End Time", (datetime.now() + timedelta(hours=1)).time()
        )
    else:
        start_time, end_time = None, None

    clean_button = st.sidebar.button("Clean Data")
    if clean_button:
        st.session_state.clean_time = datetime.now()

    info_box = st.sidebar.empty()

    col1, col2 = st.columns(2)
    events_vs_wavenumber_plot = col1.empty()
    events_over_time_plot = col2.empty()

    col3, col4 = st.columns(2)
    delta_t_vs_wavenumber_plot = col3.empty()
    tof_plot = col4.empty()

    status = st.empty()

    while True:
        try:
            while not data_queue.empty():
                new_data = data_queue.get()
                viz.update_data(new_data)

            if "clean_time" in st.session_state:
                viz.data = viz.data[
                    pd.to_datetime(viz.data["synced_time"])
                    >= st.session_state.clean_time
                ]

            notrigger_df = viz.data[viz.data["channels"] != -1]
            if viz.data.empty:
                status.warning("No data available. Waiting for data...")
                status_indicator.markdown("⚪ Status: No Data")
            elif notrigger_df.empty:
                status.warning(
                    "No events found. Only trigger events are present. Waiting for data..."
                )
                status_indicator.markdown("⚪ Status: No Ions Detected")
            else:
                last_bunch_time = pd.to_datetime(viz.data["synced_time"].max())
                total_bunches = viz.data["bunch"].max()
                time_since_last_bunch = datetime.now() - last_bunch_time

                info_box.markdown(
                    f"""
                ### Data Summary
                **Total Bunches:** {total_bunches}

                **Last Bunch Received:**
                {last_bunch_time.strftime('%Y-%m-%d %H:%M:%S')}

                **Time Since Last Bunch:**
                {time_since_last_bunch.total_seconds():.2f} seconds
                """
                )

                if time_since_last_bunch.total_seconds() > 2:
                    status_indicator.markdown("🔴 Status: Offline")
                else:
                    status_indicator.markdown("🟢 Status: Ions Online")

                if use_time_range:
                    viz.data["time"] = pd.to_datetime(
                        viz.data["synced_time"], errors="coerce"
                    ).dt.time
                    df_filtered = viz.data[
                        (viz.data["time"] >= start_time)
                        & (viz.data["time"] <= end_time)
                    ]
                    notrigger_filtered = df_filtered[df_filtered["channels"] != -1]
                else:
                    df_filtered = viz.data
                    notrigger_filtered = notrigger_df

                events_vs_wavenumber_plot.plotly_chart(
                    viz.events_vs_wavenumber(), use_container_width=True
                )
                events_over_time_plot.plotly_chart(
                    viz.events_over_time(), use_container_width=True
                )
                tof_plot.plotly_chart(viz.tof_histogram(), use_container_width=True)
                delta_t_vs_wavenumber_plot.plotly_chart(
                    viz.delta_t_vs_wavenumber(), use_container_width=True
                )
                status.success(f"Data loaded. Total events: {len(notrigger_df)}")

        except Exception as e:
            st.error(f"Error: {e}")
        time.sleep(refresh_rate)


if __name__ == "__main__":
    main()
