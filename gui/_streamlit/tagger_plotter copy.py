import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import os
import json
import sys
from scipy.stats import norm
import warnings
from influxdb_client import InfluxDBClient, QueryApi, Point, WritePrecision

warnings.simplefilter("ignore")
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from fast_tagger_gui.src.system_utils import get_secrets
from fast_tagger_gui.src.physics_utils import compute_tof_from_data
from fast_tagger_gui.src.system_utils import (
    get_secrets,
    load_path,
    update_settings_file, 
    metadata_writer,
)

db_token = get_secrets()
os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

def query_influxdb(minus_time_str, measurement_name):
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: {minus_time_str})
    |> filter(fn: (r) => r._measurement == "tagger_data")
    |> filter(fn: (r) => r.type == "{measurement_name}")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "bunch", "n_events", "channels", "timestamp"])
    '''
    try:
        result = client.query_api().query(query=query, org=INFLUXDB_ORG)
        records = []

        for table in result:
            for record in table.records:
                records.append(record.values)
        df = pd.DataFrame(records)
        df = df.rename(columns={'_time': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        column_order = ['time', 'bunch', 'n_events', 'channels', 'timestamp']
        return df[column_order]
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=['time', 'bunch', 'n_events', 'channels', 'timestamp'])


def events_numbers(og_data: pd.DataFrame):
    data = og_data.copy()[["bunch", "time", "n_events"]]
    number_of_events = data.drop_duplicates(subset=["bunch"])
    return number_of_events[['bunch', 'n_events']]

def estimate_rates(og_data: pd.DataFrame):
    data = og_data.copy()
    initial_time = data["time"].min()
    current_time = data["time"].max()
    time_diff = (current_time - initial_time).total_seconds()
    channel_counts = data["channels"].value_counts().sort_index()
    channel_counts = channel_counts.reindex(range(-1, 4), fill_value=0)
    rates = channel_counts / time_diff
    return rates


class TaggerVisualizer:
    def __init__(self):
        self.rolling_window = 10
        self.channel_names = {-1: "Trigger", 0: "A", 1: "B", 2: "C", 3: "D"}
        self.data = pd.DataFrame()

    def tof_histogram(self, data):
        try:
            tof_df = compute_tof_from_data(data)
            tofs = tof_df["tof"].values
            if len(tofs) <= 1:
                return go.Figure()
            
            mu, std = norm.fit(tofs)
            xmin, xmax = tofs.min(), tofs.max()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=tofs, 
                nbinsx=100, 
                name='TOF Data', 
                marker_color='blue', 
                opacity=0.7
            ))

            # Plot the Gaussian fit
            fig.add_trace(go.Scatter(
                x=x, 
                y=p * len(tofs) * (xmax - xmin) / 100, 
                mode='lines', 
                name='Gaussian Fit', 
                line=dict(color='red')
            ))

            # Add mean and standard deviation as a bar
            fig.add_trace(go.Scatter(
                x=[mu], 
                y=[0], 
                mode='markers', 
                marker=dict(color='green', size=10, symbol='x'), 
                name=f'Mean: {mu:.2f} μs'
            ))
            fig.add_trace(go.Scatter(
                x=[mu - std, mu + std], 
                y=[0, 0], 
                mode='markers', 
                marker=dict(color='orange', size=10, symbol='line-ew'), 
                name=f'Standard Deviation: {std:.2f} μs'
            ))

            fig.update_layout(
                xaxis_title="Time of Flight (μs)",
                yaxis_title="Events",
                showlegend=True
            )
            return fig
        except Exception as e:
            print(f"Error creating TOF histogram: {e}")
            return go.Figure()
        
    def events_per_bunch(self, og_data: pd.DataFrame):
        try:
            events_per_bunch = events_numbers(og_data)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=events_per_bunch["bunch"],
                    y=events_per_bunch["n_events"],
                    mode="lines+markers",
                    name="Events per Bunch",
                    line=dict(color="blue"),
                )
            )

            fig.update_layout(
                title="Events per Bunch",
                xaxis_title="Bunch",
                yaxis_title="Events"
            )
            return fig
        except Exception as e:
            print(f"Error creating events per bunch plot: {e}")
            return go.Figure()

    def channel_distribution(self):
        try:
            rates = estimate_rates(self.data)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=list(self.channel_names.values()),
                    y=rates.values,
                    name="Events per Second",
                    marker_color="blue"
                )
            )

            fig.update_layout(
                title="Channel Distribution",
                xaxis_title="Channel",
                yaxis_title="Events per Second"
            )
            return fig
            
        except Exception as e:
            st.error(f"Error creating channel distribution plot: {e}")
            return go.Figure()

    def events_over_time(self):
        try:
            self.data["datetime"] = pd.to_datetime(self.data["time"])
            events_over_time = self.data.set_index("datetime").resample("1S").size()

            rolling_mean = events_over_time.rolling(window=10).mean()
            rolling_std = events_over_time.rolling(window=10).std()
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
                    name="Rolling Mean (10s)",
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
                xaxis_title="Time",
                yaxis_title="Events per Second"
            )
            return fig
        except Exception as e:
            st.error(f"Error creating events over time plot: {e}")
            return go.Figure()


def main(minus_time_str, measurement_name):
    st.set_page_config(page_title="Tagger Data Visualizer", layout="wide")

    viz = TaggerVisualizer()
    status_indicator = st.sidebar.empty()

    st.sidebar.header("Settings")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 0.5, 10.0, 0.5, 1.)
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

    if "clean_time" not in st.session_state:
        st.session_state.clean_time = None

    clean_button = st.sidebar.button("Clean Data")
    if clean_button:
        st.session_state.clean_time = datetime.now(timezone.utc)

    info_box = st.sidebar.empty()

    col1, col2 = st.columns(2)
    events_over_time_plot = col1.empty()
    tof_plot = col2.empty()

    col3, col4 = st.columns(2)
    events_per_bunch_plot = col3.empty()
    channel_dist_plot = col4.empty()
    status = st.empty()

    while True:
        try:
            data = query_influxdb(minus_time_str, measurement_name)
            viz.data = data
            if st.session_state.clean_time:
                viz.data = viz.data[
                    pd.to_datetime(viz.data["time"])
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
                last_bunch_time = pd.to_datetime(viz.data["time"].max()).astimezone(timezone.utc)
                total_bunches = viz.data["bunch"].max()
                time_since_last_bunch = datetime.now(timezone.utc) - last_bunch_time

                info_box.markdown(
                    f"""
                ### Data Summary
                **Total Bunches:** {total_bunches}

                **Last Bunch Received:**
                {last_bunch_time.strftime('%Y-%m-%d %H:%M:%S %Z')}

                **Time Since Last Bunch:**
                {time_since_last_bunch.total_seconds():.2f} seconds
                """
                )

                if time_since_last_bunch.total_seconds() > 2:
                    status_indicator.markdown("🔴 Status: Offline")
                else:
                    status_indicator.markdown("🟢 Status: Ions Online")

                if use_time_range:
                    df_filtered = viz.data[
                        (viz.data["time"] >= start_time)
                        & (viz.data["time"] <= end_time)
                    ]
                    notrigger_filtered = df_filtered[df_filtered["channels"] != -1]
                else:
                    df_filtered = viz.data
                    notrigger_filtered = notrigger_df

                events_over_time_plot.plotly_chart(
                    viz.events_over_time(), use_container_width=True
                )
                tof_plot.plotly_chart(
                    viz.tof_histogram(viz.data), use_container_width=True
                )
                events_per_bunch_plot.plotly_chart(
                    viz.events_per_bunch(notrigger_filtered), use_container_width=True
                )
                channel_dist_plot.plotly_chart(
                    viz.channel_distribution(), use_container_width=True
                )
                status.success(f"Data loaded. Total events: {len(notrigger_df)}")

        except Exception as e:
            st.error(f"Error: {e}")
        time.sleep(refresh_rate)

def string_to_unix_timestamp(date_string):
    dt = datetime.strptime(date_string, "%Y_%m_%d_%H_%M_%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    
if __name__ == "__main__":
    file_location = load_path()["saving_file"]
    measurement_name = file_location.split("monitor_")[-1].split(".")[0]
    print(f"Measurement Name: {measurement_name}")
    main(string_to_unix_timestamp(measurement_name), measurement_name)