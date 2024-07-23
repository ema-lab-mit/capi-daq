import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timezone, timedelta
import time
import os
import sys
from scipy.stats import norm
import warnings
from influxdb_client import InfluxDBClient, QueryApi
import pandas as pd

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
    result = client.query_api().query(query=query, org=INFLUXDB_ORG)
    records = []

    for table in result:
        for record in table.records:
            record_dict = record.values
            record_dict['time'] = record_dict['_time']
            records.append(record_dict)

    return records

class TaggerVisualizer:
    def __init__(self):
        self.rolling_window = 10
        self.channel_names = {-1: "Trigger", 0: "A", 1: "B", 2: "C", 3: "D"}

    def tof_histogram(self, data, theme):
        try:
            tofs = compute_tof_from_data(data)
            if len(tofs) <= 1:
                return go.Figure()
            mu, std = norm.fit(tofs)
            xmin, xmax = tofs.min(), tofs.max()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            fig = go.Figure()

            # Plot the histogram
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
                title="Time-of-Flight Distribution",
                xaxis_title="Time of Flight (μs)",
                yaxis_title="Events",
                showlegend=True,
                template=theme
            )
            return fig
        except Exception as e:
            print(f"Error creating TOF histogram: {e}")
            return go.Figure()

    def events_per_bunch(self, data, theme):
        try:
            rolled_df = data.groupby("bunch").n_events.sum().rolling(self.rolling_window).sum()
            bunches = rolled_df.index
            num_events = rolled_df.values
            
            fig = px.line(x=bunches, y=num_events, 
                          title="Events per Bunch (Rolling Average)",
                          labels={"x": "Bunch Number", "y": "Number of Events"})
            fig.update_layout(showlegend=False, xaxis_title="Bunch Number", yaxis_title="Number of Events")
            fig.update_layout(template=theme)
            return fig
        except Exception as e:
            print(f"Error creating events per bunch plot: {e}")
            return go.Figure()

    def channel_distribution(self, data, theme):
        try:
            channel_counts = data['channels'].value_counts().sort_index()
            channel_counts = channel_counts.reindex(range(-1, 4), fill_value=0)
            
            fig = go.Figure()
            for channel, count in channel_counts.items():
                fig.add_trace(go.Bar(
                    x=[self.channel_names.get(channel, f"Unknown ({channel})")],
                    y=[count],
                    name=self.channel_names.get(channel, f"Unknown ({channel})"),
                    text=[count],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Channel Distribution",
                xaxis_title="Channel",
                yaxis_title="Count",
                showlegend=False,
                height=400
            )
            fig.update_layout(template=theme)
            return fig
        except Exception as e:
            print(f"Error creating channel distribution plot: {e}")
            return go.Figure()

    def events_over_time(self, data, theme):
        try:
            data['datetime'] = pd.to_datetime(data['synced_time'], errors='coerce')
            events_over_time = data.set_index('datetime').resample('1S').size()
            
            rolling_mean = events_over_time.rolling(window=60).mean()
            rolling_std = events_over_time.rolling(window=60).std()
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=events_over_time.index, 
                y=events_over_time.values,
                mode='lines',
                name='Events per Second',
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean,
                mode='lines',
                name='Rolling Mean (60s)',
                line=dict(color='orange')
            ))

            fig.add_trace(go.Scatter(
                x=upper_band.index,
                y=upper_band,
                mode='lines',
                name='Upper Bollinger Band',
                line=dict(color='green'),
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.2)'
            ))

            fig.add_trace(go.Scatter(
                x=lower_band.index,
                y=lower_band,
                mode='lines',
                name='Lower Bollinger Band',
                line=dict(color='red'),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ))

            fig.update_layout(
                title="Events Over Time with Bollinger Bands",
                xaxis_title="Time",
                yaxis_title="Events per Second",
                template=theme
            )
            return fig
        except Exception as e:
            print(f"Error creating events over time plot: {e}")
            return go.Figure()

    def events_vs_wavenumber_1(self, data, theme):
        try:
            wavenumber_df = data.dropna(subset=['wavenumber'])
            if wavenumber_df.empty:
                return go.Figure()

            fig = px.line(
                x=wavenumber_df['wavenumber'], 
                y=wavenumber_df['n_events'], 
                title="Events vs Wavenumber",
                labels={"x": "Wavenumber", "y": "Number of Events"}
            )
            fig.update_layout(template=theme)
            return fig
        except Exception as e:
            print(f"Error creating events vs wavenumber plot: {e}")
            return go.Figure()


def main(minus_time_str, measurement_name):
    st.set_page_config(page_title="Tagger Data Visualizer", layout="wide")
    st.title("Tagger Data Visualizer")

    viz = TaggerVisualizer()

    st.sidebar.header("Settings")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 0.2, 10.0, 0.1, 0.2)
    rolling_window = st.sidebar.slider("Rolling Window", 1, 100, 10)
    viz.rolling_window = rolling_window

    info_box = st.sidebar.empty()

    col1, col2 = st.columns(2)
    events_vs_wavenumber_1_plot = col1.empty()
    events_over_time_plot = col2.empty()
    
    col3, col4 = st.columns(2)
    channel_dist_plot = col3.empty()
    tof_plot = col4.empty()

    col5 = st.columns(1)
    events_per_bunch_plot = col5[0].empty()
    
    status = st.empty()

    while True:
        try:
            data = query_influxdb(minus_time_str, measurement_name)
            if not data:
                status.warning("No data available. Waiting for data...")
            else:
                last_bunch_time = max(d['time'] for d in data)
                total_bunches = max(d['bunch'] for d in data)
                time_since_last_bunch = datetime.now(timezone.utc) - last_bunch_time
                
                info_box.markdown(f"""
                ### Data Summary
                **Total Bunches:** {total_bunches}
                
                **Last Bunch Received:**
                {last_bunch_time.strftime('%Y-%m-%d %H:%M:%S')}
                
                **Time Since Last Bunch:**
                {time_since_last_bunch.total_seconds():.2f} seconds
                """)
                
                events_vs_wavenumber_1_plot.plotly_chart(viz.events_vs_wavenumber_1(data), use_container_width=True)
                events_over_time_plot.plotly_chart(viz.events_over_time(data), use_container_width=True)
                tof_plot.plotly_chart(viz.tof_histogram(data), use_container_width=True)
                events_per_bunch_plot.plotly_chart(viz.events_per_bunch(data), use_container_width=True)
                channel_dist_plot.plotly_chart(viz.channel_distribution(data), use_container_width=True)
                status.success(f"Data loaded. Total events: {len(data)}")
                
        except Exception as e:
            print(f"Error: {e}")
            status.error("An error occurred while loading data.")
        time.sleep(refresh_rate)


def string_to_unix_timestamp(date_string):
    dt = datetime.strptime(date_string, "%Y_%m_%d_%H_%M_%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

if __name__ == "__main__":
    file_location = load_path()["saving_file"]
    measurement_name = file_location.split("scan_")[-1].split(".")[0]
    print(f"Measurement Name: {measurement_name}")
    main(string_to_unix_timestamp(measurement_name), measurement_name)
