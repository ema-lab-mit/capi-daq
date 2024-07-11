import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
import logging
import sys
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from fast_tagger_gui.src.physics_utils import time_to_flops, compute_tof_from_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings():
    settings_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        return settings.get("save_path", "")
    except Exception as e:
        logging.error(f"Error loading settings: {e}")
        st.error("Error loading settings. Please check the settings file.")
        return ""

class TaggerVisualizer:
    def __init__(self):
        self.rolling_window = 10
        self.channel_names = {-1: "Trigger", 0: "A", 1: "B", 2: "C", 3: "D"}

    def tof_histogram(self, data):
        try:
            tof_df = compute_tof_from_data(data)
            tofs = tof_df["tof"].values * 1e6
            if len(tofs) <= 1:
                return go.Figure()
            
            fig = px.histogram(tof_df, x="tof", nbins=100,
                               title="Time-of-Flight Distribution",
                               labels={"tof": "Time of Flight (μs)", "count": "Events"})
            fig.update_layout(showlegend=False, xaxis_title="Time of Flight (μs)", yaxis_title="Events")
            return fig
        except Exception as e:
            logging.error(f"Error creating TOF histogram: {e}")
            st.error("Error creating TOF histogram.")
            return go.Figure()

    def events_per_bunch(self, data):
        try:
            rolled_df = data.groupby("bunch").n_events.sum().rolling(self.rolling_window).sum()
            bunches = rolled_df.index
            num_events = rolled_df.values
            
            fig = px.line(x=bunches, y=num_events, 
                          title="Events per Bunch (Rolling Average)",
                          labels={"x": "Bunch Number", "y": "Number of Events"})
            fig.update_layout(showlegend=False, xaxis_title="Bunch Number", yaxis_title="Number of Events")
            return fig
        except Exception as e:
            logging.error(f"Error creating events per bunch plot: {e}")
            st.error("Error creating events per bunch plot.")
            return go.Figure()

    def channel_distribution(self, data):
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
            return fig
        except Exception as e:
            logging.error(f"Error creating channel distribution plot: {e}")
            st.error("Error creating channel distribution plot.")
            return go.Figure()

    def events_over_time(self, data):
        try:
            # data['datetime'] = pd.to_datetime(data['synced_time'], errors='coerce')
            data.iloc[:,"synced_time"] = data['synced_time'].apply(lambda x: pd.to_datetime(x, errors='coerce'))
            events_over_time = data.set_index('datetime').resample('1S').size()
            
            fig = px.line(x=events_over_time.index, y=events_over_time.values,
                          title="Events Over Time",
                          labels={"x": "Time", "y": "Events per Second"})
            fig.update_layout(showlegend=False, xaxis_title="Time", yaxis_title="Events per Second")
            return fig
        except Exception as e:
            logging.error(f"Error creating events over time plot: {e}")
            st.error("Error creating events over time plot.")
            return go.Figure()

def load_data(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            return validate_data(df)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            st.error(f"Error loading data from {file_path}")
            return pd.DataFrame()
    else:
        logging.error(f"File not found: {file_path}")
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

def validate_data(df):
    try:
        required_columns = ['timestamp', 'synced_time', 'bunch', 'n_events', 'channels']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    except Exception as e:
        logging.error(f"Data validation error: {e}")
        st.error("Data validation error.")
        return pd.DataFrame()

def main(predefine_data_path=None):
    st.set_page_config(page_title="Tagger Data Visualizer", layout="wide")
    st.title("Tagger Data Visualizer")
    status = st.empty()
    status_indicator = st.sidebar.empty()
    viz = TaggerVisualizer()
    if not predefine_data_path:
        predefine_data_path = load_settings()
        
    if os.path.exists(predefine_data_path):
        status.warning("""No data available. Waiting for data...
                       Make sure you click the ignore on the debug pop-up to continue.
                       """)
        status_indicator.markdown("⚪ Status: No Data")
        
    data_path = predefine_data_path

    st.sidebar.header("Settings")
    data_path = st.sidebar.text_input("Data File Path", predefine_data_path)

    with st.sidebar.expander("Additional Settings"):
        refresh_rate = st.slider("Refresh Rate (seconds)", 0.5, 10.0, 2.0, 0.5)
        rolling_window = st.slider("Rolling Window", 1, 100, 10)
    viz.rolling_window = rolling_window

    st.sidebar.subheader("Events over Time Range")
    use_time_range = st.sidebar.checkbox("Use custom time range", value=False)
    if use_time_range:
        col1, col2 = st.sidebar.columns(2)
        start_time = col1.time_input("Start Time", datetime.now().time())
        end_time = col2.time_input("End Time", (datetime.now() + timedelta(hours=1)).time())
    else:
        start_time, end_time = None, None

    clean_button = st.sidebar.button("Clean Data")
    if clean_button:
        st.session_state.clean_time = datetime.now()

    info_box = st.sidebar.empty()

    col1, col2 = st.columns(2)
    events_over_time_plot = col1.empty()
    tof_plot = col2.empty()
    
    col3, col4 = st.columns(2)
    events_per_bunch_plot = col3.empty()
    channel_dist_plot = col4.empty()
    

    while True:
        df = load_data(data_path)
        if 'clean_time' in st.session_state:
            df = df[pd.to_datetime(df['synced_time']) >= st.session_state.clean_time]

        notrigger_df = df[df['channels'] != -1]
        if df.empty:
            status.warning("No data available. Waiting for data...")
            status_indicator.markdown("⚪ Status: No Data")
        elif notrigger_df.empty:
            status.warning("No events found. Only trigger events are present. Waiting for data...")
            status_indicator.markdown("⚪ Status: No Ions Detected")
        else:
            status.success(f"Data loaded. Total events: {len(notrigger_df)}")
            last_bunch_time = pd.to_datetime(df['synced_time'].max())
            total_bunches = df['bunch'].max()
            time_since_last_bunch = datetime.now() - last_bunch_time
            
            info_box.markdown(f"""
            ### Data Summary
            **Total Bunches:** {total_bunches}
            
            **Last Bunch Received:**
            {last_bunch_time.strftime('%Y-%m-%d %H:%M:%S')}
            
            **Time Since Last Bunch:**
            {time_since_last_bunch.total_seconds():.2f} seconds
            """)
            
            if time_since_last_bunch.total_seconds() > 1.5:
                status_indicator.markdown("🔴 Status: Offline")
            else:
                status_indicator.markdown("🟢 Status: Ions Online")
            
            if use_time_range:
                df['time'] = pd.to_datetime(df['synced_time'], errors='coerce').dt.time
                df_filtered = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                notrigger_filtered = df_filtered[df_filtered['channels'] != -1]
            else:
                df_filtered = df
                notrigger_filtered = notrigger_df

            events_over_time_plot.plotly_chart(viz.events_over_time(notrigger_filtered), use_container_width=True)
            tof_plot.plotly_chart(viz.tof_histogram(notrigger_filtered), use_container_width=True)
            events_per_bunch_plot.plotly_chart(viz.events_per_bunch(notrigger_filtered), use_container_width=True)
            channel_dist_plot.plotly_chart(viz.channel_distribution(df), use_container_width=True)

        time.sleep(refresh_rate)

if __name__ == "__main__":
    main()
