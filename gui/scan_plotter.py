import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
import logging
import sys
from scipy.stats import norm
import warnings
warnings.simplefilter("ignore")
# np.warnings.filterwarnings('ignore')

father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from fast_tagger_gui.src.physics_utils import time_to_flops, compute_tof_from_data

DATA_FORMAT = "parquet" # "csv"
SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings():
    try:
        with open(SETTINGS_PATH, 'r') as f:
            settings = json.load(f)
        return settings.get("saving_file", "")
    except Exception as e:
        print(f"Error loading settings: {e}")
        return ""

class TaggerVisualizer:
    def __init__(self):
        self.rolling_window = 10
        self.channel_names = {-1: "Trigger", 0: "A", 1: "B", 2: "C", 3: "D"}

    def tof_histogram(self, data, theme):
        try:
            tof_df = compute_tof_from_data(data)
            tofs = tof_df["tof"].values * 1e6
            if len(tofs) <= 1:
                return go.Figure()
            
            # Fit a Gaussian to the data
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

    def events_vs_wavenumber(self, data, theme):
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

def load_data(file_path):
    if os.path.exists(file_path):
        try:
            if DATA_FORMAT == "csv":
                df = pd.read_csv(file_path)
            elif DATA_FORMAT == "parquet":
                df = pd.read_parquet(file_path)
            return validate_data(df)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

def validate_data(df):
    try:
        required_columns = ['timestamp', 'synced_time', 'bunch', 'n_events', 'channels', 'wavenumber']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    except Exception as e:
        print(f"Data validation error: {e}")
        return pd.DataFrame()

def main(predefine_data_path=None):
    st.set_page_config(page_title="Tagger Data Visualizer", layout="wide")
    st.title("Tagger Data Visualizer")

    viz = TaggerVisualizer()
    if not predefine_data_path:
        predefine_data_path = load_settings()
    status_indicator = st.sidebar.empty()
        
    if not os.path.exists(predefine_data_path):
        st.warning("""No data available. Waiting for data...
                       Make sure you click the ignore on the debug pop-up to continue.
                       """)
        status_indicator.markdown("⚪ Status: No Data")
        
    data_path = predefine_data_path

    st.sidebar.header("Settings")
    data_path = st.sidebar.text_input("Data File Path", predefine_data_path)

    with st.sidebar.expander("Additional Settings"):
        refresh_rate = st.slider("Refresh Rate (seconds)", 0.5, 10.0, 0.5, 0.5)
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

    theme_button = st.sidebar.button("Toggle Dark/Light Mode")
    if 'theme' not in st.session_state:
        st.session_state.theme = "plotly"

    if theme_button:
        if st.session_state.theme == "plotly_dark":
            st.session_state.theme = "plotly"
        else:
            st.session_state.theme = "plotly_dark"
    
    current_theme = st.session_state.theme

    info_box = st.sidebar.empty()

    col1, col2 = st.columns(2)
    events_vs_wavenumber_plot = col1.empty()
    events_over_time_plot = col2.empty()
    
    col3, col4 = st.columns(2)
    channel_dist_plot = col3.empty()
    tof_plot = col4.empty()

    col5, col6 = st.columns(2)
    events_per_bunch_plot = col5.empty()
    
    status = st.empty()

    while True:
        try:
            data_path = load_settings()
            print(data_path)
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
                
                if time_since_last_bunch.total_seconds() > 2:
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

                events_vs_wavenumber_plot.plotly_chart(viz.events_vs_wavenumber(notrigger_filtered, current_theme), use_container_width=True)
                events_over_time_plot.plotly_chart(viz.events_over_time(notrigger_filtered, current_theme), use_container_width=True)
                tof_plot.plotly_chart(viz.tof_histogram(notrigger_filtered, current_theme), use_container_width=True)
                events_per_bunch_plot.plotly_chart(viz.events_per_bunch(notrigger_filtered, current_theme), use_container_width=True)
                channel_dist_plot.plotly_chart(viz.channel_distribution(df, current_theme), use_container_width=True)
                status.success(f"Data loaded. Total events: {len(notrigger_df)}")
                
        except Exception as e:
            print("Error")
        time.sleep(refresh_rate)

if __name__ == "__main__":
    main()
