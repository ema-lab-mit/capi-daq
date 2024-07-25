import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
import os
import sys
import warnings
from influxdb_client import InfluxDBClient
from scipy.stats import norm

warnings.simplefilter("ignore")

# Set up paths and tokens
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from fast_tagger_gui.src.system_utils import get_secrets, load_path
from fast_tagger_gui.src.physics_utils import compute_tof_from_data
import time

db_token = get_secrets()
os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"
NBATCH = 1_000
TOTAL_MAX_POINTS = int(1e8)

default_settings = {
    "tof_hist_nbins": 100,
    "tof_hist_min": 1e-6,
    "tof_hist_max": 20e-6,
    "rolling_window": 100,
}

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

class PlotGenerator:
    def __init__(self, settings_dict: dict = default_settings):
        self.settings_dict = settings_dict
        self.tof_hist_nbins = settings_dict.get("tof_hist_nbins", 100)
        self.tof_hist_min = settings_dict.get("tof_hist_min", 0)
        self.tof_hist_max = settings_dict.get("tof_hist_max", 20e-6)
        self.rolling_window = settings_dict.get("rolling_window", 100)
        
        self._historic_timeseries_columns = ["bunch", "timestamp", "n_events", "channel", "wn_1", "wn_2", "wn_3", "wn_4"]
        self.last_loaded_time = time.time()
        self.historical_data = pd.DataFrame(columns=self._historic_timeseries_columns)
        self.historical_event_numbers = np.array([])
        self.historical_events_times = np.array([])
        self.historical_trigger_numbers = np.array([])
        self.first_time = time.time()
        
        self.total_events = 0
        self.tof_mean = 0
        self.tof_var = 0
        self.last_rates = pd.Series()
        
        self.tof_histogram_bins = np.linspace(self.tof_hist_min, self.tof_hist_max, self.tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)
        

    def _update_tof_statistics(self, unseen_new_data):
        events_data = unseen_new_data.query("channel != -1")
        self.total_events += len(events_data)
        events_offset = events_data["time_offset"].values
        new_hist_counts, _ = np.histogram(events_offset, bins=self.tof_histogram_bins)
        self.histogram_counts = self.histogram_counts + new_hist_counts
        # Now use this histogram to estimate the mean and variance of the time of flight
        self.tof_mean = np.average(self.tof_histogram_bins[:-1], weights=self.histogram_counts)
        self.tof_var = np.average((self.tof_histogram_bins[:-1] - self.tof_mean)**2, weights=self.histogram_counts)
        
    def _update_historical_data(self, unseen_new_data):
        self.historical_event_numbers = np.append(self.historical_event_numbers, unseen_new_data.query("channel !=-1")["n_events"].values)
        self.historical_events_times = np.append(self.historical_events_times, unseen_new_data.query("channel !=-1")["timestamp"].values)
        rolled_new_data = unseen_new_data[self._historic_timeseries_columns].rolling(window=self.rolling_window).mean()
        self.historical_data = pd.concat([self.historical_data, rolled_new_data], ignore_index=True)
        
    def update_content(self, new_data):
        unseen_new_data = new_data[new_data['timestamp'] > self.last_loaded_time]
        self.last_loaded_time = new_data['timestamp'].max()
        self.unseen_new_data = unseen_new_data
        if unseen_new_data.empty:
            print("No new data to update!")
            return
        self._update_tof_statistics(unseen_new_data)
        self._update_historical_data(unseen_new_data)
        if self.historical_data.shape[0] > TOTAL_MAX_POINTS:
            self.historical_data = self.historical_data.tail(TOTAL_MAX_POINTS)
        # Append the new data to the historical data
        self.historical_data = pd.concat([self.historical_data, unseen_new_data], ignore_index=True)

    def plot_events_over_time(self, max_points=1_000, roll=50):
        fig = go.Figure()
        if self.historical_data.empty:
            return fig
        events = self.historical_event_numbers
        times = self.historical_events_times
        delta_ts = times - self.first_time
        if len(delta_ts) > max_points:
            events = events[-max_points:]
            delta_ts = delta_ts[-max_points:]
        
        fig.add_trace(go.Scatter(x=delta_ts, y=events, mode="markers", name="Events", marker=dict(color="blue")))
        rolled_with_numpy = np.convolve(events, np.ones(roll) / roll, mode="valid")
        fig.add_trace(go.Scatter(x=delta_ts, y=rolled_with_numpy, mode="lines", name="Rolling Mean", line=dict(color="red")))

        fig.update_layout(
            xaxis_title="Time (Since Start)",
            yaxis_title="Events Over Time",
            title="Events Over Time",
            yaxis=dict(range=[0, None])
        )
        return fig

    def plot_tof_histogram(self):
        if self.historical_data.empty:
            return go.Figure()
        fig = px.bar(x=self.tof_histogram_bins[1:]*1e6, y=self.histogram_counts / self.total_events, labels={"x": "Time of Flight (s)", "y": "Counts"})
        mean = self.tof_mean * 1e6
        variance = self.tof_var * 1e12
        sigma = np.sqrt(variance)
        x = np.linspace(self.tof_hist_min*1e6, self.tof_hist_max*1e6, 1000)
        y = norm.pdf(x, mean, sigma) 
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Gaussian Fit", line=dict(color="red")))
        fig.update_layout(
            xaxis_title="Time of Flight (micro s)",
            yaxis_title="Probability Density",
            title="Time of Flight Histogram",
        )
        return fig

    def plot_wavenumbers(self, new_data, selected_channels = [1, 2, 3, 4]):
        fig = go.Figure()
        if self.historical_data.empty:
            return fig
        colors = ["blue", "red", "green", "purple"]
        delta_ts = self.historical_data["timestamp"] - self.first_time
        for i, channel in enumerate(selected_channels):
            if f"wn_{channel}" in self.historical_data.columns and self.historical_data[f"wn_{channel}"].mean() > 0:
                fig.add_trace(go.Scatter(x=delta_ts, y=self.historical_data[f"wn_{channel}"], mode="lines", name=f"WN_{channel}", line=dict(color=colors[i])))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Wavenumber",
            title="Wavenumbers",
        )
        return fig

    def estimate_rates(self, unseen_new_data: pd.DataFrame) -> pd.Series:
        if self.unseen_new_data.empty:
            return pd.Series()
        channel_ids = unseen_new_data['channel'].unique()
        rates = {}
        delta_t = (self.historical_data['timestamp'].max() - self.historical_data['timestamp'].min())
        for channel_id in channel_ids:
            filtered_new_data = self.historical_data.query(f"channel == {channel_id}")
            if channel_id !=-1:
                event_numbers = len(filtered_new_data)
                rate = event_numbers / delta_t
            else:
                ts = self.historical_data.query("channel == -1").timestamp.diff().mean()
                rate = 1 / ts
            rates[channel_id] = rate
        series = pd.Series(rates).sort_values(ascending=False)
        self.last_rates = series
        return series
    
    def plot_channel_distribution(self):
        if self.historical_data.empty:
            return []
        if self.unseen_new_data.empty or self.unseen_new_data['channel'].nunique() == 1:
            rates = self.last_rates
        else:
            rates = self.estimate_rates(self.unseen_new_data)
        channels = rates.index[rates > 0].tolist()
        values = rates[rates > 0].tolist()

        gauges = []
        for channel, value in zip(channels, values):
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": "Trigger" if channel == -1 else f"Channel {int(channel)}", "font": {"size": 14}},
                gauge={
                    'axis': {'range': [None, max(values)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, value], 'color': 'cyan'},
                        {'range': [value, max(values)], 'color': 'lightgray'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                },
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'suffix': " Hz"}
            ))

            gauges.append(dbc.Col(dcc.Graph(figure=gauge_fig), width=12 // len(channels)))

        return gauges

    def plot_trigger_frequency(self, new_data: pd.DataFrame):
        if self.new_data.empty:
            return go.Figure()

        trigger_events = self.data[self.data['channel'] == -1]
        total_time = (self.data['time'].max() - self.data['time'].min()).total_seconds()
        trigger_frequency = len(trigger_events) / total_time if total_time > 0 else 0

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number",
            value=trigger_frequency,
            title={"text": "Trigger Frequency (Hz)"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        fig.update_layout(title="Only Trigger Events Detected")
        return fig

def query_influxdb(minus_time_str, measurement_name):
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: {minus_time_str})
    |> filter(fn: (r) => r._measurement == "tagger_data")
    |> filter(fn: (r) => r.type == "{measurement_name}")
    |> tail(n: {NBATCH})
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "bunch", "n_events", "channel", "time_offset", "timestamp", "wn_1", "wn_2", "wn_3", "wn_4"])
    '''
    try:
        result = client.query_api().query(query=query, org=INFLUXDB_ORG)
        records = []
        for table in result:
            for record in table.records:
                records.append(record.values)
        df = pd.DataFrame(records).tail(5000).dropna()
        df = df.rename(columns={'_time': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        column_order = ['time', 'bunch', 'n_events', 'channel', 'time_offset', 'timestamp', 'wn_1', 'wn_2', 'wn_3', 'wn_4']
        return df[column_order]
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=['time', 'bunch', 'n_events', 'channel', 'time_offset', "timestamp", 'wn_1', 'wn_2', 'wn_3', 'wn_4'])

def string_to_unix_timestamp(date_string):
    dt = datetime.strptime(date_string, "%Y_%m_%d_%H_%M_%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Scanning Monitor", style={'textAlign': 'center', 'marginBottom': '20px'}),
            dbc.Button("Settings", id="open-offcanvas", n_clicks=0, color="primary", style={'marginRight': '10px'}),
            dbc.Button("Clear Data", id="clear-data", n_clicks=0, color="danger"),
            dbc.Offcanvas(
                [
                    dbc.Row([
                        dbc.Col(html.Div("Refresh Rate (seconds):")),
                        dbc.Col(dcc.Slider(id='refresh-rate', min=0.5, max=10.0, step=0.5, value=0.5)),
                    ]),
                    dbc.Button("Clean Data", id="clean-data", color="danger")
                ],
                id="offcanvas", 
                is_open=False,
                title="Settings"
            )
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='events-over-time', style={'height': '300px'})
        ], width=6),
        dbc.Col([
            dcc.Graph(id='tof-histogram', style={'height': '300px'})
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='wavenumbers', style={'height': '300px'})
        ], width=6),
        dbc.Col([
            dbc.Row(id='channel-gauges', style={'height': '300px'})
        ], width=6)
    ]),
    dcc.Interval(id='interval-component', interval=0.5*1000, n_intervals=0)
], fluid=True)

viz_tool = PlotGenerator()
first_time = 0

@app.callback(
    Output("offcanvas", "is_open"),
    [Input("open-offcanvas", "n_clicks")],
    [State("offcanvas", "is_open")]
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output('interval-component', 'interval'),
    Input('refresh-rate', 'value')
)
def update_refresh_rate(refresh_rate):
    return refresh_rate * 1000

@app.callback(
    [Output('events-over-time', 'figure'),
     Output('tof-histogram', 'figure'),
     Output('wavenumbers', 'figure'),
     Output('channel-gauges', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('events-over-time', 'relayoutData')]
)
def update_plots(n_intervals, relayout_data):
    global viz_tool
    file_location = load_path()["saving_file"]
    measurement_name = file_location.split("scan_")[-1].split(".")[0]
    minus_time_str = string_to_unix_timestamp(measurement_name)
    new_data = query_influxdb(minus_time_str, measurement_name)
    if new_data.empty:
        print("No new data received")
        return go.Figure(), go.Figure(), go.Figure(), []
    viz_tool.update_content(new_data)
    fig_events_over_time = viz_tool.plot_events_over_time()
    fig_tof_histogram = viz_tool.plot_tof_histogram()
    fig_wavenumbers = viz_tool.plot_wavenumbers(new_data)
    gauges = viz_tool.plot_channel_distribution()
    return fig_events_over_time, fig_tof_histogram, fig_wavenumbers, gauges

if __name__ == "__main__":
    app.run_server(debug=True)