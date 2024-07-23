import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
import sys
import warnings
from influxdb_client import InfluxDBClient
from scipy.stats import norm

warnings.simplefilter("ignore")
first_time = 0


# Set up paths and tokens
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from fast_tagger_gui.src.system_utils import get_secrets, load_path
from fast_tagger_gui.src.physics_utils import compute_tof_from_data

db_token = get_secrets()
os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

def events_numbers(og_data: pd.DataFrame):
    data = og_data.copy()[["bunch", "time", "n_events"]]
    number_of_events = data.drop_duplicates(subset=["bunch"])
    return number_of_events[['bunch', 'n_events']]

class PlotGenerator:
    def __init__(self, data):
        self.data = data

    def plot_events_over_time(self, timewindow=120, max_points=1_000):
        global first_time
        time_delta = pd.Timedelta(seconds=timewindow)
        last_data = self.data.tail(5_000)[self.data['time'] > self.data['time'].max() - time_delta]
        
        # Decimate data if necessary
        if len(last_data) > max_points:
            last_data = last_data.iloc[::len(last_data)//max_points]
        last_data['time_in_seconds'] = (last_data['time'] - first_time).dt.total_seconds()
        last_data['time_in_seconds'] = last_data['time_in_seconds'].astype(int)
        events_over_time = last_data.query("channels != -1").groupby('time_in_seconds').size()
        rolling_mean = events_over_time.rolling(window=5).mean()
        rolling_std = events_over_time.rolling(window=5).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=events_over_time.index, y=events_over_time.values, mode="lines", name="Events per Second", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode="lines", name="Rolling Mean (10s)", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=upper_band.index, y=upper_band, mode="lines", name="Upper Bollinger Band", line=dict(color="green"), fill="tonexty", fillcolor="rgba(0, 255, 0, 0.2)"))
        fig.add_trace(go.Scatter(x=lower_band.index, y=lower_band, mode="lines", name="Lower Bollinger Band", line=dict(color="red"), fill="tonexty", fillcolor="rgba(255, 0, 0, 0.2)"))
        
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Events Over Time",
            title="Events Over Time",
            yaxis=dict(range=[0, None])  # Set the y-axis minimum to 0
        )
        
        return fig

    def plot_tof_histogram(self):
        tofs = compute_tof_from_data(self.data.tail(500))
        mu, std = norm.fit(tofs)
        xmin, xmax = tofs.min(), tofs.max()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=tofs, nbinsx=100, name='TOF Data', marker_color='blue', opacity=0.7))
        fig.add_trace(go.Scatter(x=x, y=p * len(tofs) * (xmax - xmin) / 100, mode='lines', name='Gaussian Fit', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=[mu], y=[0], mode='markers', marker=dict(color='green', size=10, symbol='x'), name=f'Mean: {mu:.2f} μs'))
        fig.add_trace(go.Scatter(x=[mu - std, mu + std], y=[0, 0], mode='markers', marker=dict(color='orange', size=10, symbol='line-ew'), name=f'Standard Deviation: {std:.2f} μs'))
        fig.update_layout(xaxis_title="Time of Flight (μs)", yaxis_title="Events", title="TOF Histogram", showlegend=True)
        return fig

    def plot_events_per_bunch(self):
        notrigger_df = self.data[self.data['channels'] != -1]
        events_per_bunch = events_numbers(notrigger_df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=events_per_bunch["bunch"], y=events_per_bunch["n_events"], mode="lines+markers", name="Events per Bunch", line=dict(color="blue")))
        fig.update_layout(xaxis_title="Bunch", yaxis_title="Events", title="Events per Bunch", yaxis=dict(range=[0, None]))  # Set the y-axis minimum to 0
        return fig

    def plot_channel_distribution(self):
        rates = estimate_rates(self.data)
        channels = rates.index[rates > 0].tolist()
        values = rates[rates > 0].tolist()

        gauges = []
        for channel, value in zip(channels, values):
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": "Trigger" if channel == -1 else f"Channel {channel}", "font": {"size": 14}},
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

    def plot_trigger_frequency(self):
        trigger_events = self.data[self.data['channels'] == -1]
        total_time = (self.data['time'].max() - self.data['time'].max()).total_seconds()
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

def query_influxdb(minus_time_str, measurement_name, aggregate_interval="1s"):
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: {minus_time_str})
    |> filter(fn: (r) => r._measurement == "tagger_data")
    |> filter(fn: (r) => r.type == "{measurement_name}")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "bunch", "n_events", "channels", "timestamp", "wavenumber_1", "wavenumber_2", "wavenumber_3", "wavenumber_4"])
    '''
    try:
        result = client.query_api().query(query=query, org=INFLUXDB_ORG)
        records = []

        for table in result:
            for record in table.records:
                records.append(record.values)
                
        data_wihth_events = [
            [rec["_time"], rec["bunch"], rec["n_events"], rec["channels"], rec["timestamp"], rec["wavenumber_1"], rec["wavenumber_2"], rec["wavenumber_3"], rec["wavenumber_4"]]
            for rec in records if (rec["n_events"] > 0 and rec["channels"] != -1)
        ] 
        print(data_wihth_events)
        MAX_EVENTS = 5000
        df = pd.DataFrame(data_wihth_events)
        if len(df) > MAX_EVENTS:
            df = df.iloc[::len(df)//MAX_EVENTS]
        df.columns = ['_time', 'bunch', 'n_events', 'channels', 'timestamp', 'wavenumber_1', 'wavenumber_2', 'wavenumber_3', 'wavenumber_4']
        df = df.rename(columns={'_time': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=['time', 'bunch', 'n_events', 'channels', 'timestamp', 'wavenumber_1', 'wavenumber_2', 'wavenumber_3', 'wavenumber_4'])

def estimate_rates(og_data: pd.DataFrame):
    initial_time = og_data["time"].min()
    current_time = og_data["time"].max()
    min_num_bunches = og_data["bunch"].min()
    max_num_bunches = og_data["bunch"].max()
    
    time_diff = (current_time - initial_time).total_seconds()
    delta_bunches = max_num_bunches - min_num_bunches
    trigger_count = delta_bunches / time_diff
    channel_counts = og_data["channels"].value_counts() 
    channel_counts = channel_counts.reindex(range(-1, 4), fill_value=0)
    rates = channel_counts / time_diff
    print(rates)
    rates[-1] = trigger_count
    return rates

def string_to_unix_timestamp(date_string):
    dt = datetime.strptime(date_string, "%Y_%m_%d_%H_%M_%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Scanning Monitor", style={'textAlign': 'center', 'marginBottom': '20px'}),
            dbc.Button("Settings", id="open-offcanvas", n_clicks=0, color="primary"),
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
            dcc.Graph(id='events-per-bunch', style={'height': '300px'})
        ], width=6),
        dbc.Col([
            dbc.Row(id='channel-gauges', style={'height': '300px'})
        ], width=6)
    ]),
    dcc.Interval(id='interval-component', interval=0.1*1000, n_intervals=0)
], fluid=True)

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
     Output('events-per-bunch', 'figure'),
     Output('channel-gauges', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('clean-data', 'n_clicks')],
    [State('events-over-time', 'relayoutData')]
)
def update_plots(n_intervals, n_clicks, relayout_data):
    global first_time
    file_location = load_path()["saving_file"]
    measurement_name = file_location.split("scan_")[-1].split(".")[0]
    minus_time_str = string_to_unix_timestamp(measurement_name)
    
    # Determine the time range to fetch based on user zoom/pan actions
    if relayout_data and 'xaxis.range' in relayout_data:
        start_time, end_time = relayout_data['xaxis.range']
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        minus_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        aggregate_interval = f"{int((end_time - start_time).total_seconds() // 100)}s"  # Example aggregation interval
    else:
        aggregate_interval = "1s"  # Default aggregation interval
    
    data = query_influxdb(minus_time_str, measurement_name, aggregate_interval)
    print(len(data))
    if first_time == 0:
        first_time = data['time'].min()
    
    if data.empty:
        return go.Figure(), go.Figure(), go.Figure(), []

    plot_generator = PlotGenerator(data)
    
    last_data_batch = data.tail(100)
    last_signal_events = last_data_batch[last_data_batch['channels'] != -1]
    if last_signal_events.empty:
        fig_trigger_frequency = plot_generator.plot_trigger_frequency()
        return fig_trigger_frequency, go.Figure(), go.Figure(), []

    fig_events_over_time = plot_generator.plot_events_over_time()
    fig_tof_histogram = plot_generator.plot_tof_histogram()
    fig_events_per_bunch = plot_generator.plot_events_per_bunch()
    
    gauges = plot_generator.plot_channel_distribution()
    return fig_events_over_time, fig_tof_histogram, fig_events_per_bunch, gauges

if __name__ == "__main__":
    app.run_server(debug=True)
