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

# Default settings
NBATCH = 1_000
TOTAL_MAX_POINTS = int(50_000)
MAX_POINTS_FOR_PLOT = 100
default_settings = {
    "tof_hist_nbins": 100,
    "tof_hist_min": 1e-6,
    "tof_hist_max": 20e-6,
    "plot_rolling_window": 10,
    "integration_window": 10,
}
REFRESH_RATE = 0.5  # in seconds

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

global_tof_min = default_settings['tof_hist_min']
global_tof_max = default_settings['tof_hist_max']

class PlotGenerator:
    def __init__(self, settings_dict: dict = default_settings):
        self.settings_dict = settings_dict
        print("YES")
        self.tof_hist_nbins = settings_dict.get("tof_hist_nbins", 100)
        self.tof_hist_min = settings_dict.get("tof_hist_min", 0)
        self.tof_hist_max = settings_dict.get("tof_hist_max", 20e-6)
        self.plot_rolling_window = settings_dict.get("plot_rolling_window", 100)
        self.integration_window = settings_dict.get("integration_window", 10)
        self._historic_timeseries_columns = ["bunch", "timestamp", "n_events", "channel", "wn_1", "wn_2", "wn_3", "wn_4", "voltage"]
        self.last_loaded_time = time.time()
        self.historical_data = pd.DataFrame(columns=self._historic_timeseries_columns)
        self.historical_event_numbers = np.array([])
        self.historical_rates_ns = np.array([])
        self.historical_rates_ts = np.array([])
        self.first_time = time.time()
        self.number_records = 0
        
        self.total_events = 0
        self.tof_mean = 0
        self.tof_var = 0
        self.last_rates = pd.Series()
        
        self.tof_histogram_bins = np.linspace(self.tof_hist_min, self.tof_hist_max, self.tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)

        self.prev_tof_hist_min = self.tof_hist_min
        self.prev_tof_hist_max = self.tof_hist_max
        self.prev_tof_hist_nbins = self.tof_hist_nbins
        self.trigger_rate = 1

    def update_histogram_bins(self, tof_hist_min, tof_hist_max, tof_hist_nbins):
        self.tof_hist_min = tof_hist_min
        self.tof_hist_max = tof_hist_max
        self.tof_hist_nbins = tof_hist_nbins
        self.tof_histogram_bins = np.linspace(tof_hist_min, tof_hist_max, tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)
        
    def _update_tof_statistics(self, unseen_new_data):
        if len(unseen_new_data) == 0:
            return
        events_data = unseen_new_data.query("channel != -1")
        events_data = events_data[((events_data['time_offset'] >= global_tof_min) & (events_data['time_offset'] <= global_tof_max)) | (events_data['channel'] == -1)]
        self.total_events += len(events_data)
        events_offset = events_data["time_offset"].values
        if len(events_data) > 0:
            new_hist_counts, _ = np.histogram(events_offset, bins=self.tof_histogram_bins)
            self.histogram_counts = self.histogram_counts + new_hist_counts
            self.tof_mean = np.average(self.tof_histogram_bins[:-1], weights=self.histogram_counts)
            self.tof_var = np.average((self.tof_histogram_bins[:-1] - self.tof_mean)**2, weights=self.histogram_counts)
        
    def _update_historical_data(self, unseen_new_data):
        events_bunch_data = unseen_new_data.query("channel!=-1").drop_duplicates("bunch")
        if events_bunch_data.empty:
            proc_events_time = 0
            bunch_times = time.time()
        else:
            proc_events_time = events_bunch_data.n_events # events/bunch
            bunch_times = events_bunch_data.timestamp
        self.historical_event_numbers = np.append(self.historical_event_numbers, unseen_new_data.query("channel !=-1")["n_events"].values)
        self.historical_rates_ns = np.append(self.historical_rates_ns, proc_events_time) # Average number of events per second
        self.historical_rates_ts = np.append(self.historical_rates_ts, bunch_times)
        
    def update_content(self, new_data):
        unseen_new_data = new_data[new_data['time'].apply(lambda x: x.timestamp()) > self.last_loaded_time]
        # Filter within the time of flight range
        unseen_new_data = unseen_new_data[((unseen_new_data['time_offset'] >= global_tof_min) & (unseen_new_data['time_offset'] <= global_tof_max)) | (unseen_new_data['channel'] == -1)]

        self.last_loaded_time = new_data['timestamp'].max()
        self.unseen_new_data = unseen_new_data
        self.get_trigger_rate()
        self.integration_time = 1 / (self.integration_window * self.trigger_rate)
        self._update_historical_data(unseen_new_data)
        self._update_tof_statistics(unseen_new_data)
        self.historical_data = pd.concat([self.historical_data, unseen_new_data], ignore_index=True)

    def get_trigger_rate(self):
        if self.unseen_new_data.empty:
            return 1
        number_bunches = self.unseen_new_data['bunch'].max() - self.unseen_new_data['bunch'].min() + 1
        time_diff = self.unseen_new_data['timestamp'].max() - self.unseen_new_data['timestamp'].min()
        self.trigger_rate = number_bunches / time_diff
        return self.trigger_rate
    
    def estimate_rates(self, unseen_new_data: pd.DataFrame) -> pd.Series:
        print("Estimate rates")
        if self.unseen_new_data.empty:
            return pd.Series()
        channel_ids = unseen_new_data['channel'].unique()
        rates = {}
        delta_t = (unseen_new_data['time'].max().timestamp() - unseen_new_data['time'].min().timestamp())
        for channel_id in channel_ids:
            filtered_new_data = unseen_new_data.drop_duplicates("bunch").query(f"channel == {channel_id}")
            if channel_id !=-1:
                event_numbers = len(filtered_new_data)
                rate = event_numbers / delta_t
            else:
                rate = self.trigger_rate
            rates[channel_id] = rate
        series = pd.Series(rates).sort_values(ascending=False)
        self.last_rates = series
        return series

    def plot_events_over_time(self, max_points=MAX_POINTS_FOR_PLOT):
        fig = go.Figure()
        delta_ts = self.historical_rates_ts - self.first_time
        events = self.historical_rates_ns
        if len(delta_ts) > max_points:
            events = events[-max_points:]
            delta_ts = delta_ts[-max_points:]
        fig.add_trace(go.Scatter(x=delta_ts, y=events, mode="markers", name="Number of events", line=dict(color="rgba(135, 206, 250, 0.5)", dash="dash")))
        rolled_events = np.convolve(events, np.ones(self.plot_rolling_window), mode="valid") / self.plot_rolling_window
        rolled_ts = np.linspace(delta_ts[0], delta_ts[-1], len(rolled_events))
        fig.add_trace(go.Scatter(x=rolled_ts, y=rolled_events, mode="lines", name=f"Averaged {self.plot_rolling_window} Bunches", line=dict(color="blue")))

        fig.update_layout(
            xaxis_title="Monitoring Time (s)",
            yaxis_title="Total Counts/Bunch",
            yaxis=dict(range=[0, None]),
            legend=dict(title="Displaying"),
            template="plotly_white",
            uirevision='events_over_time',  # Preserve UI state
        )
        # Put the leggend on the top left corner (inside the plot)
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="Black",
                borderwidth=1
            )
        )
        return fig


    def plot_tof_histogram(self):
        if self.historical_data.empty:
            return go.Figure()
        total_plotted = np.sum(self.histogram_counts)
        fig = px.bar(x=self.tof_histogram_bins[1:]*1e6, y=self.histogram_counts / total_plotted, labels={"x": "Time of Flight (s)", "y": "Counts"})
        mean = self.tof_mean * 1e6
        variance = self.tof_var * 1e12
        sigma = np.sqrt(variance)
        x = np.linspace(self.tof_hist_min*1e6, self.tof_hist_max*1e6, 1000)
        y = norm.pdf(x, mean, sigma) 
        y = y * np.max(self.histogram_counts) / (np.max(y) * total_plotted)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Fit: ToF={mean:.2f} ± {sigma:.2f} µs", line=dict(color="red"), showlegend=True))
        fig.add_shape(
            dict(
                type="line",
                x0=mean,
                y0=0,
                x1=mean,
                y1=np.max(y),
                line=dict(color="black", width=2)
            )
        )
        fig.update_layout(
            xaxis_title="Time of Flight (µs)",
            yaxis_title="Probability Density",
            uirevision='tof_histogram',  # Preserve UI state
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="Black",
                borderwidth=1
            )
        )
        return fig

    def plot_wavenumbers(self, new_data, selected_channels=[1, 2, 3, 4], max_points=200):
        fig = go.Figure()
        if self.historical_data.empty:
            return fig
        colors = ["blue", "red", "green", "purple"]
        delta_ts = self.historical_data["time"].apply(lambda x: x.timestamp()).values - self.first_time
        for i, channel in enumerate(selected_channels):
            if f"wn_{channel}" in self.historical_data.columns and self.historical_data[f"wn_{channel}"].mean() >= 0:
                if channel == 1:
                    print( self.historical_data[f"wn_{channel}"].iloc[-1])           
                last_data = self.historical_data[f"wn_{channel}"].iloc[-1]
                decimation_factor = (len(delta_ts) // max_points) if len(delta_ts) > max_points else 1
                decimated_ts = delta_ts[::decimation_factor]
                decimated_data = self.historical_data[f"wn_{channel}"][::decimation_factor]
                fig.add_trace(go.Scatter(x=decimated_ts, y=decimated_data, mode="lines", name=f"wavenumber_{channel} = {round(last_data, 12)}", line=dict(color=colors[i])))

        fig.update_layout(
            xaxis_title="Monitoring Time (s)",
            yaxis_title="Wavenumber",
            uirevision='wavenumbers',  # Preserve UI state
            template="plotly_white",
            legend=dict(
                x=0.01,  # Position legend inside the plot
                y=0.99,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=1
            )
        )
        return fig

    def plot_voltage(self, max_points=500):
        fig = go.Figure()
        if self.historical_data.empty:
            return fig
        delta_ts = self.historical_data["time"].apply(lambda x: x.timestamp()) - self.first_time
        if "voltage" in self.historical_data.columns and self.historical_data["voltage"].mean() >= 0:
            last_data = self.historical_data["voltage"].iloc[-1]
            decimated_data = self.historical_data["voltage"].tail(max_points)
            decimated_ts = delta_ts.tail(max_points)
            fig.add_trace(go.Scatter(x=decimated_ts, y=decimated_data, mode="lines", name=f"Voltage = {round(last_data, 12)}", line=dict(color="orange")))

        fig.update_layout(
            xaxis_title="Monitoring Time (s)",
            yaxis_title="Voltage",
            template="plotly_white",
            uirevision='voltage'  # Preserve UI state
        )
        return fig

    def plot_channel_distribution(self):
        if self.historical_data.empty:
            return go.Figure()
        if self.unseen_new_data.empty or self.unseen_new_data['channel'].nunique() == 1:
            rates = self.last_rates
        else:
            rates = self.estimate_rates(self.unseen_new_data)
        channels = rates.index[rates > 0].tolist()
        values = rates[rates > 0].tolist()
        colors = ["blue", "red", "green", "purple"]
        fig = go.Figure(go.Bar(
            x=values,
            y=[f"Channel {int(ch)}" if ch != -1 else "Trigger" for ch in channels],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{value:.2f}" for value in values],
            textposition='auto'
        ))

        fig.update_layout(
            xaxis_title="Rate (Hz)",
            yaxis_title="Channel",
            template="plotly_white",
            uirevision='channel_distribution'
        )
        
        return fig


def query_influxdb(minus_time_str, measurement_name):
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: {minus_time_str})
    |> filter(fn: (r) => r._measurement == "tagger_data")
    |> filter(fn: (r) => r.type == "{measurement_name}")
    |> tail(n: {NBATCH})
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "bunch", "n_events", "channel", "time_offset", "timestamp", "wn_1", "wn_2", "wn_3", "wn_4", "voltage"])
    '''
    try:
        result = client.query_api().query(query=query, org=INFLUXDB_ORG)
        records = []
        for table in result:
            for record in table.records:
                records.append(record.values)
        df = pd.DataFrame(records).dropna()
        df = df.rename(columns={'_time': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        column_order = ['time', 'bunch', 'n_events', 'channel', 'time_offset', 'timestamp', 'wn_1', 'wn_2', 'wn_3', 'wn_4', 'voltage']
        cols_in_df = [c for c in column_order if c in df.columns]
        for col in column_order:
            if col not in cols_in_df:   
                df[col] = [None] * len(df)
        return df[column_order]
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=['time', 'bunch', 'n_events', 'channel', 'time_offset', "timestamp", 'wn_1', 'wn_2', 'wn_3', 'wn_4', 'voltage'])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("Settings", id="open-offcanvas", n_clicks=0)),
            dbc.NavItem(dbc.NavLink("Clear Data", id="clear-data", n_clicks=0, className="ml-auto")),
        ],
        brand="Scanning Monitor - CAPI DAQ - EMA Lab",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    # html.H4("Summary Statistics", className="card-title"),
                    dbc.Row(id="summary-statistics", className="card-text")
                ])
            )
        ], width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='events-over-time', style={'height': '400px'}),
            dbc.Row([
                dbc.Col(width=4),  # Left padding
                dbc.Col(dbc.Button("+", id="events-settings-button", n_clicks=0, className="d-block mx-auto"), width=4),  # Centered button
                dbc.Col(width=4)  # Right padding
            ])
        ], width=6),
        dbc.Col([
            dcc.Graph(id='tof-histogram', style={'height': '400px'}),
            dbc.Row([
                dbc.Col(width=4),  # Left padding
                dbc.Col(dbc.Button("+", id="tof-settings-button", n_clicks=0, className="d-block mx-auto"), width=4),  # Centered button
                dbc.Col(width=4)  # Right padding
            ])
        ], width=6)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='wavenumbers', style={'height': '300px'}),
        ], width=4),
        dbc.Col([
            dcc.Graph(id='voltage', style={'height': '300px'}),
        ], width=4),
        dbc.Col([
            dcc.Graph(id='channel-distribution', style={'height': '300px'})
        ], width=4),
    ]),
    dcc.Interval(id='interval-component', interval=REFRESH_RATE*1000, n_intervals=0),
    dbc.Offcanvas(
        [
            dbc.Row([
                dbc.Col(html.Div("Refresh Rate (seconds): ")),
                dbc.Col(dcc.Slider(id='refresh-rate', min=0.2, max=10.0, step=0.1, value=REFRESH_RATE, tooltip={"placement": "bottom", "always_visible": True}, marks={i: str(i) for i in np.arange(0.2, 1, 0.01)})),
            ], style={'padding': '20px'}),
            dbc.Row([
                dbc.Col(html.Div("Batch Size (NBATCH): ")),
                dbc.Col(dcc.Input(id='nbatch-input', type='number', value=NBATCH, step=100)),
            ], style={'padding': '20px'}),
            dbc.Row([
                dbc.Col(html.Div("Total Max Points: ")),
                dbc.Col(dcc.Input(id='total-max-points-input', type='number', value=TOTAL_MAX_POINTS, step=1000)),
            ], style={'padding': '20px'}),
            dbc.Row([
                dbc.Col(html.Div("Max Points for Plot: ")),
                dbc.Col(dcc.Input(id='max-points-for-plot-input', type='number', value=MAX_POINTS_FOR_PLOT, step=100)),
            ], style={'padding': '20px'}),
            dbc.Row([
                dbc.Col(html.Div("Plot Rolling Window: ")),
                dbc.Col(dcc.Input(id='plot-rolling-window-input', type='number', value=default_settings['plot_rolling_window'], step=10)),
            ], style={'padding': '20px'}),
            dbc.Row([
                dbc.Col(html.Div("Integration Window: ")),
                dbc.Col(dcc.Input(id='integration-window-input', type='number', value=default_settings['integration_window'], step=1)),
            ], style={'padding': '20px'}),
        ],
        id="offcanvas", 
        is_open=False,
        title="Settings"
    ),
    dbc.Modal(
        [
            dbc.ModalHeader("Events Over Time Settings"),
            dbc.ModalBody([
                dbc.Label("Plot Rolling Window:"),
                dcc.Input(id='events-roll-input', type='number', value=default_settings['plot_rolling_window']),
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-events-modal", className="ml-auto")
            ])
        ],
        id="events-settings-modal",
        is_open=False,
    ),
    dbc.Modal(
        [
            dbc.ModalHeader("ToF Histogram Settings"),
            dbc.ModalBody([
                dbc.Label("ToF Histogram Range (µs)"),
                dcc.RangeSlider(
                    id='tof-hist-range-slider',
                    min=1, max=20, step=1,
                    value=[default_settings['tof_hist_min']*1e6, default_settings['tof_hist_max']*1e6],
                    marks={i: str(i) for i in range(1, 21)}
                ),
                dbc.Label("Number of Bins"),
                dcc.Slider(
                    id='tof-bins-slider',
                    min=1, max=100, step=5,
                    value=default_settings['tof_hist_nbins'],
                    marks={i: str(i) for i in range(5, 101, 5)}
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Update Histogram Parameters", id="update-tof-histogram", className="ml-auto"),
                dbc.Button("Close", id="close-tof-modal", className="ml-auto")
            ])
        ],
        id="tof-settings-modal",
        is_open=False,
    ),
    dbc.Modal(
        [
            dbc.ModalHeader("Wavenumbers Settings"),
            dbc.ModalBody([
                dbc.Label("Selected Channels"),
                dcc.Checklist(
                    id='wavenumbers-channels-input',
                    options=[
                        {'label': 'Channel 1', 'value': 1},
                        {'label': 'Channel 2', 'value': 2},
                        {'label': 'Channel 3', 'value': 3},
                        {'label': 'Channel 4', 'value': 4}
                    ],
                    value=[1, 2, 3, 4]
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-wavenumbers-modal", className="ml-auto")
            ])
        ],
        id="wavenumbers-settings-modal",
        is_open=False,
    ),
], fluid=True)

viz_tool = PlotGenerator()
first_time = 0

@app.callback(
    Output("events-settings-modal", "is_open"),
    [Input("events-settings-button", "n_clicks"), Input("close-events-modal", "n_clicks")],
    [State("events-settings-modal", "is_open")]
)
def toggle_events_settings(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("tof-settings-modal", "is_open"),
    [Input("tof-settings-button", "n_clicks"), Input("close-tof-modal", "n_clicks")],
    [State("tof-settings-modal", "is_open")]
)
def toggle_tof_settings(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

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
    global REFRESH_RATE
    REFRESH_RATE = refresh_rate
    return int(refresh_rate * 1000)

@app.callback(
    [Output('nbatch-input', 'value'),
     Output('total-max-points-input', 'value'),
     Output('max-points-for-plot-input', 'value'),
     Output('plot-rolling-window-input', 'value'),
     Output('integration-window-input', 'value')],
    [Input('nbatch-input', 'value'),
     Input('total-max-points-input', 'value'),
     Input('max-points-for-plot-input', 'value'),
     Input('plot-rolling-window-input', 'value'),
     Input('integration-window-input', 'value')]
)
def update_settings(nbatch, total_max_points, max_points_for_plot, plot_rolling_window, integration_window):
    global NBATCH, TOTAL_MAX_POINTS, MAX_POINTS_FOR_PLOT, default_settings
    NBATCH = nbatch
    TOTAL_MAX_POINTS = total_max_points
    MAX_POINTS_FOR_PLOT = max_points_for_plot
    default_settings['plot_rolling_window'] = plot_rolling_window
    default_settings['integration_window'] = integration_window
    viz_tool.plot_rolling_window = plot_rolling_window
    viz_tool.integration_window = integration_window
    return nbatch, total_max_points, max_points_for_plot, plot_rolling_window, integration_window

@app.callback(
    Output('tof-hist-range-slider', 'value'),
    Output('tof-bins-slider', 'value'),
    Input('update-tof-histogram', 'n_clicks'),
    State('tof-hist-range-slider', 'value'),
    State('tof-bins-slider', 'value')
)
def update_tof_histogram_settings(n_clicks, tof_hist_range, tof_hist_nbins):
    global global_tof_min, global_tof_max
    if n_clicks:
        global_tof_min = tof_hist_range[0] * 1e-6
        global_tof_max = tof_hist_range[1] * 1e-6
        viz_tool.update_histogram_bins(global_tof_min, global_tof_max, tof_hist_nbins)
    return tof_hist_range, tof_hist_nbins

@app.callback(
    [Output('events-over-time', 'figure'),
     Output('tof-histogram', 'figure'),
     Output('wavenumbers', 'figure'),
     Output('voltage', 'figure'),
     Output('channel-distribution', 'figure'),
     Output('summary-statistics', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('clear-data', 'n_clicks'),
     Input('events-roll-input', 'value'),
     Input('update-tof-histogram', 'n_clicks'),
     Input('wavenumbers-channels-input', 'value')],
    [State('events-over-time', 'relayoutData'),
     State('tof-histogram', 'relayoutData'),
     State('wavenumbers', 'relayoutData')]
)
def update_plots(n_intervals, clear_clicks, plot_rolling_window, update_histogram_clicks, wavenumbers_channels, events_relayout_data, tof_relayout_data, wavenumbers_relayout_data):
    try:
        global viz_tool, global_tof_min, global_tof_max
        ctx = dash.callback_context

        if ctx.triggered and 'clear-data' in ctx.triggered[0]['prop_id'] or viz_tool.total_events > TOTAL_MAX_POINTS:
            viz_tool = PlotGenerator()
            return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [dbc.Col("No data available.", width=12)]

        if 'tof-histogram.relayoutData' in ctx.triggered[0]['prop_id'] and tof_relayout_data:
            if 'xaxis.range[0]' in tof_relayout_data and 'xaxis.range[1]' in tof_relayout_data:
                global_tof_min = tof_relayout_data['xaxis.range[0]'] * 1e-6
                global_tof_max = tof_relayout_data['xaxis.range[1]'] * 1e-6

        file_location = load_path()["saving_file"]
        measurement_name = file_location.split("monitor_")[-1].split(".")[0]
        minus_time_str = datetime.strptime(measurement_name, "%Y_%m_%d_%H_%M_%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        new_data = query_influxdb(minus_time_str, measurement_name)
        viz_tool.update_content(new_data)

        if new_data.empty:
            viz_tool.historical_event_numbers = np.append(viz_tool.historical_event_numbers, 0)
        try:
            fig_events_over_time = viz_tool.plot_events_over_time()
        except Exception as e:
            print(f"Error updating events over time: {e}")
            fig_events_over_time = go.Figure()
        try:
            fig_tof_histogram = viz_tool.plot_tof_histogram()
        except Exception as e:
            print(f"Error updating ToF histogram: {e}")
            fig_tof_histogram = go.Figure()            
        try:
            fig_wavenumbers = viz_tool.plot_wavenumbers(new_data, selected_channels=wavenumbers_channels)
        except Exception as e:
            print(f"Error updating wavenumbers: {e}")
            fig_wavenumbers = go.Figure()
        try:
            fig_voltage = viz_tool.plot_voltage()
        except Exception as e:
            print(f"Error updating voltage plot: {e}")
            fig_voltage = go.Figure()
        try:
            fig_channel_distribution = viz_tool.plot_channel_distribution()
        except Exception as e:
            print(f"Error updating channel distribution plot: {e}")
            fig_channel_distribution = go.Figure()

        fig_events_over_time.update_layout(uirevision='events_over_time')
        fig_tof_histogram.update_layout(uirevision='tof_histogram')
        fig_wavenumbers.update_layout(uirevision='wavenumbers')
        fig_voltage.update_layout(uirevision='voltage')
        fig_channel_distribution.update_layout(uirevision='channel_distribution')
        status_color = {"color": "green"} if time.time() - viz_tool.historical_data.query('channel!=-1')['timestamp'].max() < 2 else {"color": "red"}
        status_text = "Status: Online" if time.time() - viz_tool.historical_data.query('channel!=-1')['timestamp'].max() < 2 else "Status: Offline"
        summary_text = [
            # Status indicator: green if the last event was less than 1 second ago, red otherwise
            dbc.Col(status_text, style=status_color, width=2),
            dbc.Col(f"Bunch Count: {len(viz_tool.historical_data['bunch'].unique())}", width=2),
            dbc.Col(f"Total Events: {viz_tool.total_events}", width=2),
            dbc.Col(f"Running Time: {round(viz_tool.historical_data['timestamp'].max() - viz_tool.historical_data['timestamp'].min(), 2)} s", width=3),
            dbc.Col(f"Time since last event: {round(time.time() - viz_tool.historical_data.query('channel!=-1')['timestamp'].max(), 2)} s", width=3),
            dbc.Col(f"λ1: {viz_tool.historical_data['wn_1'].iloc[-1].round(12)}", width=2),
            dbc.Col(f"Voltage: {viz_tool.historical_data['voltage'].iloc[-1]} V", width=2),
            dbc.Col(f"Trigger Rate: {viz_tool.trigger_rate:.2f} Hz", width=2),
        ]

        return fig_events_over_time, fig_tof_histogram, fig_wavenumbers, fig_voltage, fig_channel_distribution, summary_text
    except Exception as e:
        print(f"Error updating plots: {e}")
        loading_icon = go.Scatter(x=[0], y=[0], mode="text", text=["Loading..."], textfont_size=24, textposition="middle center")
        return go.Figure(data=loading_icon), go.Figure(data=loading_icon), go.Figure(data=loading_icon), go.Figure(data=loading_icon), [dbc.Col("No data available.", width=12)]

if __name__ == "__main__":
    app.run_server(debug=True)
