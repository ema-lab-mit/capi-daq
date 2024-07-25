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
NBATCH = 1_000
TOTAL_MAX_POINTS = int(50_000)
MAX_POINTS_FOR_PLOT = 500

default_settings = {
    "tof_hist_nbins": 100,
    "tof_hist_min": 1e-6,
    "tof_hist_max": 20e-6,
    "rolling_window": 100,
}

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

global_tof_min = default_settings['tof_hist_min']
global_tof_max = default_settings['tof_hist_max']

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

        self.prev_tof_hist_min = self.tof_hist_min
        self.prev_tof_hist_max = self.tof_hist_max
        self.prev_tof_hist_nbins = self.tof_hist_nbins

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
        events_data = events_data[(events_data['time_offset'] >= global_tof_min) & (events_data['time_offset'] <= global_tof_max)]
        self.total_events += len(events_data)
        events_offset = events_data["time_offset"].values
        if len(events_data) > 0:
            new_hist_counts, _ = np.histogram(events_offset, bins=self.tof_histogram_bins)
            self.histogram_counts = self.histogram_counts + new_hist_counts
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
        self.historical_data = pd.concat([self.historical_data, unseen_new_data], ignore_index=True)

    def plot_events_over_time(self, max_points=MAX_POINTS_FOR_PLOT, roll=10):
        fig = go.Figure()
        if self.historical_data.empty:
            return fig
        events = self.historical_event_numbers
        times = self.historical_events_times
        delta_ts = times - self.first_time
        if len(delta_ts) > max_points:
            events = events[-max_points:]
            delta_ts = delta_ts[-max_points:]
        
        try:
            rolled_with_numpy = np.convolve(events, np.ones(roll)/roll, mode="valid")
        except Exception as e:
            print(f"Error calculating rolling average: {e}")
            rolled_with_numpy = np.zeros(delta_ts[roll-1:])
        fig.add_trace(go.Scatter(x=delta_ts[roll-1:], y=rolled_with_numpy, mode="lines", name=f"Integrated {roll} bounches", line=dict(color="red")))
        # Generate the Bollinger Bands
        upper_band = rolled_with_numpy + 2 * np.std(rolled_with_numpy)
        lower_band = rolled_with_numpy - 2 * np.std(rolled_with_numpy)
        fig.add_trace(go.Scatter(x=delta_ts[roll-1:], y=upper_band, mode="lines", name="Upper Band", line=dict(color="green", dash="dash")))
        fig.add_trace(go.Scatter(x=delta_ts[roll-1:], y=lower_band, mode="lines", name="Lower Band", line=dict(color="green", dash="dash")))

        # Fill between the Bollinger Bands
        fig.add_trace(go.Scatter(
            x=np.concatenate([delta_ts[roll-1:], delta_ts[roll-1:][::-1]]),
            y=np.concatenate([upper_band, lower_band[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Bollinger Bands'
        ))
        
        fig.update_layout(
            xaxis_title="Monitoring Time (s)",
            yaxis_title="Total Counts (s)",
            yaxis=dict(range=[0, None]),
            legend=dict(title="Displaying"),
            template="plotly_white",
            uirevision='events_over_time'  # Preserve UI state
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
        # Rescale y to match the histogram
        y = y * np.max(self.histogram_counts) / (np.max(y) * total_plotted)
        
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Gaussian Fit", line=dict(color="red")))
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
        fig.add_annotation(
            x=mean,
            y=np.max(y)+0.05,
            text=f"Fit: ToF={mean:.2f} ± {sigma:.2f} µs",
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.update_layout(
            xaxis_title="Time of Flight (µs)",
            yaxis_title="Probability Density",
            uirevision='tof_histogram'  # Preserve UI state
        )
        return fig

    def plot_wavenumbers(self, new_data, selected_channels = [1, 2, 3, 4], max_points=10_000):
        fig = go.Figure()
        if self.historical_data.empty:
            return fig
        colors = ["blue", "red", "green", "purple"]
        delta_ts = self.historical_data["timestamp"] - self.first_time
        for i, channel in enumerate(selected_channels):
            if f"wn_{channel}" in self.historical_data.columns and self.historical_data[f"wn_{channel}"].mean() > 0:
                last_data = self.historical_data[f"wn_{channel}"].iloc[-1]
                decimation_factor = (len(delta_ts) // max_points) if len(delta_ts) > max_points else 1
                decimated_ts = delta_ts[::decimation_factor]
                decimated_data = self.historical_data[f"wn_{channel}"][::decimation_factor]
                fig.add_trace(go.Scatter(x=decimated_ts, y=decimated_data, mode="lines", name=f"wavenumber_{channel} = {round(last_data, 12)}", line=dict(color=colors[i])))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Wavenumber",
            uirevision='wavenumbers'  # Preserve UI state
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
        rates[-1] = rates.get(-1, 0) + np.sum([rates[channel_id] for channel_id in channel_ids if channel_id != -1])
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
        df = pd.DataFrame(records).tail(NBATCH).dropna()
        df = df.rename(columns={'_time': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        column_order = ['time', 'bunch', 'n_events', 'channel', 'time_offset', 'timestamp', 'wn_1', 'wn_2', 'wn_3', 'wn_4']
        return df[column_order]
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=['time', 'bunch', 'n_events', 'channel', 'time_offset', "timestamp", 'wn_1', 'wn_2', 'wn_3', 'wn_4'])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
            dbc.Row([
                dbc.Col(width=4),  # Left padding
                dbc.Col(dbc.Button("+", id="wavenumbers-settings-button", n_clicks=0, className="d-block mx-auto"), width=4),  # Centered button
                dbc.Col(width=4)  # Right padding
            ])
        ], width=6),
        dbc.Col([
            dbc.Row(id='channel-gauges', style={'height': '100px'})
        ], width=6)
    ]),
    dcc.Interval(id='interval-component', interval=0.3*1000, n_intervals=0),
    dbc.Offcanvas(
        [
            dbc.Row([
                dbc.Col(html.Div("Refresh Rate (seconds): ")),
                dbc.Col(dcc.Slider(id='refresh-rate', min=0.2, max=10.0, step=0.5, value=0.5, tooltip={"placement": "bottom", "always_visible": True}, marks={i: str(i) for i in np.arange(0.5, 11, 0.5)})),
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
                dbc.Label(r"Integration Window = "),
                dcc.Input(id='events-roll-input', type='number', value=10),
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
                dbc.Label("Min "),
                dcc.Input(id='tof-hist-min-input', type='number', value=default_settings['tof_hist_min'], step=1e-6),
                dbc.Label("ToF Histogram Max\n"),
                dcc.Input(id='tof-hist-max-input', type='number', value=default_settings['tof_hist_max'], step=1e-6),
                dbc.Label("Number of Bins"),
                dcc.Slider(id='tof-bins-slider', min=1, max=100, step=5, value=default_settings['tof_hist_nbins'], marks={i: str(i) for i in range(4, 101, 5)}),
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
    Output("wavenumbers-settings-modal", "is_open"),
    [Input("wavenumbers-settings-button", "n_clicks"), Input("close-wavenumbers-modal", "n_clicks")],
    [State("wavenumbers-settings-modal", "is_open")]
)
def toggle_wavenumbers_settings(n1, n2, is_open):
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
    return int(refresh_rate * 1000)

@app.callback(
    Output('tof-hist-min-input', 'value'),
    Output('tof-hist-max-input', 'value'),
    Output('tof-bins-slider', 'value'),
    Input('update-tof-histogram', 'n_clicks'),
    State('tof-hist-min-input', 'value'),
    State('tof-hist-max-input', 'value'),
    State('tof-bins-slider', 'value')
)
def update_tof_histogram_settings(n_clicks, tof_hist_min, tof_hist_max, tof_hist_nbins):
    if n_clicks:
        viz_tool.update_histogram_bins(tof_hist_min, tof_hist_max, tof_hist_nbins)
    return tof_hist_min, tof_hist_max, tof_hist_nbins

@app.callback(
    [Output('events-over-time', 'figure'),
     Output('tof-histogram', 'figure'),
     Output('wavenumbers', 'figure'),
     Output('channel-gauges', 'children'),
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
def update_plots(n_intervals, clear_clicks, events_roll, update_histogram_clicks, wavenumbers_channels, events_relayout_data, tof_relayout_data, wavenumbers_relayout_data):
    try:
        global viz_tool, global_tof_min, global_tof_max
        ctx = dash.callback_context

        if ctx.triggered and 'clear-data' in ctx.triggered[0]['prop_id'] or viz_tool.total_events > TOTAL_MAX_POINTS:
            viz_tool = PlotGenerator()
            return go.Figure(), go.Figure(), go.Figure(), [], [dbc.Col("No data available.", width=12)]

        if 'tof-histogram.relayoutData' in ctx.triggered[0]['prop_id'] and tof_relayout_data:
            if 'xaxis.range[0]' in tof_relayout_data and 'xaxis.range[1]' in tof_relayout_data:
                global_tof_min = tof_relayout_data['xaxis.range[0]'] * 1e-6
                global_tof_max = tof_relayout_data['xaxis.range[1]'] * 1e-6

        file_location = load_path()["saving_file"]
        measurement_name = file_location.split("monitor_")[-1].split(".")[0]
        minus_time_str = datetime.strptime(measurement_name, "%Y_%m_%d_%H_%M_%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        new_data = query_influxdb(minus_time_str, measurement_name)
        viz_tool.update_content(new_data)
        try:
            fig_events_over_time = viz_tool.plot_events_over_time(roll=events_roll)
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
        gauges = viz_tool.plot_channel_distribution()
        fig_events_over_time.update_layout(uirevision='events_over_time')
        fig_tof_histogram.update_layout(uirevision='tof_histogram')
        fig_wavenumbers.update_layout(uirevision='wavenumbers')
        summary_text = [
            dbc.Col(f"Number of Total Bunches: {len(viz_tool.historical_data['bunch'].unique())}", width=3),
            dbc.Col(f"Total Events: {viz_tool.total_events}", width=3),
            dbc.Col(f"Running Time: {round(viz_tool.historical_data['timestamp'].max() - viz_tool.historical_data['timestamp'].min(), 3)} s", width=3),
            dbc.Col(f"Time since last event: {round(time.time() - viz_tool.historical_data.query('channel!=-1')['timestamp'].max(), 3)} s", width=3),
        ]

        return fig_events_over_time, fig_tof_histogram, fig_wavenumbers, gauges, summary_text
    except Exception as e:
        print(f"Error updating plots: {e}")
        return go.Figure(), go.Figure(), go.Figure(), [], [dbc.Col("No data available.", width=12)]

if __name__ == "__main__":
    app.run_server(debug=True)
