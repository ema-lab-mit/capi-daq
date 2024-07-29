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
import plotly.colors as colors
import threading
import time

warnings.simplefilter("ignore")
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from fast_tagger_gui.src.system_utils import get_secrets, load_path
from fast_tagger_gui.src.physics_utils import compute_tof_from_data

# Get database token
db_token = get_secrets()
os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"
NBATCH = 1_000
TOTAL_MAX_POINTS = int(100_000_000)
MAX_POINTS_FOR_PLOT = 500

# Default settings for the plots
default_settings = {
    "tof_hist_nbins": 100,
    "tof_hist_min": 1e-6,
    "tof_hist_max": 20e-6,
    "rolling_window": 100,
}

# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

global_tof_min = default_settings['tof_hist_min']
global_tof_max = default_settings['tof_hist_max']

er_wn_hist_cache = None
tof_hist_cache = None
tof_rw_hist_cache = None
cache_lock = threading.Lock()

class PlotGenerator:
    def __init__(self, settings_dict: dict = default_settings):
        self.settings_dict = settings_dict
        
        self.tof_hist_nbins = settings_dict.get("tof_hist_nbins", 100)
        self.tof_hist_min = settings_dict.get("tof_hist_min", 0)
        self.tof_hist_max = settings_dict.get("tof_hist_max", 20e-6)
        
        self.rolling_window = settings_dict.get("rolling_window", 100)
        
        self._historic_timeseries_columns = ["bunch", "timestamp", "n_events", "channel", "wn_1", "wn_2", "wn_3", "wn_4", "voltage"]
        self.historical_data = pd.DataFrame(columns=self._historic_timeseries_columns)
        self.unseen_new_data = pd.DataFrame(columns=self._historic_timeseries_columns)
        self.all_wns_measurements = np.array([])
        self.all_nevents_measurements = np.array([])
        
        # Rolled all wns and nevents
        self.rolled_all_wns = np.array([])
        self.rolled_all_nevents = np.array([])
        self.rolled_all_errors = np.array([])
        
        self.first_time = time.time()
        self.last_loaded_time = time.time()
        
        self.total_events = 0
        self.tof_mean = 0
        self.tof_var = 0
        
        self.tof_histogram_bins = np.linspace(self.tof_hist_min, self.tof_hist_max, self.tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)
        
        self.wn_min = 12e3
        self.wn_max = 13e3
            
        self.integration_window = 50
        self.wn_channel_selected = 1
        self.ER_WN_BINS = 50
        
        self.er_wn_hist = np.zeros((self.ER_WN_BINS, self.ER_WN_BINS))
        self.er_edges = np.linspace(self.wn_min, self.wn_max, self.ER_WN_BINS + 1)
        self.wn_edges = np.linspace(self.wn_min, self.wn_max, self.ER_WN_BINS + 1)

    def update_tof_histogram_bins(self, tof_hist_min, tof_hist_max, tof_hist_nbins):
        self.tof_hist_min = tof_hist_min
        self.tof_hist_max = tof_hist_max
        self.tof_hist_nbins = tof_hist_nbins
        self.tof_histogram_bins = np.linspace(tof_hist_min, tof_hist_max, tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)
        
    def _update_tof_statistics(self, unseen_new_data):
        if unseen_new_data.empty:
            return
        events_data = unseen_new_data.query("channel != -1")
        events_data = events_data[(events_data['time_offset'] >= global_tof_min) & (events_data['time_offset'] <= global_tof_max)]
        self.total_events += len(events_data)
        events_offset = events_data["time_offset"].values
        if len(events_data) > 0:
            new_hist_counts, _ = np.histogram(events_offset, bins=self.tof_histogram_bins)
            self.histogram_counts += new_hist_counts
            self.tof_mean = np.average(self.tof_histogram_bins[:-1], weights=self.histogram_counts)
            self.tof_var = np.average((self.tof_histogram_bins[:-1] - self.tof_mean)**2, weights=self.histogram_counts)
            
    def get_trigger_rate(self):
        if self.unseen_new_data.empty:
            return 0
        number_bunches = self.unseen_new_data['bunch'].max() - self.unseen_new_data['bunch'].min() + 1
        time_diff = self.unseen_new_data['timestamp'].max() - self.unseen_new_data['timestamp'].min()
        self.trigger_rate = number_bunches / time_diff
        return self.trigger_rate
    
    def compute_counts_time(self, integration_time=0.5):
        # How many triggers as a function of time
        ntrigs = self.trigger_rate * integration_time # trigs/s
        # How many events per trigger
        number_events_per_trigger = self.proc_events_time / len(self.historical_data)
        # Total counts in the integration time
        total_counts = ntrigs * number_events_per_trigger
        return total_counts # counts/s
        
    
    def _update_wavenumber_statistics(self):
        er_wn_dataframe = self.unseen_new_data.query("channel != -1").drop_duplicates(subset=["bunch"])[[f"wn_{self.wn_channel_selected}", "n_events"]]
        if er_wn_dataframe.empty or len(er_wn_dataframe) == 0:
            return
        self.all_wns_measurements = np.concatenate([self.all_wns_measurements, er_wn_dataframe[f"wn_{self.wn_channel_selected}"].values])
        self.all_nevents_measurements = np.concatenate([self.all_nevents_measurements, er_wn_dataframe["n_events"].values])
        if len(self.all_wns_measurements) > self.rolling_window:
            time_window = self.trigger_rate * self.integration_window
            self.rolled_all_wns = np.convolve(self.all_wns_measurements, np.ones(self.rolling_window)/self.rolling_window, mode="valid")
            self.rolled_all_nevents = np.convolve(self.all_nevents_measurements, np.ones(self.rolling_window) / time_window, mode="valid")
            
            # Compute errors for the convolution
            nevents_errors = np.sqrt(self.all_nevents_measurements)
            self.rolled_all_errors = np.sqrt(np.convolve(nevents_errors**2, np.ones(self.rolling_window) / time_window, mode="valid"))
            
    def _update_historical_data(self, unseen_new_data):
        self.historical_data = pd.concat([self.historical_data, unseen_new_data], ignore_index=True)
        if self.historical_data.shape[0] > TOTAL_MAX_POINTS:
            self.historical_data = self.historical_data.tail(TOTAL_MAX_POINTS)
        
    def update_content(self, new_data):
        unseen_new_data = new_data[new_data['timestamp'] > self.last_loaded_time]
        self.last_loaded_time = new_data['timestamp'].max()
        self.unseen_new_data = unseen_new_data
        if unseen_new_data.empty:
            return
        self.get_trigger_rate()
        self._update_tof_statistics(unseen_new_data)
        self._update_wavenumber_statistics()
        self._update_historical_data(unseen_new_data)

    def plot_events_over_time(self, max_points=MAX_POINTS_FOR_PLOT, roll=10):
        fig = go.Figure()
        if len(self.historical_data.n_events) == 0:
            return fig
        events = self.historical_data.n_events.values
        times = self.historical_data.timestamp.values
        delta_ts = times - self.first_time
        if len(delta_ts) > max_points:
            events = events[-max_points:]
            delta_ts = delta_ts[-max_points:]

        if len(events) < roll:
            rolled_with_numpy = np.zeros(len(events))
        else:
            rolled_with_numpy = np.convolve(events, np.ones(roll)/roll, mode="valid")

        fig.add_trace(go.Scatter(x=delta_ts[roll-1:], y=rolled_with_numpy, mode="lines", name=f"Integrated {roll} B", line=dict(color="red")))
        
        # Generate the Bollinger Bands
        if len(rolled_with_numpy) > 0:
            upper_band = rolled_with_numpy + 2 * np.std(rolled_with_numpy)
            lower_band = rolled_with_numpy - 2 * np.std(rolled_with_numpy)
        else:
            upper_band = np.zeros(len(delta_ts[roll-1:]))
            lower_band = np.zeros(len(delta_ts[roll-1:]))

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
        fig = px.bar(y=self.tof_histogram_bins[1:]*1e6, x=self.histogram_counts / total_plotted, orientation='h',
                     labels={"y": "Time of Flight (s)", "x": "Counts"},
                        title="Time of Flight Histogram", template="plotly_white",
        )                        
                  
        fig.update_xaxes(range=[0, 1])
        mean = self.tof_mean * 1e6
        variance = self.tof_var * 1e12
        sigma = np.sqrt(variance)
        x = np.linspace(self.tof_hist_min*1e6, self.tof_hist_max*1e6, 1000)
        y = norm.pdf(x, mean, sigma) 
        y = y * np.max(self.histogram_counts) / (np.max(y) * total_plotted)
        fig.add_trace(go.Scatter(x=y, y=x, mode="lines", name="Gaussian Fit", line=dict(color="red")))
        fig.add_shape(
            dict(
                type="line",
                x0=0,
                y0=mean,
                x1=np.max(y),
                y1=mean,
                line=dict(color="black", width=2)
            )
        )
        fig.add_annotation(
            x=np.max(y),
            y=mean+10,
            text=f"Fit: ToF={mean:.2f} ± {sigma:.2f} µs",
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(
            xaxis_title="Density",
            yaxis_title="Time of Flight (µs)",
            uirevision='tof_histogram'
        )
        # Update the x axis to be between o and max of the histogram +0.1
        fig.update_xaxes(range=[0, np.max(y) + 0.1])
        return fig

    def plot_rate_vs_wavenumber_2d_histogram(self, num_bins=50):
        """
        Plot the 2D histogram of the event rate vs the wavenumber with reduced bins
        """
        if self.historical_data.empty:
            return go.Figure()

        # Create a DataFrame with wavenumber and event count
        df = pd.DataFrame({
            "wn": self.rolled_all_wns,
            "n_events": self.rolled_all_nevents
        })

        # Calculate bin edges
        wn_min, wn_max = df['wn'].min(), df['wn'].max()
        bin_edges = pd.interval_range(start=wn_min, end=wn_max, periods=num_bins)

        # Bin the data and sum the events in each bin
        df['wn_bin'] = pd.cut(df['wn'], bins=bin_edges)
        binned_df = df.groupby('wn_bin')['n_events'].sum().reset_index()

        # Extract the bin midpoints for plotting
        binned_df['wn_mid'] = binned_df['wn_bin'].apply(lambda x: x.mid)

        # Create the plot
        fig = px.bar(binned_df, x="wn_mid", y="n_events", template="plotly_white",
                    title="Event Rate vs Wavenumber",
                    labels={"wn_mid": "Wavenumber (cm^-1)", "n_events": "Event Rate"})
        fig = px.scatter(binned_df, x="wn_mid", y="n_events", template="plotly_white",
                        title="Event Rate vs Wavenumber",
                        labels={"wn_mid": "Wavenumber (cm^-1)", "n_events": "Event Rate"})
        # Add lines between the points
        fig.update_traces(mode="lines+markers", line=dict(width=2))
        # Simple polynomial fit
        

        fig.update_layout(
            xaxis_title="Wavenumber (cm^-1)",
            yaxis_title="Event Rate",
            uirevision='rate_vs_wavenumber'
        )

        return fig
    
    def plot_3d_tof_rw(self):
        fig = go.Figure()
        if self.historical_data.empty:
            return fig
        df_events = self.historical_data.query("channel != -1")
        fig = px.density_heatmap(df_events, x="wn_1", y="time_offset", nbinsx=self.ER_WN_BINS, nbinsy=self.ER_WN_BINS,
                                    title="Event Rate vs ToF", template="plotly_white", marginal_x="histogram", marginal_y="violin")
        fig.update_layout(
            xaxis_title="Wavenumber (cm^-1)",
            yaxis_title="Time of Flight (s)",
            uirevision='rate_vs_wavenumber'
        )
        return fig

    def _compute_er_wn_hist(self):
        er_wn_dataframe = self.unseen_new_data.query("channel != -1").drop_duplicates(subset=["bunch"])[[f"wn_{self.wn_channel_selected}", "n_events"]]
        if er_wn_dataframe.empty or len(er_wn_dataframe) == 0:
            return self.rolled_all_wns, self.rolled_all_nevents
        
        self.all_wns_measurements = np.concatenate([self.all_wns_measurements, er_wn_dataframe[f"wn_{self.wn_channel_selected}"].values])
        self.all_nevents_measurements = np.concatenate([self.all_nevents_measurements, er_wn_dataframe["n_events"].values])

        if len(self.all_wns_measurements) > self.rolling_window:
            time_window = self.trigger_rate * self.integration_window
            self.rolled_all_wns = np.convolve(self.all_wns_measurements, np.ones(self.rolling_window)/self.rolling_window, mode="valid")
            self.rolled_all_nevents = np.convolve(self.all_nevents_measurements, np.ones(self.rolling_window) / time_window, mode="valid")
            
            # Compute errors for the convolution
            nevents_errors = np.sqrt(self.all_nevents_measurements)
            self.rolled_all_errors = np.sqrt(np.convolve(nevents_errors**2, np.ones(self.rolling_window) / time_window, mode="valid"))
        
        return self.rolled_all_wns, self.rolled_all_nevents, self.rolled_all_errors

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
        return df[column_order]
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=['time', 'bunch', 'n_events', 'channel', 'time_offset', "timestamp", 'wn_1', 'wn_2', 'wn_3', 'wn_4', 'voltage'])

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
                    dbc.Row(id="summary-statistics", className="card-text")
                ])
            )
        ], width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='rate-vs-wavenumber', style={'height': '400px'}),
        ], width=6),
        dbc.Col([
            dcc.Graph(id='events-over-time', style={'height': '400px'}),
        ], width=6)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='3d-bar-rate-vs-wavenumber', style={'height': '400px'}),
        ], width=6),
        dbc.Col([
            dcc.Graph(id='tof-histogram', style={'height': '400px'}),
        ], width=6)
    ], className="mb-4"),
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
                dbc.Label("ToF Histogram Max"),
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
    )
], fluid=True)

viz_tool = PlotGenerator()
first_time = 0

def update_histogram_thread():
    global er_wn_hist_cache, tof_hist_cache, tof_rw_hist_cache, cache_lock
    while True:
        with cache_lock:
            er_wn_hist_cache = viz_tool._compute_er_wn_hist()
            tof_hist_cache = viz_tool.plot_tof_histogram()
            tof_rw_hist_cache = viz_tool.plot_3d_tof_rw()
        time.sleep(10)  # Update every 10 seconds

threading.Thread(target=update_histogram_thread, daemon=True).start()

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
        viz_tool.update_tof_histogram_bins(tof_hist_min, tof_hist_max, tof_hist_nbins)
    return tof_hist_min, tof_hist_max, tof_hist_nbins

@app.callback(
    [Output('rate-vs-wavenumber', 'figure'),
     Output('events-over-time', 'figure'),
     Output('3d-bar-rate-vs-wavenumber', 'figure'),
     Output('tof-histogram', 'figure'),
     Output('summary-statistics', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('clear-data', 'n_clicks'),
     Input('events-roll-input', 'value'),
     Input('update-tof-histogram', 'n_clicks')],
    [State('events-over-time', 'relayoutData'),
     State('tof-histogram', 'relayoutData')]
)
def update_plots(n_intervals, clear_clicks, events_roll, update_histogram_clicks, events_relayout_data, tof_relayout_data):
    global viz_tool, global_tof_min, global_tof_max, er_wn_hist_cache, tof_hist_cache, tof_rw_hist_cache
    ctx = dash.callback_context
    try:

        if ctx.triggered and 'clear-data' in ctx.triggered[0]['prop_id'] or viz_tool.total_events > TOTAL_MAX_POINTS:
            viz_tool = PlotGenerator()
            return go.Figure(), go.Figure(), go.Figure(), go.Figure(), [dbc.Col("No data available.", width=12)]

        if 'tof-histogram.relayoutData' in ctx.triggered[0]['prop_id'] and tof_relayout_data:
            if 'xaxis.range[0]' in tof_relayout_data and 'xaxis.range[1]' in tof_relayout_data:
                global_tof_min = tof_relayout_data['xaxis.range[0]'] * 1e-6
                global_tof_max = tof_relayout_data['xaxis.range[1]'] * 1e-6

        file_location = load_path()["saving_file"]
        measurement_name = file_location.split("scan_")[-1].split(".")[0]
        minus_time_str = datetime.strptime(measurement_name, "%Y_%m_%d_%H_%M_%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        new_data = query_influxdb(minus_time_str, measurement_name)
        
        viz_tool.update_content(new_data)
        
        with cache_lock:
            fig_events_over_time = viz_tool.plot_events_over_time(roll=events_roll)
            fig_tof_histogram = tof_hist_cache
            fig_total_counts_vs_wavenumber = viz_tool.plot_rate_vs_wavenumber_2d_histogram()
            fig_3d_tof_vs_rw = tof_rw_hist_cache
            status_color = {"color": "green"} if time.time() - viz_tool.historical_data.query('channel!=-1')['timestamp'].max() < 1 else {"color": "red"}
            status_text = "Status: Online" if time.time() - viz_tool.historical_data.query('channel!=-1')['timestamp'].max() < 1 else "Status: Offline"
            summary_text = [
                # Status indicator: green if the last event was less than 1 second ago, red otherwise
                dbc.Col(status_text, style=status_color, width=2),
                dbc.Col(f"Bunch Count: {len(viz_tool.historical_data['bunch'].unique())}", width=2),
                dbc.Col(f"Total Events: {viz_tool.total_events}", width=2),
                dbc.Col(f"Running Time: {round(viz_tool.historical_data['timestamp'].max() - viz_tool.historical_data['timestamp'].min(), 2)} s", width=3),
                dbc.Col(f"Time since last event: {round(time.time() - viz_tool.historical_data.query('channel!=-1')['timestamp'].max(), 2)} s", width=3),
                dbc.Col(f"λ1: {viz_tool.historical_data['wn_1'].iloc[-1].round(12)}", width=2),
                dbc.Col(f"Voltage: {viz_tool.historical_data['voltage'].iloc[-1]} V", width=2),
            ]

        return fig_total_counts_vs_wavenumber, fig_events_over_time, fig_3d_tof_vs_rw, fig_tof_histogram, summary_text
    
    except Exception as e:
        print(f"Error updating plots: {e}")
        loading_icon = go.Scatter(x=[0], y=[0], mode="text", text=["Loading..."], textfont_size=24, textposition="middle center")
        return go.Figure(data=loading_icon), go.Figure(data=loading_icon), go.Figure(data=loading_icon), go.Figure(data=loading_icon), [dbc.Col("No data available.", width=12)]

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
