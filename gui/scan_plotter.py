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
TOTAL_MAX_POINTS = int(10_000_000)
MAX_POINTS_FOR_PLOT = 500

CHANNEL_USED =  'wn_1'

# Default settings
default_settings = {
    "tof_hist_nbins": 100,
    "tof_hist_min": 1e-6,
    "tof_hist_max": 20e-6,
    "rolling_window": 10,
    "wn_bin_width_mhz": 10,
}

# Initialize InfluxDB
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

global_tof_min = default_settings['tof_hist_min']
global_tof_max = default_settings['tof_hist_max']

cache_lock = threading.Lock()

class PlotGenerator:
    def __init__(self, settings_dict: dict = default_settings):
        self.settings_dict = settings_dict
        self.tof_hist_nbins = settings_dict.get("tof_hist_nbins", 100)
        self.tof_hist_min = settings_dict.get("tof_hist_min", 0)
        self.tof_hist_max = settings_dict.get("tof_hist_max", 20e-6)
        self.rolling_window = settings_dict.get("rolling_window", 10)
        self.wn_bin_width_mhz = settings_dict.get("wn_bin_width_mhz", 10)
        self.wn_bin_width_cm1 = self.wn_bin_width_mhz / 29.9792458e3

        self.historical_data = pd.DataFrame()
        self.unseen_new_data = pd.DataFrame()
        
        self.last_loaded_time = None
        self.first_time = time.time()

        # For the histogram
        self.tof_histogram_bins = np.linspace(self.tof_hist_min, self.tof_hist_max, self.tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)
        self.tof_mean = 0
        self.tof_var = 0
        self.total_events = 0
        self.first_time = time.time()
        self.last_loaded_time = time.time()

    def update_tof_histogram_bins(self, tof_hist_min, tof_hist_max, tof_hist_nbins):
        self.tof_hist_min = tof_hist_min
        self.tof_hist_max = tof_hist_max
        self.tof_hist_nbins = tof_hist_nbins
        self.tof_histogram_bins = np.linspace(tof_hist_min, tof_hist_max, tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)

    def update_wn_bin_width(self, wn_bin_width_mhz):
        self.wn_bin_width_mhz = wn_bin_width_mhz
        self.wn_bin_width_cm1 = wn_bin_width_mhz / 29.9792458e3

    def _update_tof_statistics(self, unseen_new_data):
        if unseen_new_data.empty:
            return
        # Only keep valid channels and time offsets in [global_tof_min, global_tof_max]
        events_data = events_data[
            (events_data['_time_offset'] >= global_tof_min) &
            (events_data['_time_offset'] <= global_tof_max)
        ]
        self.total_events += len(events_data)
        if len(events_data) > 0:
            offsets = events_data["time_offset"].values
            new_hist_counts, _ = np.histogram(offsets, bins=self.tof_histogram_bins)
            self.histogram_counts += new_hist_counts
            # Weighted average for mean & variance
            bin_centers = 0.5*(self.tof_histogram_bins[:-1]+self.tof_histogram_bins[1:])
            total_count = np.sum(self.histogram_counts)
            if total_count > 0:
                self.tof_mean = np.average(bin_centers, weights=self.histogram_counts)
                self.tof_var  = np.average((bin_centers - self.tof_mean)**2, weights=self.histogram_counts)

    def _update_historical_data(self, unseen_new_data):
        self.historical_data = pd.concat(
            [self.historical_data, unseen_new_data],
            ignore_index=True
        )
        if self.historical_data.shape[0] > TOTAL_MAX_POINTS:
            self.historical_data = self.historical_data.tail(TOTAL_MAX_POINTS)

    def update_content(self, new_data):
        if not self.historical_data.empty:
            unseen_new_data = new_data[~new_data["id_timestamp"].isin(self.historical_data["id_timestamp"])]
        else:
            unseen_new_data = new_data
        if not new_data.empty:
            self.last_loaded_time = new_data['_time'].max().timestamp()
        print(len(new_data), len(unseen_new_data))
        unseen_new_data = unseen_new_data[
            ((new_data["time_offset"] >= global_tof_min) & (new_data["time_offset"] <= global_tof_max))
        ]
        self.unseen_new_data = unseen_new_data
        # if there's a non-zero trigger_rate in unseen_new_data, update self.trigger_rate
        if not unseen_new_data.empty and "trigger_rate" in unseen_new_data.columns:
            # Use the last nonzero trigger_rate found
            last_trigger_rate_series = unseen_new_data[unseen_new_data["trigger_rate"] != 0]["trigger_rate"]
            if not last_trigger_rate_series.empty:
                self.trigger_rate = last_trigger_rate_series.iloc[-1]
        # Update hist, etc.
        self.historical_data = pd.concat([self.historical_data, unseen_new_data]).tail(TOTAL_MAX_POINTS)

        self._update_tof_statistics(unseen_new_data)

    def plot_events_over_time(self, max_points=100, yaxis_range=None,
                              show_rolling_average=False, rolling_window_size=100):
        try:
            fig = go.Figure()
            

            df = self.historical_data.copy().drop_duplicates(subset=["bunch"])
            if len(df) == 0:
                return fig
            
            events_per_trigger = df["n_events"].values
            trigger_rate = df["trigger_rate"].values
            
            events_per_second = events_per_trigger * trigger_rate

            df["id_timestamp"] = pd.to_datetime(df["id_timestamp"], unit="s")
            times_events = df["id_timestamp"].values
            
            # If the last event was more than 1 second ago, add a dummy point with 0 events
            if time.time() - self.last_loaded_time > 1.0:
                times_events = np.append(times_events, times_events[-1] + np.timedelta64(1, 's'))
                events_per_second = np.append(events_per_second, 0)

            fig.add_trace(
                go.Scatter(
                    x=times_events,
                    y=events_per_second,
                    mode="lines",
                    name="Events per second",
                    line=dict(color="blue"),
                )
            )

            fig.update_layout(
                xaxis_title="Monitoring Time (s)",
                yaxis_title="Total Events/s",
                template="plotly_white",
                uirevision="events_over_time",
            )

            if yaxis_range is not None:
                fig.update_yaxes(range=yaxis_range)

            return fig
        except Exception as e:
            print(f"Error in plot_events_over_time: {e}")
            return go.Figure()
        
    def plot_tof_histogram(self):
        # same as your original, except we keep it consistent
        fig = go.Figure()
        total_counts = np.sum(self.histogram_counts)
        if total_counts == 0:
            fig.update_layout(
                title="Time of Flight Histogram",
                xaxis_title="Normalized Counts",
                yaxis_title="Time of Flight (µs)",
                template="plotly_white",
                uirevision='tof_histogram'
            )
            return fig

        # Build bar plot
        bar_x = self.histogram_counts / total_counts
        bar_y = self.tof_histogram_bins[1:] * 1e6  # convert to microseconds

        fig = px.bar(
            x=bar_x,
            y=bar_y,
            orientation='h',
            template="plotly_white",
            labels={"x": "Normalized Counts", "y": "Time of Flight (µs)"},
            title="Time of Flight Histogram"
        )

        mean_us = self.tof_mean * 1e6
        var_us2 = self.tof_var * 1e12
        sigma_us = np.sqrt(var_us2) if var_us2 > 0 else 0

        # Attempt a Gaussian overlay
        x = np.linspace(self.tof_hist_min*1e6, self.tof_hist_max*1e6, 1000)
        if sigma_us > 1e-12:
            y = norm.pdf(x, mean_us, sigma_us)
            # scale max to the histogram's max (roughly):
            scale_factor = np.max(bar_x) / np.max(y)
            y_plot = y * scale_factor
        else:
            y_plot = np.zeros_like(x)

        fig.add_trace(
            go.Scatter(
                x=y_plot, y=x, mode="lines", name="Gaussian Fit", line=dict(color="red")
            )
        )

        fig.add_shape(
            dict(
                type="line",
                x0=0, y0=mean_us,
                x1=np.max(y_plot), y1=mean_us,
                line=dict(color="black", width=2)
            )
        )
        fig.add_annotation(
            x=np.max(y_plot),
            y=mean_us,
            text=f"Mean={mean_us:.2f}µs ± {sigma_us:.2f}µs",
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(
            xaxis_title="Normalized Counts",
            yaxis_title="Time of Flight (µs)",
            uirevision='tof_histogram'
        )
        return fig

    def plot_rate_vs_wavenumber_2d_histogram(self):
        if self.historical_data.empty:
            return go.Figure()

        df = self.historical_data.copy()
        df['t_s'] = df['_time'].apply(lambda x: x.timestamp())
        df.sort_values('t_s', inplace=True)
        df['_time_diff'] = df['t_s'].shift(-1) - df['t_s']
        if len(df) > 1:
            df.at[df.index[-1], '_time_diff'] = df['_time_diff'].iloc[-2]
        else:
            df.at[df.index[-1], '_time_diff'] = 0
        df['_time_diff'] = df['_time_diff'].clip(lower=1e-9)

        wn_col = CHANNEL_USED
        wn_min, wn_max = df[wn_col].min(), df[wn_col].max()
        wn_range = wn_max - wn_min
        if self.wn_bin_width_cm1 == 0:
            bins = 1
        else:
            bins = int(np.ceil(wn_range / self.wn_bin_width_cm1))
        bins = max(bins, 1)
        bin_edges = np.linspace(wn_min, wn_max, bins+1)
        df['wn_bin'] = pd.cut(df[wn_col], bin_edges)

        grouped = df.groupby('wn_bin')
        total_events = grouped['n_events'].sum()
        total_time   = grouped['_time_diff'].sum()

        mask_nz = total_time > 0
        total_events = total_events[mask_nz]
        total_time   = total_time[mask_nz]
        if total_events.empty:
            return go.Figure()

        rate = total_events / total_time
        bin_labels = total_events.index.categories
        bin_mid = [interval.mid for interval in bin_labels if interval in rate.index]

        plot_df = pd.DataFrame({'wn_mid': bin_mid, 'rate': rate.values})
        fig = px.scatter(
            plot_df,
            x='wn_mid',
            y='rate',
            template='plotly_white',
            title='Event Rate vs Wavenumber',
            labels={'wn_mid': 'Wavenumber (cm⁻¹)', 'rate': 'Rate (events/s)'}
        )
        fig.update_traces(mode='lines+markers')
        fig.update_layout(
            xaxis_title="Wavenumber (cm^-1)",
            yaxis_title="Event Rate (events/s)",
            uirevision='rate_vs_wavenumber'
        )
        return fig

    def plot_3d_tof_rw(self):
        """
        Optional 2D histogram or any advanced 3D. We keep as is for brevity.
        """
        # Example same code
        if self.historical_data.empty:
            return go.Figure()

        df_events = self.historical_data.query("channel != -1")
        wn_col = CHANNEL_USED
        fig = px.density_heatmap(
            df_events,
            x=wn_col,
            y="time_offset",
            nbinsx=50, nbinsy=50,
            title="Event Rate vs ToF",
            template="plotly_white",
            marginal_x="histogram",
            marginal_y="violin"
        )
        fig.update_layout(
            xaxis_title="Wavenumber (cm^-1)",
            yaxis_title="Time of Flight (s)",
            uirevision='rate_vs_wavenumber'
        )
        return fig


# --------------------------------------------------------------------------------
# Query to fetch data from InfluxDB
# --------------------------------------------------------------------------------
def query_influxdb(minus_time_str, measurement_name):
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: {minus_time_str})
    |> filter(fn: (r) => r._measurement == "hits")
    |> filter(fn: (r) => r.type == "{measurement_name}")
    |> tail(n: {NBATCH})
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "bunch", "n_events", "channel", "time_offset", "id_timestamp", "wn_1", "wn_2", "wn_3", "wn_4", "voltage", "trigger_rate"])
    """
    try:
        result = client.query_api().query(query=query, org=INFLUXDB_ORG)
        records = []
        for table in result:
            for record in table.records:
                records.append(record.values)
        df = pd.DataFrame(records).dropna(how="all")
        return df
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=[
            "_time", "bunch", "n_events", "channel", "time_offset", "id_timestamp",
            "wn_1", "wn_2", "wn_3", "wn_4", "voltage", "trigger_rate"
        ])


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

viz_tool = PlotGenerator()

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
            dbc.Row([
                dbc.Col(width=4),
                dbc.Col(dbc.Button("+", id="wn-settings-button", n_clicks=0, className="d-block mx-auto"), width=4),
                dbc.Col(width=4)
            ])
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
            dbc.Row([
                dbc.Col(width=4),
                dbc.Col(dbc.Button("+", id="tof-settings-button", n_clicks=0, className="d-block mx-auto"), width=4),
                dbc.Col(width=4)
            ])
        ], width=6)
    ], className="mb-4"),
    dcc.Interval(id='interval-component', interval=0.3*1000, n_intervals=0),
    dbc.Offcanvas(
        [
            dbc.Row([
                dbc.Col(html.Div("Refresh Rate (seconds): ")),
                dbc.Col(
                    dcc.Slider(
                        id='refresh-rate',
                        min=0.2,
                        max=10.0,
                        step=0.5,
                        value=0.5,
                        tooltip={"placement": "bottom", "always_visible": True},
                        marks={i: str(i) for i in np.arange(0.5, 11, 0.5)}
                    )
                ),
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
                dbc.Label("Integration (rolling) Window = "),
                dcc.Input(id='events-roll-input', type='number', value=10, min=1),
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
                dbc.Label("Min (s)"),
                dcc.Input(id='tof-hist-min-input', type='number', value=default_settings['tof_hist_min'], step=1e-6, min=0),
                dbc.Label("Max (s)"),
                dcc.Input(id='tof-hist-max-input', type='number', value=default_settings['tof_hist_max'], step=1e-6, min=0),
                dbc.Label("Number of Bins"),
                dcc.Slider(
                    id='tof-bins-slider',
                    min=1,
                    max=100,
                    step=5,
                    value=default_settings['tof_hist_nbins'],
                    marks={i: str(i) for i in range(5, 101, 5)}
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Update Histogram", id="update-tof-histogram", className="ml-auto"),
                dbc.Button("Close", id="close-tof-modal", className="ml-auto")
            ])
        ],
        id="tof-settings-modal",
        is_open=False,
    ),
    dbc.Modal(
        [
            dbc.ModalHeader("Rate vs Wavenumber Settings"),
            dbc.ModalBody([
                dbc.Label("Bin Width (MHz)"),
                dcc.Input(
                    id='wn-bin-width-input',
                    type='number',
                    value=default_settings['wn_bin_width_mhz'],
                    min=0.1,
                    step=0.1,
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Update Wavenumber Binning", id="update-wn-histogram", className="ml-auto"),
                dbc.Button("Close", id="close-wn-modal", className="ml-auto")
            ])
        ],
        id="wn-settings-modal",
        is_open=False,
    ),
], fluid=True)


def update_histogram_thread():
    """
    (Optional) If you want background updates. Currently not strictly needed 
    if we do everything in the callback anyway.
    """
    while True:
        with cache_lock:
            # do any heavy-lifting or caching
            pass
        time.sleep(10)

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
    Output("wn-settings-modal", "is_open"),
    [Input("wn-settings-button", "n_clicks"), Input("close-wn-modal", "n_clicks")],
    [State("wn-settings-modal", "is_open")]
)
def toggle_wn_settings(n1, n2, is_open):
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
    [Output('tof-hist-min-input', 'value'),
     Output('tof-hist-max-input', 'value'),
     Output('tof-bins-slider', 'value')],
    [Input('update-tof-histogram', 'n_clicks')],
    [State('tof-hist-min-input', 'value'),
     State('tof-hist-max-input', 'value'),
     State('tof-bins-slider', 'value')]
)
def update_tof_histogram_settings(n_clicks, tof_hist_min, tof_hist_max, tof_hist_nbins):
    if n_clicks:
        viz_tool.update_tof_histogram_bins(tof_hist_min, tof_hist_max, tof_hist_nbins)
    return tof_hist_min, tof_hist_max, tof_hist_nbins

@app.callback(
    Output('wn-bin-width-input', 'value'),
    [Input('update-wn-histogram', 'n_clicks')],
    [State('wn-bin-width-input', 'value')]
)
def update_wn_histogram_settings(n_clicks, wn_bin_width_mhz):
    if n_clicks:
        viz_tool.update_wn_bin_width(wn_bin_width_mhz)
    return wn_bin_width_mhz

@app.callback(
    [
        Output('rate-vs-wavenumber', 'figure'),
        Output('events-over-time', 'figure'),
        Output('3d-bar-rate-vs-wavenumber', 'figure'),
        Output('tof-histogram', 'figure'),
        Output('summary-statistics', 'children')
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('clear-data', 'n_clicks'),
        Input('events-roll-input', 'value'),
        Input('update-tof-histogram', 'n_clicks'),
        Input('update-wn-histogram', 'n_clicks')
    ]
)
def update_plots(n_intervals, clear_clicks, *_):
    global viz_tool, global_tof_min, global_tof_max
    ctx = dash.callback_context

    # Handle clear data
    if ctx.triggered and 'clear-data' in ctx.triggered[0]['prop_id']:
        viz_tool = PlotGenerator()
        return (go.Figure(), go.Figure(), go.Figure(), go.Figure(), [dbc.Col("Data cleared.", width=12)])

    # If we have a huge number of events, forcibly reset
    if viz_tool.total_events > TOTAL_MAX_POINTS:
        viz_tool = PlotGenerator()
        return (go.Figure(), go.Figure(), go.Figure(), go.Figure(), [dbc.Col("Data capacity reached.", width=12)])

    # 1) Query new data from Influx
    file_location = load_path()["saving_file"]
    measurement_name = os.path.basename(file_location).split("scan_")[-1].split(".")[0]
    minus_time_str = f"0"  # Start from the beginning, or you can parse measurement_name if you want
    new_data = query_influxdb(minus_time_str, measurement_name)

    if new_data.empty and len(viz_tool.historical_data) == 0:
        return (
            go.Figure(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            [dbc.Col("No data available yet.", width=12)],
        )
        
    with cache_lock:
        viz_tool.update_content(new_data)

        fig_total_counts_vs_wavenumber = viz_tool.plot_rate_vs_wavenumber_2d_histogram()
        fig_events_over_time = viz_tool.plot_events_over_time()
        fig_3d_tof_vs_rw = viz_tool.plot_3d_tof_rw()
        fig_tof_histogram = viz_tool.plot_tof_histogram()

        # 4) Generate summary
        status_color = "red"
        status_text = "Status: Offline"
        time_since_last_event = 9999
        if not viz_tool.historical_data.empty:
            if '_time' in viz_tool.historical_data.columns:
                tmax = viz_tool.historical_data['_time'].max()
                time_since_last_event = time.time() - tmax.timestamp()
                if time_since_last_event < 1.0:
                    status_color = "green"
                    status_text = "Status: Online"

        if not viz_tool.historical_data.empty:
            bunch_count = len(viz_tool.historical_data['bunch'].unique())
            total_events = viz_tool.total_events
            scan_id = measurement_name
            if '_time' in viz_tool.historical_data.columns:
                time_span = viz_tool.historical_data['_time'].max() - viz_tool.historical_data['_time'].min()
                running_time = round(time_span.total_seconds(),2)
            else:
                running_time = 0
            last_voltage = viz_tool.historical_data['voltage'].iloc[-1] if 'voltage' in viz_tool.historical_data else 0
        else:
            bunch_count = 0
            total_events = 0
            running_time = 0
            last_voltage = 0
            scan_id = "N/A"

        summary_text = [
            dbc.Col(html.Div(status_text, style={'color': status_color}), width=2),
            dbc.Col(f"Bunch Count: {bunch_count}", width=2),
            dbc.Col(f"Total Events: {total_events}", width=2),
            dbc.Col(f"Running Time: {running_time} s", width=3),
            dbc.Col(f"Time since last event: {time_since_last_event:.2f} s", width=3),
            dbc.Col(f"Voltage: {last_voltage:.2f} V", width=2),
            dbc.Col(f"Scan ID: {scan_id}", width=2),
        ]

    return (
        fig_total_counts_vs_wavenumber,
        fig_events_over_time,
        fig_3d_tof_vs_rw,
        fig_tof_histogram,
        summary_text
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
