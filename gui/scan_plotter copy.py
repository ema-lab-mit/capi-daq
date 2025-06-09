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
import json

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
NBATCH = 5_00
TOTAL_MAX_POINTS = int(100_000)
MAX_POINTS_FOR_PLOT = 500

# Default settings
default_settings = {
    "tof_hist_nbins": 50,
    "tof_hist_min": 0.5e-6,
    "tof_hist_max": 20e-6,
    "rolling_window": 10,
    "wn_bin_width_mhz": 10,
    "pv_name": "wavenumber_3",
}
# Attempt to load user settings
try:
    with open(SETTINGS_PATH, 'r') as f:
        user_settings = json.load(f)
        default_settings["tof_hist_min"] = float(user_settings.get("tof_hist_min", default_settings["tof_hist_min"]))
        default_settings["tof_hist_max"] = float(user_settings.get("tof_hist_max", default_settings["tof_hist_max"]))
        default_settings["pv_name"] = user_settings.get("pv_name", default_settings["pv_name"])
        print("UPDATED tof SETTINGS_PATH")
except Exception as e:  
    print(f"Error loading user settings: {e}")
    pass

CHANNEL_USED =  default_settings["pv_name"].replace("wavenumber_", "wn_")
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
        self.tof_hist_max = settings_dict.get("tof_hist_max", 40e-6)
        self.rolling_window = settings_dict.get("rolling_window", 10)
        self.wn_bin_width_mhz = settings_dict.get("wn_bin_width_mhz", 10)
        self.wn_bin_width_cm1 = self.wn_bin_width_mhz / 29.9792458e3
        
        # NEW: λ offset attribute
        self.wn_offset = 0.0

        self.historical_data = pd.DataFrame()
        self.unseen_new_data = pd.DataFrame()
        self.padded_historical_data = pd.DataFrame()
        self.non_duplicated_data = pd.DataFrame()
        
        self.last_loaded_time = None
        self.first_time = time.time()
        self.last_wavenumber = 0

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

    # NEW: Method to update the offset
    def update_wn_offset(self, offset_value: float):
        self.wn_offset = offset_value

    def _update_tof_statistics(self, unseen_new_data):
        events_data = self.unseen_new_data
        if events_data.empty:
            return
        self.total_events += len(events_data)
        if len(events_data) > 0:
            offsets = events_data["time_offset"].values
            new_hist_counts, _ = np.histogram(offsets, bins=self.tof_histogram_bins)
            self.histogram_counts += new_hist_counts
            bin_centers = 0.5*(self.tof_histogram_bins[:-1]+self.tof_histogram_bins[1:])
            total_count = np.sum(self.histogram_counts)
            if total_count > 0:
                self.tof_mean = np.average(bin_centers, weights=self.histogram_counts)
                self.tof_var  = np.average((bin_centers - self.tof_mean)**2, weights=self.histogram_counts)
                
    def update_content(self, new_data):
        print("yes")
        if not self.historical_data.empty:
            unseen_new_data = new_data[~new_data["id_timestamp"].isin(self.historical_data["id_timestamp"])]
        else:
            unseen_new_data = new_data
        
        if not unseen_new_data.empty:
            unseen_new_data = unseen_new_data[
                ((new_data["time_offset"] >= global_tof_min) & (new_data["time_offset"] <= global_tof_max))
            ]
            self.unseen_new_data = unseen_new_data
        
        if not unseen_new_data.empty and "trigger_rate" in unseen_new_data.columns:
            # Use the last nonzero trigger_rate found
            last_trigger_rate_series = unseen_new_data[unseen_new_data["trigger_rate"] != 0]["trigger_rate"]
            if not last_trigger_rate_series.empty:
                self.trigger_rate = last_trigger_rate_series.iloc[-1]
    
        # Update hist, etc.
        self.historical_data = pd.concat([self.historical_data, unseen_new_data]).tail(TOTAL_MAX_POINTS)
        # If there were no new points, add a dummy point to keep the plot updating to self.padded_historical_data
        if unseen_new_data.empty:
            if not self.padded_historical_data.empty:
                last_id_ts = self.padded_historical_data["id_timestamp"].values[-1]
            elif not self.historical_data.empty:
                last_id_ts = self.historical_data["id_timestamp"].values[-1]
            else:
                last_id_ts = 0

            dummy_data = pd.DataFrame(
                {
                    "bunch": [self.historical_data["bunch"].values[-1] if not self.historical_data.empty else 0],
                    "n_events": [0],
                    "time_offset": [0],
                    "id_timestamp": [last_id_ts + 0.5],
                    "trigger_rate": [getattr(self, "trigger_rate", 0)],
                }
            )
            self.padded_historical_data = pd.concat([self.padded_historical_data, dummy_data]).tail(2_000)
        else:
            self.padded_historical_data = pd.concat([self.padded_historical_data, unseen_new_data.drop_duplicates(subset=["bunch"])])
            self.non_duplicated_data = pd.concat([self.non_duplicated_data, unseen_new_data.drop_duplicates(subset=["bunch"])])
            
        self._update_tof_statistics(unseen_new_data)
        # Drop duplicates from historical_data
        self.historical_data = self.historical_data.drop_duplicates(subset=["id_timestamp", "bunch"])
        self.last_wavenumber = self.historical_data[CHANNEL_USED].iloc[-1] if not self.historical_data.empty else 0
        self.last_loaded_time = self.historical_data["id_timestamp"].max() if not self.historical_data.empty else None

    
    def plot_events_over_time(self, max_points=100, yaxis_range=None,
                              show_rolling_average=True, rolling_window_size=10):
        try:
            fig = go.Figure()
            
            df = self.padded_historical_data.copy()
            df["id_timestamp"] = pd.to_datetime(df["id_timestamp"], unit="s", utc=True)
            df.set_index("id_timestamp", inplace=True)
            events_per_second = df["n_events"].tail(500).resample("S").sum()
            times = events_per_second.index
            nevents = events_per_second.values
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=nevents,
                    mode="lines",
                    name="Events per second",
                    line=dict(color="blue"),
                )
            )

            if show_rolling_average and rolling_window_size > 1:
                rolling_avg = np.convolve(nevents, np.ones(rolling_window_size) / rolling_window_size, mode="same")
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(times),
                        y=rolling_avg,
                        mode="lines",
                        name=f"Rolling Avg ({rolling_window_size} pts)",
                        line=dict(color="red"),
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
            scale_factor = np.max(bar_x) / np.max(y) if np.max(y) != 0 else 1
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
        FMW = 2.355 * sigma_us
        fig.add_annotation(
            x=np.max(y_plot),
            y=mean_us + sigma_us,
            text=f"Mean={mean_us:.2f}(±{sigma_us:.2f})µs, FMW={FMW:.2f}µs",
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
        """
        An upgraded, more robust method for plotting event rate vs. wavenumber.
        
        We add error bars (standard error of the mean) to each bin:
          SEM = std / sqrt(N)
        where std is the sample standard deviation in that bin 
        and N is the number of points in that bin.
        """
        if self.historical_data.empty:
            return go.Figure()
        
        df = self.non_duplicated_data.copy()
        df = df[df["trigger_rate"] > 0]
        if df.empty:
            return go.Figure()
        
        # Each row's local rate
        df["local_rate"] = df["event_rate"]
        
        wn_col = CHANNEL_USED  # e.g. 'wn_3'
        wn_min, wn_max = df[wn_col].min(), df[wn_col].max()
        
        # Guard against zero or negative bin width
        if self.wn_bin_width_cm1 <= 0:
            bins = 1
        else:
            wn_range = wn_max - wn_min
            bins = max(int(np.ceil(wn_range / self.wn_bin_width_cm1)), 1)
        
        # Bin edges and bin assignment
        bin_edges = np.linspace(wn_min, wn_max, bins + 1)
        df["wn_bin"] = pd.cut(df[wn_col], bin_edges)
        
        # We compute mean, count, and std per bin
        grouped = df.groupby("wn_bin")["local_rate"].agg(["mean", "count", "std"])
        
        # Standard error of the mean
        grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
        grouped["sem"] = grouped["sem"].fillna(0)  # replace NaNs with 0 if any
        
        # Build a plotting DataFrame
        bin_mids = np.array([interval.mid for interval in grouped.index])
        plot_df = pd.DataFrame({
            "wn_mid": bin_mids - self.wn_offset, 
            "rate": grouped["mean"],
            "count": grouped["count"],
            "rate_sem": grouped["sem"]
        }).dropna(subset=["rate"])  # remove bins with no data
        
        if plot_df.empty:
            return go.Figure()
        
        # Create Plotly figure with error bars
        fig = px.scatter(
            plot_df,
            x="wn_mid",
            y="rate",
            error_y="rate_sem",             # <-- ADDING ERROR BARS HERE
            template="plotly_white",
            title="Event Rate vs λ",
            labels={"wn_mid": "λ (cm⁻¹)", "rate": "Estimated Rate (events/s)"}
        )
        
        # Draw lines + markers for clarity
        fig.update_traces(mode='lines+markers')
        
        # Add vertical red line for the last wavenumber value received, shifted by wn_offset
        last_wn = self.last_wavenumber - self.wn_offset
        fig.add_vline(x=last_wn, line=dict(color='red', dash='dash'), name='Last λ')
        
        fig.update_layout(
            xaxis_title=f"λ (cm⁻¹) + \nOffset: {self.wn_offset} cm⁻¹",
            yaxis_title="Event Rate (events/s)",
            uirevision='rate_vs_wavenumber'
        )
        return fig

    def plot_3d_tof_rw(self):
        """
        Optional 2D histogram or advanced 3D. This is a placeholder example.
        """
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
            xaxis_title="λ (cm⁻¹)",
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
    |> filter(fn: (r) => r._measurement == "scan")
    |> filter(fn: (r) => r.type == "{measurement_name}")
    |> tail(n: {NBATCH})
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "bunch", "n_events", "channel", "time_offset", "id_timestamp", "wn_1", "wn_2", "wn_3", "wn_4", "voltage", "spectr_peak", "trigger_rate", "event_rate"])
    """
    try:
        result = client.query_api().query(query=query, org=INFLUXDB_ORG)
        records = []
        for table in result:
            for record in table.records:
                records.append(record.values)
        df = pd.DataFrame(records).dropna(how="all")
        df["spectr_peak"] = df["spectr_peak"].astype("float")
        print(df["spectr_peak"])
        return df
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=[
            "_time", "bunch", "n_events", "channel", "time_offset", "id_timestamp",
            "wn_1", "wn_2", "wn_3", "wn_4", "voltage", "spectr_peak", "trigger_rate", "event_rate"
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
    # Updated: Rate vs λ Settings with λ Offset
    dbc.Modal(
        [
            dbc.ModalHeader("Rate vs λ Settings"),
            dbc.ModalBody([
                dbc.Label("Bin Width (MHz) "),
                dcc.Input(
                    id='wn-bin-width-input',
                    type='number',
                    value=default_settings['wn_bin_width_mhz'],
                    min=0.1,
                    step=0.1,
                ),
                html.Hr(),
                dbc.Label("λ Offset (cm⁻¹) "),
                dcc.Input(
                    id='wn-offset-input',
                    type='number',
                    value=0.0,
                    step=0.1
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Update λ Binning", id="update-wn-histogram", className="ml-auto"),
                dbc.Button("Close", id="close-wn-modal", className="ml-auto")
            ])
        ],
        id="wn-settings-modal",
        is_open=False,
    ),
], fluid=True)

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

# Updated callback to also handle wavenumber offset
@app.callback(
    [Output('wn-bin-width-input', 'value'),
     Output('wn-offset-input', 'value')],
    [Input('update-wn-histogram', 'n_clicks')],
    [State('wn-bin-width-input', 'value'),
     State('wn-offset-input', 'value')]
)
def update_wn_histogram_settings(n_clicks, wn_bin_width_mhz, wn_offset_value):
    if n_clicks:
        viz_tool.update_wn_bin_width(wn_bin_width_mhz)
        viz_tool.update_wn_offset(wn_offset_value)
    return wn_bin_width_mhz, wn_offset_value

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
    minus_time_str = f"0"  # Start from the beginning
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
                tmax = viz_tool.historical_data['id_timestamp'].max()
                time_since_last_event = time.time() - tmax
                if time_since_last_event < 10.:
                    status_color = "green"
                    status_text = "Status: Online"

        if not viz_tool.historical_data.empty:
            bunch_count = len(viz_tool.historical_data['bunch'].unique())
            total_events = viz_tool.total_events
            scan_id = measurement_name
            if '_time' in viz_tool.historical_data.columns:
                time_span = time.time() - viz_tool.first_time
                running_time = time_span
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
            dbc.Col(f"Running Time: {running_time:.2f} s", width=3),
            dbc.Col(f"Time since last event: {time_since_last_event:.5f} s", width=3),
            dbc.Col(f"Voltage: {last_voltage:.5f} V", width=2),
            dbc.Col(f"Scan ID: {scan_id}", width=2),
            dbc.Col(f"λ [{CHANNEL_USED}]: {viz_tool.last_wavenumber} cm⁻¹", width=2),
            dbc.Col(f"Trigger Rate: {getattr(viz_tool, 'trigger_rate', 0):.2f} Hz", width=2),
            dbc.Col(f"Event Rate: {viz_tool.historical_data['event_rate'].iloc[-1]:.2f} Hz", width=2),
            dbc.Col(f"Spectrum Peak: {viz_tool.historical_data['spectr_peak'].values[-1]} Hz", width=2),
        ]

    return (
        fig_total_counts_vs_wavenumber,
        fig_events_over_time,
        fig_3d_tof_vs_rw,
        fig_tof_histogram,
        summary_text
    )

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
