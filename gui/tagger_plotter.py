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
import json
import time
from collections import deque

from influxdb_client import InfluxDBClient
from scipy.stats import norm

warnings.simplefilter("ignore")

# --------------------------------------------------------------------------------
# Adjust to your actual paths and environment
# --------------------------------------------------------------------------------
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)

SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"

# --------------------------------------------------------------------------------
# Default settings, overridden by the JSON file if present
# --------------------------------------------------------------------------------
default_settings = {
    "tof_hist_nbins": 100,
    "tof_hist_min": 1e-6,   # 1 microsecond
    "tof_hist_max": 150e-6, # 150 microseconds
    "plot_rolling_window": 100,
    "integration_window": 10,
}

# Attempt to load user settings
try:
    with open(SETTINGS_PATH, 'r') as f:
        user_settings = json.load(f)
        default_settings["tof_hist_min"] = float(user_settings.get("tof_hist_min", default_settings["tof_hist_min"]))
        default_settings["tof_hist_max"] = float(user_settings.get("tof_hist_max", default_settings["tof_hist_max"]))
        print("UPDATED tof SETTINGS_PATH")
except Exception as e:  
    print(f"Error loading user settings: {e}")
    pass

# --------------------------------------------------------------------------------
# Global parameters and environment variables
# --------------------------------------------------------------------------------
global_tof_min = default_settings["tof_hist_min"]
global_tof_max = default_settings["tof_hist_max"]

db_token = os.getenv("INFLUXDB_TOKEN", "")
if not db_token:
    # Fallback, or read from somewhere else if needed
    db_token = "mTS9iQyj_ua5sAbJZ0ubOWQMQX_VcwhjhnZaikS1ImQuuY7xGtZDqelrugvBThfxpwYbuomtIyFMEkZdV657PA=="  

os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"

NBATCH = 2_00
TOTAL_MAX_POINTS = 50_000
MAX_POINTS_FOR_PLOT = 100
BEAMLINE_FREQUENCY = 50  # Hz

REFRESH_RATE = 0.5 # seconds

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

# --------------------------------------------------------------------------------
# Plotting/Computation Utilities with Optimized Data Handling
# --------------------------------------------------------------------------------
class PlotGenerator:
    def __init__(self, settings_dict: dict = default_settings):
        self.settings_dict = settings_dict
        self.init_time = time.time()
        self.tof_hist_nbins = settings_dict.get("tof_hist_nbins", 100)
        self.tof_hist_min = settings_dict.get("tof_hist_min", 1e-6)
        self.tof_hist_max = settings_dict.get("tof_hist_max", 150e-6)
        self.plot_rolling_window = settings_dict.get("plot_rolling_window", 100)
        self.integration_window = settings_dict.get("integration_window", 10)

        # Define maximum lengths for historical data to prevent memory bloat
        self.max_historical_length = 10000  # Adjust based on your requirements

        self.historical_data = pd.DataFrame()

        self.last_loaded_time = None
        self.first_time = time.time()
        self.number_records = 0
        self.last_data_batch = pd.DataFrame()

        self.tof_mean = 0
        self.tof_var = 0
        self.tof_histogram_bins = np.linspace(self.tof_hist_min, self.tof_hist_max, self.tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)

        self.prev_tof_hist_min = self.tof_hist_min
        self.prev_tof_hist_max = self.tof_hist_max
        self.prev_tof_hist_nbins = self.tof_hist_nbins
        self.trigger_rate = 0  # Start with zero if no data.

    def update_histogram_bins(self, tof_hist_min, tof_hist_max, tof_hist_nbins):
        """Explicitly update only if user changes the slider, not from auto-zoom."""
        self.tof_hist_min = tof_hist_min
        self.tof_hist_max = tof_hist_max
        self.tof_hist_nbins = tof_hist_nbins
        self.tof_histogram_bins = np.linspace(tof_hist_min, tof_hist_max, tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)

    def _update_tof_statistics(self, unseen_new_data):
        if len(unseen_new_data) == 0:
            return
        events_data = self.historical_data
        
        events_offset = events_data["time_offset"].values
        if len(events_data) > 0:
            new_hist_counts, _ = np.histogram(events_offset, bins=self.tof_histogram_bins)
            self.histogram_counts += new_hist_counts
            
            # Weighted average for mean and variance
            bin_centers = 0.5 * (self.tof_histogram_bins[:-1] + self.tof_histogram_bins[1:])
            if np.sum(self.histogram_counts) > 0:
                self.tof_mean = np.average(bin_centers, weights=self.histogram_counts)
                self.tof_var = np.average((bin_centers - self.tof_mean) ** 2, weights=self.histogram_counts)
            else:
                self.tof_mean = 0
                self.tof_var = 0

    def update_content(self, new_data: pd.DataFrame):
        """
        Add only the portion of new_data that is more recent than the last loaded time.
        Then update histograms, triggers, etc.
        """
        unseen_new_data = new_data[
            ((new_data["time_offset"] >= global_tof_min) & (new_data["time_offset"] <= global_tof_max))
        ]

        self.trigger_rate = unseen_new_data["trigger_rate"].values[0] if unseen_new_data["trigger_rate"].values[0] !=0 else self.trigger_rate
        
        # Limit the size of historical data to prevent memory bloat
        self.historical_data = pd.concat([self.historical_data, unseen_new_data]).tail(self.max_historical_length)
        
        self.historical_data.drop_duplicates(inplace=True)  # Ensure no duplicates in historical data
        self._update_tof_statistics(unseen_new_data)
        
        
    def plot_events_over_time(self, max_points=100, yaxis_range=None,
                              show_rolling_average=False, rolling_window_size=100):
        try:
            fig = go.Figure()

            df = self.historical_data.copy()
            df["id_timestamp"] = pd.to_datetime(df["id_timestamp"], unit="s")
            df.set_index("id_timestamp", inplace=True)

            # Sum up "n_events" every second
            events_per_second = df["n_events"].resample("1S").sum()
            delta_ts = (events_per_second.index - events_per_second.index.min()).total_seconds()

            if len(delta_ts) > max_points:
                delta_ts = delta_ts[-max_points:]
                events_per_second = events_per_second[-max_points:]

            fig.add_trace(
                go.Scatter(
                    x=delta_ts,
                    y=events_per_second,
                    mode="lines",
                    name="Events per second",
                    line=dict(color="blue"),
                )
            )

            if show_rolling_average and rolling_window_size > 1:
                rolling_avg = events_per_second.rolling(window=rolling_window_size, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=delta_ts,
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
        try:
            fig = go.Figure()
            if len(self.historical_data) == 0:
                return fig

            total_plotted = np.sum(self.histogram_counts)
            if total_plotted < 1:
                return fig

            # Probability distribution
            bin_edges = self.tof_histogram_bins
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            fig = px.bar(
                x=bin_centers * 1e6,
                y=self.histogram_counts,
                labels={"x": "ToF (µs)", "y": "Counts"},
            )
            mean = self.tof_mean * 1e6
            variance = self.tof_var * 1e12
            sigma = np.sqrt(variance) if variance > 0 else 0
            x = np.linspace(self.tof_hist_min * 1e6, self.tof_hist_max * 1e6, 1000)
            if sigma > 0:
                y = norm.pdf(x, mean, sigma)
                # scale it to match max of our histogram
                y_scaled = y * (np.max(self.histogram_counts) / np.max(y)) if np.max(y) > 0 else 0
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_scaled,
                        mode="lines",
                        name=f"Fit: ToF={mean:.2f} ± {sigma:.2f} µs",
                        line=dict(color="red"),
                    )
                )
                # add a vertical line for the mean
                fig.add_shape(
                    dict(
                        type="line",
                        x0=mean,
                        y0=0,
                        x1=mean,
                        y1=np.max(y_scaled),
                        line=dict(color="black", width=2),
                    )
                )

            fig.update_layout(
                xaxis_title="Time of Flight (µs)",
                yaxis_title="Counts",
                uirevision="tof_histogram",
                template="plotly_white",
            )
            return fig
        except Exception as e:
            print(f"Error in plot_tof_histogram: {e}")
            return go.Figure()

    def plot_wavenumbers(self, selected_channels=[1, 2, 3, 4], max_points=200):
        try:
            fig = go.Figure()
            if len(self.historical_data) == 0:
                return fig
            colors = ["blue", "red", "green", "purple"]

            # Convert deque to DataFrame for plotting
            df = pd.DataFrame(self.historical_data)
            if df.empty:
                return fig

            df["_time"] = pd.to_datetime(df["_time"])  # treat _time as real datetimes
            df.set_index("_time", inplace=True)

            decimated_df = df.iloc[-max_points:].copy()
            for i, channel in enumerate(selected_channels):
                wn_key = f"wn_{channel}"
                if wn_key in decimated_df.columns and decimated_df[wn_key].notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=decimated_df.index,
                            y=decimated_df[wn_key],
                            mode="lines",
                            name=f"wn_{channel}",
                            line=dict(color=colors[i % len(colors)]),
                        )
                    )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Wavenumber",
                uirevision="wavenumbers",
                template="plotly_white",
            )
            return fig
        except Exception as e:
            print(f"Error in plot_wavenumbers: {e}")
            return go.Figure()

    def plot_voltage(self, max_points=500):
        try:
            fig = go.Figure()
            if len(self.historical_data) == 0:
                return fig

            # Convert deque to DataFrame for plotting
            df = self.historical_data.copy().sort_values("_time")  # Ensure the DataFrame is sorted by time

            df["_time"] = pd.to_datetime(df["_time"])  # treat Influx time as real datetime
            df.set_index("_time", inplace=True)

            decimated_df = df.iloc[-max_points:].copy()
            if "voltage" in decimated_df.columns and decimated_df["voltage"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=decimated_df.index,
                        y=decimated_df["voltage"],
                        mode="lines",
                        name="Voltage",
                        line=dict(color="orange"),
                    )
                )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Voltage (V)",
                template="plotly_white",
                uirevision="voltage",
            )
            return fig

        except Exception as e:
            print(f"Error in plot_voltage: {e}")
            return go.Figure()


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


# --------------------------------------------------------------------------------
# Dash App Layout
# --------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

viz_tool = PlotGenerator()
first_time = 0

app.layout = dbc.Container(
    [
        # Navbar
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

        # Summary Statistics Row
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [dbc.Row(id="summary-statistics", className="card-text")]
                        ),
                        style={"height": "100%"},
                    ),
                    width=12,
                )
            ],
            className="mb-4",
        ),

        # Top Row: Events Over Time + ToF Histogram
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="events-over-time", style={"height": "400px"}),
                        dbc.Row(
                            [
                                dbc.Col(width=4),
                                dbc.Col(dbc.Button("+", id="events-settings-button", n_clicks=0, className="d-block mx-auto"), width=4),
                                dbc.Col(width=4),
                            ]
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="tof-histogram", style={"height": "400px"}),
                        dbc.Row(
                            [
                                dbc.Col(width=4),
                                dbc.Col(dbc.Button("+", id="tof-settings-button", n_clicks=0, className="d-block mx-auto"), width=4),
                                dbc.Col(width=4),
                            ]
                        ),
                    ],
                    width=6,
                ),
            ],
            className="mb-4",
        ),

        # Bottom Row: Wavenumbers + Voltage
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="wavenumbers", style={"height": "300px"})], width=6),
                dbc.Col([dcc.Graph(id="voltage", style={"height": "300px"})], width=6),
            ],
            className="mb-4",
        ),

        # Interval for updates
        dcc.Interval(id="interval-component", interval=REFRESH_RATE * 1000, n_intervals=0),

        # Offcanvas for general settings
        dbc.Offcanvas(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Div("Refresh Rate (seconds): "), width=4),
                        dbc.Col(
                            dcc.Slider(
                                id="refresh-rate",
                                min=0.2,
                                max=10.0,
                                step=0.1,
                                value=REFRESH_RATE,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            width=8,
                        ),
                    ],
                    style={"padding": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div("Batch Size (NBATCH): "), width=4),
                        dbc.Col(dcc.Input(id="nbatch-input", type="number", value=NBATCH, step=100), width=8),
                    ],
                    style={"padding": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div("Total Max Points: "), width=4),
                        dbc.Col(dcc.Input(id="total-max-points-input", type="number", value=TOTAL_MAX_POINTS, step=1000), width=8),
                    ],
                    style={"padding": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div("Max Points for Plot: "), width=4),
                        dbc.Col(dcc.Input(id="max-points-for-plot-input", type="number", value=MAX_POINTS_FOR_PLOT, step=100), width=8),
                    ],
                    style={"padding": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div("Integration Window: "), width=4),
                        dbc.Col(dcc.Input(id="integration-window-input", type="number",
                                          value=default_settings["integration_window"], step=1), width=8),
                    ],
                    style={"padding": "20px"},
                ),
            ],
            id="offcanvas",
            is_open=False,
            title="Settings",
        ),

        # Events Over Time modal
        dbc.Modal(
            [
                dbc.ModalHeader("Events Over Time Settings"),
                dbc.ModalBody(
                    [
                        dbc.Label("Show Rolling Average:"),
                        dbc.Checklist(
                            options=[{"label": "Show Rolling Average", "value": "show_rolling_average"}],
                            value=[],
                            id="show-rolling-average-checkbox",
                            inline=True,
                        ),
                        html.Br(),
                        dbc.Label("Rolling Window Size (points):"),
                        dcc.Input(id="events-rolling-window-size", type="number",
                                  value=default_settings["plot_rolling_window"], min=1),
                        html.Br(),
                        dbc.Label("Y-axis Min:"),
                        dcc.Input(id="events-ymin-input", type="number", value=None),
                        html.Br(),
                        dbc.Label("Y-axis Max:"),
                        dcc.Input(id="events-ymax-input", type="number", value=None),
                    ]
                ),
                dbc.ModalFooter([dbc.Button("Close", id="close-events-modal", className="ml-auto")]),
            ],
            id="events-settings-modal",
            is_open=False,
        ),

        # ToF Histogram modal
        dbc.Modal(
            [
                dbc.ModalHeader("ToF Histogram Settings"),
                dbc.ModalBody(
                    [
                        dbc.Label("ToF Histogram Range (µs)"),
                        dcc.RangeSlider(
                            id="tof-hist-range-slider",
                            min=0.0,
                            max=200.0,  # Adjusted max based on tof_hist_max
                            step=0.1,
                            value=[default_settings["tof_hist_min"] * 1e6, default_settings["tof_hist_max"] * 1e6],
                            marks={i: str(i) for i in range(0, 201, 20)},
                        ),
                        html.Br(),
                        dbc.Label("Number of Bins"),
                        dcc.Slider(
                            id="tof-bins-slider",
                            min=1,
                            max=200,
                            step=5,
                            value=default_settings["tof_hist_nbins"],
                            marks={i: str(i) for i in range(5, 201, 25)},
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("Update Histogram Parameters", id="update-tof-histogram", className="ml-auto"),
                        dbc.Button("Close", id="close-tof-modal", className="ml-auto"),
                    ]
                ),
            ],
            id="tof-settings-modal",
            is_open=False,
        ),

    ],
    fluid=True,
)


# --------------------------------------------------------------------------------
# Callbacks for pop-up modals and offcanvas
# --------------------------------------------------------------------------------
@app.callback(
    Output("events-settings-modal", "is_open"),
    [Input("events-settings-button", "n_clicks"), Input("close-events-modal", "n_clicks")],
    [State("events-settings-modal", "is_open")],
)
def toggle_events_settings(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("tof-settings-modal", "is_open"),
    [Input("tof-settings-button", "n_clicks"), Input("close-tof-modal", "n_clicks")],
    [State("tof-settings-modal", "is_open")],
)
def toggle_tof_settings(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(Output("offcanvas", "is_open"), [Input("open-offcanvas", "n_clicks")], [State("offcanvas", "is_open")])
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


# --------------------------------------------------------------------------------
# Main Settings
# --------------------------------------------------------------------------------
@app.callback(Output("interval-component", "interval"), Input("refresh-rate", "value"))
def update_refresh_rate(refresh_rate):
    global REFRESH_RATE
    REFRESH_RATE = refresh_rate
    return int(refresh_rate * 1000)


@app.callback(
    [
        Output("nbatch-input", "value"),
        Output("total-max-points-input", "value"),
        Output("max-points-for-plot-input", "value"),
        Output("integration-window-input", "value"),
    ],
    [
        Input("nbatch-input", "value"),
        Input("total-max-points-input", "value"),
        Input("max-points-for-plot-input", "value"),
        Input("integration-window-input", "value"),
    ],
)
def update_settings(nbatch, total_max_points, max_points_for_plot, integration_window):
    global NBATCH, TOTAL_MAX_POINTS, MAX_POINTS_FOR_PLOT, default_settings
    NBATCH = nbatch
    TOTAL_MAX_POINTS = total_max_points
    MAX_POINTS_FOR_PLOT = max_points_for_plot
    default_settings["integration_window"] = integration_window
    viz_tool.integration_window = integration_window
    return nbatch, total_max_points, max_points_for_plot, integration_window


@app.callback(
    [Output("tof-hist-range-slider", "value"), Output("tof-bins-slider", "value")],
    Input("update-tof-histogram", "n_clicks"),
    State("tof-hist-range-slider", "value"),
    State("tof-bins-slider", "value"),
)
def update_tof_histogram_settings(n_clicks, tof_hist_range, tof_hist_nbins):
    global global_tof_min, global_tof_max
    if n_clicks:
        global_tof_min = tof_hist_range[0] * 1e-6
        global_tof_max = tof_hist_range[1] * 1e-6
        viz_tool.update_histogram_bins(global_tof_min, global_tof_max, tof_hist_nbins)
    return tof_hist_range, tof_hist_nbins


# --------------------------------------------------------------------------------
# MAIN CALLBACK: refresh all plots
# --------------------------------------------------------------------------------
@app.callback(
    [
        Output("events-over-time", "figure"),
        Output("tof-histogram", "figure"),
        Output("wavenumbers", "figure"),
        Output("voltage", "figure"),
        Output("summary-statistics", "children"),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("clear-data", "n_clicks"),
        Input("update-tof-histogram", "n_clicks"),
        Input("events-ymin-input", "value"),
        Input("events-ymax-input", "value"),
        Input("show-rolling-average-checkbox", "value"),
        Input("events-rolling-window-size", "value"),
    ],
)

def update_plots(
    n_intervals,
    clear_clicks,
    update_histogram_clicks,
    events_ymin,
    events_ymax,
    show_rolling_average_values,
    events_rolling_window_size,
):
    global viz_tool  # Important to let us reassign this reference
    ctx = dash.callback_context
    try:
        # If "Clear Data" is pressed or we exceed the limit -> reinitialize the entire PlotGenerator
        if (ctx.triggered and "clear-data" in ctx.triggered[0]["prop_id"]):
            viz_tool = PlotGenerator()  # Reset everything
            # Return empty figures + summary
            return (
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                [dbc.Col("No data available.", width=12)],
            )

        from fast_tagger_gui.src.system_utils import load_path
        file_location = load_path()["saving_file"]
        measurement_name = file_location.split("monitor_")[-1].split(".")[0]
    except ImportError:
        # Fallback if load_path is not available
        measurement_name = "default_measurement"  # Replace with your default measurement name
        minus_time_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        minus_time_str = datetime.strptime(measurement_name, "%Y_%m_%d_%H_%M_%S").strftime("%Y-%m-%dT%H:%M:%SZ")

    new_data = query_influxdb(minus_time_str, measurement_name)
    viz_tool.update_content(new_data)

    if new_data.empty and len(viz_tool.historical_data) == 0:
        return (
            go.Figure(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            [dbc.Col("No data available yet.", width=12)],
        )

    # Handle rolling average
    show_rolling_average = "show_rolling_average" in show_rolling_average_values if show_rolling_average_values else False
    if not events_rolling_window_size or events_rolling_window_size <= 0:
        events_rolling_window_size = default_settings["plot_rolling_window"]

    # Build all figures
    fig_events_over_time = viz_tool.plot_events_over_time(
        yaxis_range=[events_ymin, events_ymax] if (events_ymin is not None and events_ymax is not None) else None,
        show_rolling_average=show_rolling_average,
        rolling_window_size=events_rolling_window_size,
    )
    fig_tof_histogram = viz_tool.plot_tof_histogram()
    fig_wavenumbers = viz_tool.plot_wavenumbers(selected_channels=[1, 2, 3, 4])
    fig_voltage = viz_tool.plot_voltage()

    # Status checks
    status_text = "Status: Offline"
    status_style = {"color": "red"}
    last_time_event = None
    # events_only = pd.DataFrame([data for data in viz_tool.historical_data if data["channel"] != -1])
    if not viz_tool.historical_data.empty:
        last_time_event_val = viz_tool.historical_data["id_timestamp"].max()
        # Convert to float
        last_time_event = float(last_time_event_val) if not pd.isnull(last_time_event_val) else None
        print(last_time_event)
        if last_time_event and (time.time() - last_time_event) < 2:
            status_text = "Status: Online"
            status_style = {"color": "green"}

    # Summaries
    if viz_tool.historical_data.empty:
        summary_text = [dbc.Col("No data available.", width=12)]
    else:
        run_time = 0
        if not viz_tool.historical_data.empty:
            run_time = round(viz_tool.historical_data["id_timestamp"].max() - viz_tool.init_time, 2)
        if last_time_event:
            time_since_last = round(time.time() - last_time_event, 2)
        last_wn = 0
        last_voltage = 0
        if "wn_3" in viz_tool.historical_data.columns and viz_tool.historical_data["wn_3"].notna().any():
            last_wn = viz_tool.historical_data["wn_3"].dropna().iloc[-1]
        if "voltage" in viz_tool.historical_data.columns and viz_tool.historical_data["voltage"].notna().any():
            last_voltage = viz_tool.historical_data["voltage"].dropna().iloc[-1]

        summary_text = [
            dbc.Col(status_text, style=status_style, width=2),
            dbc.Col(f"Bunch Count: {viz_tool.historical_data['bunch'].values[-1]}", width=2),
            dbc.Col(f"Running Time: {run_time} s", width=2),
            dbc.Col(f"Time since last event: {time_since_last} s", width=2),
            dbc.Col(f"λ: {round(last_wn, 6)}", width=2),
            dbc.Col(f"Voltage: {round(last_voltage, 4)} V", width=2),
            dbc.Col(f"Bunching Rate: {viz_tool.trigger_rate:.2f} Hz", width=2),
        ]

    # Update uirevision to maintain state
    fig_events_over_time.update_layout(uirevision="events_over_time")
    fig_tof_histogram.update_layout(uirevision="tof_histogram")
    fig_wavenumbers.update_layout(uirevision="wavenumbers")
    fig_voltage.update_layout(uirevision="voltage")

    return (
        fig_events_over_time,
        fig_tof_histogram,
        fig_wavenumbers,
        fig_voltage,
        summary_text,
    )

if __name__ == "__main__":
    app.run_server(debug=True)
