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
from scipy.optimize import curve_fit

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
    "tof_min": 0,#1e-6,   # 1 microsecond
    "tof_max": 150e-6, # 150 microseconds
    "plot_rolling_window": 10,
    "integration_window": 10,
    "tof_num_gaussians": 1,  # Used if multi-peak is enabled
}
# Attempt to load user settings
with open(SETTINGS_PATH, 'r') as f:
    user_settings = json.load(f)
    print(user_settings)
# --------------------------------------------------------------------------------
# Global parameters and environment variables
# --------------------------------------------------------------------------------
global_tof_min = float(user_settings.get("tof_start", default_settings.get("tof_min")))
global_tof_max = float(user_settings.get("tof_end", default_settings.get("tof_max")))

default_settings["tof_min"] = global_tof_min
default_settings["tof_max"] = global_tof_max

db_token = os.getenv("INFLUXDB_TOKEN", "")
if not db_token:
    # Fallback, or read from somewhere else if needed
    db_token = "mTS9iQyj_ua5sAbJZ0ubOWQMQX_VcwhjhnZaikS1ImQuuY7xGtZDqelrugvBThfxpwYbuomtIyFMEkZdV657PA=="

os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"

NBATCH = 1_000
TOTAL_MAX_POINTS = 5_000_000
MAX_POINTS_FOR_PLOT = 100

REFRESH_RATE = 0.5  # seconds

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
        self.tof_min = settings_dict.get("tof_min", 0e-6)
        self.tof_max = settings_dict.get("tof_max", 150e-6)
        self.plot_rolling_window = settings_dict.get("plot_rolling_window", 10)
        self.integration_window = settings_dict.get("integration_window", 10)

        # For multi-peak fitting
        self.n_gaussians = settings_dict.get("tof_num_gaussians", 2)
        self.enable_multi_peak = True

        # Define maximum lengths for historical data to prevent memory bloat
        self.max_historical_length = TOTAL_MAX_POINTS

        self.historical_data = pd.DataFrame()
        self.unseen_new_data = pd.DataFrame()
        self.padded_historical_data = pd.DataFrame()

        self.last_loaded_time = None
        self.first_time = time.time()
        self.max_empty_counter = 10
        self.empty_counter = 0

        self.tof_mean = 0
        self.tof_var = 0
        self.tof_histogram_bins = np.linspace(self.tof_min, self.tof_max, self.tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)

        self.prev_tof_min = self.tof_min
        self.prev_tof_max = self.tof_max
        self.prev_tof_hist_nbins = self.tof_hist_nbins
        self.trigger_rate = 0  # Start with zero if no data.

    def update_histogram_bins(self, tof_min, tof_max, tof_hist_nbins):
        """Update bins only if user changes the slider."""
        self.tof_min = tof_min
        self.tof_max = tof_max
        self.tof_hist_nbins = tof_hist_nbins
        self.tof_histogram_bins = np.linspace(tof_min, tof_max, tof_hist_nbins + 1)
        self.histogram_counts = np.zeros(self.tof_hist_nbins)

    def update_num_gaussians(self, n_gaussians):
        """Update the number of Gaussians to fit (for multi-peak)."""
        self.n_gaussians = n_gaussians

    def set_multi_peak_enabled(self, enabled: bool):
        """Enable/disable multi-peak fitting."""
        self.enable_multi_peak = enabled

    def _update_tof_statistics(self, unseen_new_data):
        if unseen_new_data.empty:
            return

        events_offset = unseen_new_data["time_offset"].values
        if len(events_offset) > 0:
            new_hist_counts, _ = np.histogram(events_offset, bins=self.tof_histogram_bins)
            self.histogram_counts += new_hist_counts
            bin_centers = 0.5 * (self.tof_histogram_bins[:-1] + self.tof_histogram_bins[1:])
            if np.sum(self.histogram_counts) > 0:
                self.tof_mean = np.average(bin_centers, weights=self.histogram_counts)
                self.tof_var = np.average((bin_centers - self.tof_mean) ** 2, weights=self.histogram_counts)
            else:
                self.tof_mean = 0
                self.tof_var = 0

    def update_content(self, new_data: pd.DataFrame):
        """Append new data to historical, filter by time_offset, etc."""
        if not self.historical_data.empty:
            unseen_new_data = new_data[~new_data["id_timestamp"].isin(self.historical_data["id_timestamp"])]
        else:
            unseen_new_data = new_data

        # Filter time_offset by global tof settings
        if not unseen_new_data.empty:
            unseen_new_data = unseen_new_data[
                ((new_data["time_offset"] >= global_tof_min) & (new_data["time_offset"] <= global_tof_max))
            ]
        self.unseen_new_data = unseen_new_data
        # print("Got new data", len(self.unseen_new_data), "events")

        # Update trigger_rate if available
        if not unseen_new_data.empty and "trigger_rate" in unseen_new_data.columns:
            last_trigger_rate_series = unseen_new_data[unseen_new_data["trigger_rate"] != 0]["trigger_rate"]
            if not last_trigger_rate_series.empty:
                self.trigger_rate = last_trigger_rate_series.iloc[-1]

        # Append to historical
        self.historical_data = pd.concat([self.historical_data, unseen_new_data]).tail(self.max_historical_length)

        # Padded data for plots
        if unseen_new_data.empty and not self.padded_historical_data.empty:
            # Insert dummy if no new points to keep times updating
            dummy_data = pd.DataFrame(
                {
                    "bunch": [self.padded_historical_data["bunch"].values[-1] + 1 if not self.historical_data.empty else 0],
                    "n_events": [0],
                    "time_offset": [0],
                    "id_timestamp": [self.padded_historical_data["id_timestamp"].values[-1] + 0.5],
                    "trigger_rate": [self.trigger_rate],
                }
            )
            self.padded_historical_data = pd.concat([self.padded_historical_data, dummy_data]).tail(2_000)
        else:
            self.padded_historical_data = pd.concat([self.padded_historical_data, unseen_new_data.drop_duplicates(subset=["bunch"])]).tail(2_000)

        self._update_tof_statistics(unseen_new_data)
        self.historical_data = self.historical_data.drop_duplicates(subset=["id_timestamp"])

        self.last_loaded_time = (
            self.historical_data["id_timestamp"].max() if not self.historical_data.empty else None
        )

    def plot_events_over_time(
        self, max_points=100, yaxis_range=None,
        show_rolling_average=True, rolling_window_size=10
    ):
        """Plot events/s vs time, optionally with rolling average."""
        try:
            fig = go.Figure()

            df = self.padded_historical_data.copy()
            if df.empty:
                return fig

            df["id_timestamp"] = pd.to_datetime(df["id_timestamp"], unit="s")
            df.set_index("id_timestamp", inplace=True)

            events_per_second = df["n_events"].resample("1S").sum()
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

            if show_rolling_average and rolling_window_size > 1 and len(events_per_second) >= rolling_window_size:
                rolling_avg = np.convolve(nevents, np.ones(rolling_window_size) / rolling_window_size, mode="valid")
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

    # --- Gaussian definitions ---
    @staticmethod
    def _single_gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def _multi_gaussian(self, x, *params):
        """
        Sum of N Gaussians:
        params = [A1, mu1, sigma1, A2, mu2, sigma2, ..., AN, muN, sigmaN]
        """
        n = self.n_gaussians
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            A = params[3*i + 0]
            mu = params[3*i + 1]
            sigma = params[3*i + 2]
            y += PlotGenerator._single_gaussian(x, A, mu, sigma)
        return y

    def plot_tof_histogram(self):
        """
        Plot the histogram of time-of-flight. 
        - If enable_multi_peak is False, we do single Gaussian fit. 
        - If enable_multi_peak is True, we do multi-peak with n_gaussians.
        - If fit fails, we skip showing the fits in that iteration.
        - y-axis is forced to start at 0.
        """
        fig = go.Figure()
        if len(self.historical_data) == 0 or np.sum(self.histogram_counts) == 0:
            # Just return empty if no data
            fig.update_yaxes(range=[0, None])
            return fig

        bin_edges = self.tof_histogram_bins
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        time_total = time.time() - self.init_time
        counts_raw = self.histogram_counts.copy()
        counts_display = counts_raw / (time_total if time_total > 0 else 1)

        # Basic bar for the histogram
        fig = px.bar(
            x=bin_centers * 1e6,
            y=counts_display,
            labels={"x": "ToF (µs)", "y": "Count Rate (counts/s)"},
        )

        fit_failed = False  # track if fit fails

        # Try fitting if there's enough data
        if np.any(counts_raw > 0):
            try:
                # Single peak if multi-peak is disabled
                if not self.enable_multi_peak:
                    # Single peak: p0 = [peak_height, mid, sigma_guess]
                    peak_height_guess = max(counts_raw)
                    x_min, x_max = bin_centers[0], bin_centers[-1]
                    x_center_guess = bin_centers[np.argmax(counts_raw)]
                    sigma_guess = (x_max - x_min) / 10.0

                    p0 = [peak_height_guess, x_center_guess, sigma_guess]
                    popt, pcov = curve_fit(
                        self._single_gaussian, bin_centers, counts_raw, p0=p0, maxfev=5000
                    )

                    # Plot the resulting single-peak
                    x_fit = np.linspace(x_min, x_max, 1000)
                    y_fit_raw = self._single_gaussian(x_fit, *popt)
                    y_fit_display = y_fit_raw / (time_total if time_total > 0 else 1)

                    # sum_of_fit_color:
                    sum_color = "black"
                    A, mu, sigma = popt
                    fwhm = 2.354820045 * sigma
                    fig.add_trace(
                        go.Scatter(
                            x=x_fit * 1e6,
                            y=y_fit_display,
                            mode="lines",
                            line=dict(color=sum_color, width=2),
                            name=(
                                f"Single Peak Fit: {mu*1e6:.1f} ± {sigma*1e6:.1f} µs, "
                                f"FWHM={fwhm*1e6:.2f}µs"
                            ),
                        )
                    )

                else:
                    # Multi-peak fit with n_gaussians
                    n = self.n_gaussians
                    x_min, x_max = bin_centers[0], bin_centers[-1]
                    range_span = (x_max - x_min)
                    guess_amplitude = max(counts_raw) / n if n > 0 else 1

                    # Build initial guesses
                    p0 = []
                    for i in range(n):
                        fraction = (i + 0.5) / n
                        mu_guess = x_min + fraction * range_span
                        sigma_guess = range_span / (10.0 * n)
                        p0 += [guess_amplitude, mu_guess, sigma_guess]

                    popt, pcov = curve_fit(
                        self._multi_gaussian,
                        bin_centers,
                        counts_raw,
                        p0=p0,
                        maxfev=5000
                    )

                    # Evaluate total fit
                    x_fit = np.linspace(x_min, x_max, 1000)
                    y_fit_raw = self._multi_gaussian(x_fit, *popt)
                    y_fit_display = y_fit_raw / (time_total if time_total > 0 else 1)

                    # sum-of-fit line
                    sum_color = "black"
                    fig.add_trace(
                        go.Scatter(
                            x=x_fit * 1e6,
                            y=y_fit_display,
                            mode="lines",
                            name="Sum of Fit",
                            line=dict(color=sum_color, width=2),
                        )
                    )

                    # Up to 5 different colors for the peaks, in a fixed order
                    component_colors = ["red", "green", "magenta", "orange", "purple"]

                    # Plot each individual Gaussian
                    for i in range(n):
                        A = popt[3*i + 0]
                        mu = popt[3*i + 1]
                        sigma = popt[3*i + 2]
                        fwhm = 2.354820045 * sigma

                        # Build single component
                        y_single_raw = self._single_gaussian(x_fit, A, mu, sigma)
                        y_single_display = y_single_raw / (time_total if time_total > 0 else 1)
                        
                        # Replace the negative values with zeros
                        y_single_display = np.where(y_single_display < 0, 0, y_single_display)

                        color_i = component_colors[i % len(component_colors)]
                        fig.add_trace(
                            go.Scatter(
                                x=x_fit * 1e6,
                                y=y_single_display,
                                mode="lines",
                                line=dict(color=color_i, dash="dot"),
                                name=(
                                    f"Peak {i+1}: {mu*1e6:.1f} ± {sigma*1e6:.1f} µs, "
                                    f"FWHM={fwhm*1e6:.1f}µs"
                                ),
                            )
                        )

            except Exception as fit_e:
                print(f"Fit failed this iteration: {fit_e}")
                fit_failed = True

        # Force y-axis to start at zero (no negative values)
        fig.update_layout(
            xaxis_title="Time of Flight (µs)",
            yaxis_title="Total Events/s",
            uirevision="tof_histogram",
            template="plotly_white",
        )
        fig.update_yaxes(range=[0, None])

        return fig

    def plot_wavenumbers(self, selected_channels=[1, 2, 3, 4], max_points=200):
        """Plot wavenumbers for the given channels."""
        try:
            fig = go.Figure()
            if len(self.historical_data) == 0:
                return fig

            df = self.padded_historical_data.copy()
            if df.empty or "_time" not in df.columns:
                return fig

            df["_time"] = pd.to_datetime(df["_time"])  # treat _time as real datetimes
            df.set_index("_time", inplace=True)

            decimated_df = df.iloc[-max_points:].copy()
            colors = ["blue", "red", "green", "purple"]
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
            fig.add_trace(go.Scatter(
                x=decimated_df.index,
                y=decimated_df["spectr_peak"].astype("float"),
                mode="lines",
                name=f"Spectrometer",
            ))
                
                
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
        """Plot the voltage over time."""
        try:
            fig = go.Figure()
            if len(self.padded_historical_data) == 0:
                return fig

            df = self.padded_historical_data.sort_values("_time")
            df["_time"] = pd.to_datetime(df["_time"])
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
    |> filter(fn: (r) => r._measurement == "tagger")
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
        return df
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=[
            "_time", "bunch", "n_events", "channel", "time_offset", "id_timestamp",
            "wn_1", "wn_2", "wn_3", "wn_4", "voltage", "spectr_peak", "trigger_rate", "event_rate"
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
                dbc.NavItem(
                    dbc.Button("Export Data", id="export-data", n_clicks=0, color="secondary", className="ml-2")
                ),
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

        # A hidden div to show export status
        html.Div(id="export-status", style={"margin": "10px 0", "fontWeight": "bold"}),

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
                                max=1.0,
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
                        dbc.Col(
                            dcc.Input(
                                id="integration-window-input", 
                                type="number", 
                                value=default_settings["integration_window"], 
                                step=1
                            ), 
                            width=8
                        ),
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
                        dcc.Input(
                            id="events-rolling-window-size", 
                            type="number",
                            value=default_settings["plot_rolling_window"], 
                            min=1
                        ),
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
                            max=200.0,  # Adjusted max or your preference
                            step=0.1,
                            value=[
                                default_settings["tof_min"] * 1e6, 
                                default_settings["tof_max"] * 1e6,
                            ],
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
                        html.Br(),
                        dbc.Label("Enable multi-peak fit?"),
                        dbc.Checklist(
                            options=[{"label": "Enable multi-peak fit", "value": "enable_multi"}],
                            value=[],
                            id="tof-multi-peak-checkbox",
                            inline=True,
                        ),
                        html.Br(),
                        dbc.Label("Number of Gaussians to Fit:"),
                        dcc.Slider(
                            id="tof-num-gaussians-slider",
                            min=1,
                            max=5,
                            step=1,
                            value=default_settings["tof_num_gaussians"],  # default 1
                            marks={i: str(i) for i in range(1, 6)},
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
    [
        Output("tof-hist-range-slider", "value"), 
        Output("tof-bins-slider", "value"),
        Output("tof-num-gaussians-slider", "value"),
        Output("tof-multi-peak-checkbox", "value"),
    ],
    Input("update-tof-histogram", "n_clicks"),
    State("tof-hist-range-slider", "value"),
    State("tof-bins-slider", "value"),
    State("tof-num-gaussians-slider", "value"),
    State("tof-multi-peak-checkbox", "value"),
)
def update_tof_histogram_settings(n_clicks, tof_hist_range, tof_hist_nbins, tof_num_gaussians, multi_checkbox_values):
    global global_tof_min, global_tof_max
    if n_clicks:
        global_tof_min = tof_hist_range[0] * 1e-6
        global_tof_max = tof_hist_range[1] * 1e-6
        viz_tool.update_histogram_bins(global_tof_min, global_tof_max, tof_hist_nbins)
        viz_tool.update_num_gaussians(tof_num_gaussians)

        # Enable/disable multi-peak based on the checklist
        enable_multi = ("enable_multi" in multi_checkbox_values) if multi_checkbox_values else False
        viz_tool.set_multi_peak_enabled(enable_multi)

    return tof_hist_range, tof_hist_nbins, tof_num_gaussians, multi_checkbox_values

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
    global viz_tool
    ctx = dash.callback_context
    try:
        # If "Clear Data" is pressed -> reinitialize
        if (ctx.triggered and "clear-data" in ctx.triggered[0]["prop_id"]):
            viz_tool.__init__()
            return (
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                [dbc.Col("No data available.", width=12)],
            )

        # Attempt to read measurement_name from fast_tagger_gui
        from fast_tagger_gui.src.system_utils import load_path
        file_location = load_path()["saving_file"]
        measurement_name = file_location.split("monitor_")[-1].split(".")[0]
    except ImportError:
        # Fallback if load_path is not available
        measurement_name = "default_measurement"
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

    # Rolling average
    show_rolling_average = ("show_rolling_average" in show_rolling_average_values) if show_rolling_average_values else True
    if not events_rolling_window_size or events_rolling_window_size <= 0:
        events_rolling_window_size = default_settings["plot_rolling_window"]

    # Build figures
    fig_events_over_time = viz_tool.plot_events_over_time(
        yaxis_range=[events_ymin, events_ymax] 
        if (events_ymin is not None and events_ymax is not None) 
        else None,
        show_rolling_average=show_rolling_average,
        rolling_window_size=events_rolling_window_size,
    )
    fig_tof_histogram = viz_tool.plot_tof_histogram()
    fig_wavenumbers = viz_tool.plot_wavenumbers(selected_channels=[1, 2, 3, 4])
    fig_voltage = viz_tool.plot_voltage()

    # Status & Summaries
    status_text = "Status: Offline"
    status_style = {"color": "red"}
    last_time_event = None

    if not viz_tool.historical_data.empty:
        last_time_event_val = viz_tool.historical_data["id_timestamp"].max()
        if not pd.isnull(last_time_event_val):
            last_time_event = float(last_time_event_val)
        # If the last event happened <2s ago, call it "Online"
        if last_time_event and (time.time() - last_time_event) < 2:
            status_text = "Status: Online"
            status_style = {"color": "green"}

    if viz_tool.historical_data.empty:
        summary_text = [dbc.Col("No data available.", width=12)]
    else:
        run_time = 0
        if not viz_tool.historical_data.empty:
            run_time = round(
                viz_tool.historical_data["id_timestamp"].max() 
                - viz_tool.historical_data["id_timestamp"].min(), 2
            )
        time_since_last = round(time.time() - last_time_event, 2) if last_time_event else None
        last_wn = 0
        last_voltage = 0
        if "wn_3" in viz_tool.historical_data.columns and viz_tool.historical_data["wn_3"].notna().any():
            last_wn = viz_tool.historical_data["wn_3"].dropna().iloc[-1]
        if "voltage" in viz_tool.historical_data.columns and viz_tool.historical_data["voltage"].notna().any():
            last_voltage = viz_tool.historical_data["voltage"].dropna().iloc[-1]

        summary_text = [
            dbc.Col(status_text, style=status_style, width=2),
            dbc.Col(f"Bunch Count: {viz_tool.historical_data['bunch'].values[-1]}", width=2),
            dbc.Col(f"Events displayed: {len(viz_tool.historical_data)}",  width=2),
            dbc.Col(f"Running Time: {run_time} s", width=2),
            dbc.Col(f"Time since last event: {time_since_last} s", width=2),
            dbc.Col(f"λ: {round(last_wn, 6)}", width=2),
            dbc.Col(f"Voltage: {round(last_voltage, 4)} V", width=2),
            dbc.Col(f"Bunching Rate: {viz_tool.trigger_rate:.2f} Hz", width=2),
            dbc.Col(f"Event Rate: {viz_tool.historical_data['event_rate'].values[-1]:.2f} Hz", width=2),
            dbc.Col(f"Spectrum Peak: {float(viz_tool.historical_data['spectr_peak'].values[-1]):.2f} nm", width=2),
        ]

    # Preserve state
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

@app.callback(
    Output("export-status", "children"),
    Input("export-data", "n_clicks"),
    prevent_initial_call=True
)
def export_data(n_clicks):
    """
    When the user clicks the "Export Data" button, save the 
    current viz_tool.historical_data DataFrame to a CSV on the Desktop.
    """
    if n_clicks:
        try:
            desktop_path = r"C:\Users\EMALAB\Desktop"
            timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"historical_data_{timestamp_str}.csv"
            full_path = os.path.join(desktop_path, filename)

            if not viz_tool.historical_data.empty:
                viz_tool.historical_data.to_csv(full_path, index=False)
                return f"Data exported to: {full_path}"
            else:
                return "No data to export."
        except Exception as e:
            return f"Error exporting data: {str(e)}"
    return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=False)
