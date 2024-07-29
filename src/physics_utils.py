import pandas as pd
import numpy as np

def convert_to_stoptime(t):
    # 30000 -> ~15 us
    convertion = 30_000 / (15e-6)  # p / s
    return convertion * t


def time_to_flops(t):
    quatization = 100e-12  # seconds / flops
    return t / quatization


def flops_to_time(ft):
    quatization = 100e-12  # seconds / flops
    return ft * quatization


def compute_tof_from_data(data: pd.DataFrame):
    latest_trigger_time = 0
    tofs = []
    for index, d in data.iterrows():
        is_trigger = d.channels == -1
        if is_trigger:
            latest_trigger_time = d.timestamp
        else:
            tof = d["timestamp"] - latest_trigger_time
            tofs.append(flops_to_time(tof))
    return np.array(tofs)


def event_rate_per_wavelength(data: pd.DataFrame, wavelength_binning=10):
    """
    Builds the histogram of the number of events per time (in seconds)
    versus the wavelength of the event.
    """
    # Now we need to compute the number of events
    # PASS
    return None


def transform_data_highly_optimized(read_data: pd.DataFrame):
    try:
        # Ensure unique indices for trigger data
        trigger_data = (
            read_data[read_data["channels"] == -1]
            .drop_duplicates("bunch")
            .set_index("bunch")["timestamp"]
        )

        # Create a dictionary to hold the results
        transformed = {
            "bunch": [],
            "n_events": [],
            "trigger_timestamp": [],
            "events_timestamps": [],
            "channels": [],
            "wavelengths": [],
            "tof": [],
        }

        # Loop through the grouped data to populate the transformed dictionary
        for bunch, group in read_data.groupby("bunch"):
            trigger_timestamp = trigger_data.get(bunch, None)
            events_timestamps = group["timestamp"].tolist()
            transformed["bunch"].append(bunch)
            transformed["n_events"].append(group["n_events"].iloc[0])
            transformed["trigger_timestamp"].append(trigger_timestamp)
            transformed["events_timestamps"].append(events_timestamps)
            transformed["channels"].append(group["channels"].tolist())
            transformed["wavelengths"].append(group["wavelength"].tolist())
            transformed["tof"].append(
                (group["timestamp"] - trigger_timestamp).tolist()
                if trigger_timestamp is not None
                else []
            )

        # Convert the transformed dictionary to a DataFrame
        return pd.DataFrame(transformed)
    except Exception as e:
        print(f"Error transforming data: {e}")
        return pd.DataFrame()
