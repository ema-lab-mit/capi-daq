import sys
import os
import time
import pandas as pd
import queue
import argparse
import ntplib
import threading
from datetime import datetime, timedelta
import json
from influxdb_client import InfluxDBClient, Point, WritePrecision, SYNCHRONOUS
from epics import PV
import joblib


this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from fast_tagger_gui.src.physics_utils import time_to_flops, flops_to_time
from fast_tagger_gui.src.tag_interface import Tagger

N_CHANNELS = ["A", "B", "C", "D"]
SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
TOKEN_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\secrets.json"
MAX_DF_LENGTH_IN_MEMORY_IF_SCANNING = 1_000_000

# read the token file for the secret keys
def get_secrets():
    try:
        with open(TOKEN_PATH, 'r') as f:
            secrets = json.load(f)
        return secrets.get("tokens", "")
    except Exception as e:
        print(f"Error loading secrets: {e}")
        return ""

db_token = get_secrets()
os.environ["INFLUXDB_TOKEN"] = db_token

# InfluxDB configuration
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"

# Read the configuration file for the initialization parameters
def get_card_settings():
    try:
        with open(SETTINGS_PATH, 'r') as f:
            settings = json.load(f)
        return {
            "tof_start": float(settings.get("tof_start", "1e-6")),
            "tof_end": float(settings.get("tof_end", "20e-6")),
            "channel_level": float(settings.get("channel_level", "-0.5")),
            "trigger_level": float(settings.get("trigger_level", "-0.5")),
            "pv_name": settings.get("pv_name", "LaserLab:wavenumber_1"),
            "data_format": settings.get("data_format", "parquet"),
            "saving_file": settings.get("saving_file", "data.parquet"),
        }
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}

modified_settings = get_card_settings()
STOP_TIME_WINDOW = modified_settings.get("tof_end", 20e-6)
INIT_TIME = modified_settings.get("tof_start", 1e-6)
CHANNEL_LEVEL = modified_settings.get("channel_level", -0.5)
SAVING_FORMAT = modified_settings.get("data_format", "parquet")
SAVING_FILE = modified_settings.get("saving_file", "data.parquet")

initialization_params = {
    "trigger": {
        "channels": [True, True, True, True],
        "levels": [CHANNEL_LEVEL for _ in range(4)],
        "types": [False for _ in range(4)],
        "starts": [int(time_to_flops(INIT_TIME)) for _ in range(4)],
        "stops": [int(time_to_flops(STOP_TIME_WINDOW)) for _ in range(4)],
    },
    "refresh_rate": 0.2,
}
ntp_time_offset = 0.0
last_sync_time = None
time_sync_interval = 120  # Sync every 2 minutes
ntp_sync_queue = queue.Queue()

def sync_time():
    global ntp_time_offset, last_sync_time
    c = ntplib.NTPClient()
    try:
        response = c.request('pool.ntp.org', timeout=5)
        ntp_time = datetime.fromtimestamp(response.tx_time)
        local_time = datetime.now()
        ntp_time_offset = (ntp_time - local_time).total_seconds()
        last_sync_time = local_time
        ntp_sync_queue.put(("success", ntp_time_offset))
    except Exception as e:
        ntp_sync_queue.put(("error", str(e)))

def get_synced_time():
    return datetime.now() + timedelta(seconds=ntp_time_offset)

def time_sync_thread():
    while True:
        sync_time()
        time.sleep(time_sync_interval)


def write_to_influxdb(data, wnums):
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    points = []
    for d in data:
        point = Point("tagger_data") \
            .field("bunch", d[0]) \
            .field("n_events", d[1]) \
            .field("channels", d[2]) \
            .field("timestamp", d[3]) \
            .field("wavenumber_1", wnums[0]) \
            .field("wavenumber_2", wnums[1]) \
            .field("wavenumber_3", wnums[2]) \
            .field("wavenumber_4", wnums[3]) \
            .time(get_synced_time(), WritePrecision.NS)
        points.append(point)
    
    write_api.write(bucket=INFLUXDB_BUCKET, record=points)
    client.close()

def write_to_parquet(df, saving_path: str):
    if not df.empty:
        df.to_parquet(saving_path, index=False, compression='snappy')

def parquet_writer_thread(queue, saving_path):
    while True:
        df = queue.get()
        write_to_parquet(df, saving_path)
        queue.task_done()

def build_dataframe(data, wnums):
    df = pd.DataFrame(data=data, columns=["bunch", "n_events", "channels", "timestamp"])
    df['timestamp'] = df['timestamp'].apply(flops_to_time)
    df['wavenumber_1'] = wnums[:, 0]
    df['wavenumber_2'] = wnums[:, 1]
    df['wavenumber_3'] = wnums[:, 2]
    df['wavenumber_4'] = wnums[:, 3]
    return df

def get_wnum(i=0):
    try:
        return round(float(wavenumbers_pvs[i].get()), 5)
    except Exception as e:
        # print(f"Error getting wavenumber: {e}")
        return 0.0
    
def get_wavenumbers():
    # Get the wavenumbers from the PVs... but do it with multiprocessing
    return joblib.Parallel(n_jobs=4)(joblib.delayed(get_wnum)(i) for i in range(4))

def main_loop(tagger, is_scanning=False, saving_file_path=SAVING_FILE):
    tagger.set_trigger_falling()
    tagger.set_trigger_level(-0.5)
    tagger.start_reading()
    sync_thread = threading.Thread(target=time_sync_thread, daemon=True)
    sync_thread.start()
    parquet_queue = queue.Queue()
    parquet_thread = threading.Thread(target=parquet_writer_thread, args=(parquet_queue, saving_file_path), daemon=True)
    parquet_thread.start()
    wavenumbers_pv_names = ["LaserLab:wavenumber_1", "LaserLab:wavenumber_2", "LaserLab:wavenumber_3", "LaserLab:wavenumber_4"]
    
    global wavenumbers_pvs
    wavenumbers_pvs = [PV(name) for name in wavenumbers_pv_names]
    
    print("Connecting to devices")
    # Get the time (UNIX time)
    initial_time = datetime.now().timestamp()
    while True:
        try:
            sync_result, sync_data = ntp_sync_queue.get_nowait()
            if sync_result == "success":
                print(f"Time synced. Offset: {sync_data}")
            else:
                print(f"Time sync error: {sync_data}")
        except queue.Empty:
            pass
        data = tagger.get_data()
        if data is not None:
            current_wnum = get_wnum()
            df = build_dataframe(data, current_wnum)
            parquet_queue.put(df)
            write_to_influxdb(data, current_wnum)

        time.sleep(initialization_params["refresh_rate"])
        
def load_path():
    try:
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        return {
            "saving_folder": settings.get("saving_folder", ""),
            "saving_file": settings.get("saving_file", ""),
        }
    except Exception as e:
        print(f"Error loading settings: {e}")
        return ""


def update_settings_file(saving_file):
    try:
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        settings["saving_file"] = saving_file
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        print(f"Error updating settings file: {e}")


def metadata_writer(folder_location, 
                    data_complete_path: str, 
                    initialization_params: dict
                    ) -> None:
    """
    Writes some metadata to a
    """
    file_name = "metadata_tagger_monitor.csv"
    payload = {
        "refresh_rate": initialization_params["refresh_rate"],
        "stop_time_window": STOP_TIME_WINDOW,
        "init_time": INIT_TIME,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_complete_path": data_complete_path,
        "format": SAVING_FORMAT,
    }
    if os.path.exists(os.path.join(folder_location, file_name)):
        metadata_df = pd.read_csv(os.path.join(folder_location, file_name))
        metadata_df = metadata_df.append(payload, ignore_index=True)
    else:
        metadata_df = pd.DataFrame(payload, index=[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_rate", type=float, default=0.5)
    parser.add_argument("--is_scanning", type=bool, default=False)
    args = parser.parse_args()
    initialization_params["refresh_rate"] = args.refresh_rate
    folder_location = load_path()["saving_folder"]
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    identifier = str(time_now).replace(":", "-").replace(" ", "_").replace("-", "_")
    if not os.path.exists(folder_location):
        os.makedirs(folder_location, exist_ok=True)
    name = "scan_" + identifier + "." + SAVING_FORMAT
    save_path = os.path.join(folder_location, name)
    update_settings_file(save_path)
    metadata_writer(folder_location, save_path)
    initialization_params["save_path"] = save_path
    print(f"Saving data to {save_path}")
    tagger = Tagger(initialization_params = initialization_params)
    main_loop(tagger, args.is_scanning, save_path)
