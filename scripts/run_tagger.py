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
from influxdb_client import InfluxDBClient, Point, WritePrecision
from threading import Lock, Thread
from influxdb_client.client.write_api import SYNCHRONOUS
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from fast_tagger_gui.src.physics_utils import time_to_flops, flops_to_time
from fast_tagger_gui.src.tag_interface import Tagger
from fast_tagger_gui.src.system_utils import (
    get_secrets,
    load_path,
    update_settings_file, 
    metadata_writer,
)

SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
MAX_DF_LENGTH_IN_MEMORY_IF_SCANNING = 1_000_000
db_token = get_secrets()
os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"

def get_card_settings(settings_path=SETTINGS_PATH):
    try:
        with open(settings_path, 'r') as f:
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
    "refresh_rate": 0.1,
}

        
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_to_influxdb(data, data_name):
    points = []
    for d in data:
        dt = datetime.now()
        bunch = int(d[0])
        nevents = int(d[1])
        channels = float(d[2])
        timestamp = float(d[3])
        point_bunch = Point("tagger_data").tag("type", data_name).field("bunch", bunch).time(dt, WritePrecision.NS)
        points.append(point_bunch)
        point_n_events = Point("tagger_data").tag("type", data_name).field("n_events", nevents).time(dt, WritePrecision.NS)
        points.append(point_n_events)
        point_channels = Point("tagger_data").tag("type", data_name).field("channels", channels).time(dt, WritePrecision.NS)
        points.append(point_channels)
        point_timestamp = Point("tagger_data").tag("type", data_name).field("timestamp", timestamp).time(dt, WritePrecision.NS)
        points.append(point_timestamp)
    write_api.write(bucket=INFLUXDB_BUCKET, record=points)

def write_to_parquet(df, saving_path: str):
    if not df.empty:
        df.to_parquet(saving_path, index=False, compression='snappy')

def parquet_writer_thread(queue, saving_path):
    while True:
        df = queue.get()
        write_to_parquet(df, saving_path)
        queue.task_done()

def build_dataframe(data):
    df = pd.DataFrame(data=data, columns=["bunch", "n_events", "channels", "timestamp"])
    df['timestamp'] = df['timestamp'].apply(flops_to_time)
    return df

def process_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_rate", type=float, default=0.5)
    parser.add_argument("--is_scanning", type=bool, default=False)
    args = parser.parse_args()
    return args.refresh_rate, args.is_scanning

def create_saving_path(folder_location, saving_format, label = "scan_"):
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    identifier = str(time_now).replace(":", "-").replace(" ", "_").replace("-", "_")
    if not os.path.exists(folder_location):
        os.makedirs(folder_location, exist_ok=True)
    name = label + identifier + "." + saving_format
    return os.path.join(folder_location, name)


def main_loop(tagger, data_name):
    tagger.set_trigger_falling()
    tagger.set_trigger_level(-0.4)
    tagger.start_reading()
    while True:
        data = tagger.get_data()
        if data is not None:
            write_to_influxdb(data, data_name)

if __name__ == "__main__":
    refresh_rate, is_scanning = process_input_args()
    initialization_params["refresh_rate"] = refresh_rate
    folder_location = load_path()["saving_folder"]
    save_path = create_saving_path(folder_location, SAVING_FORMAT, label="monitor_")
    update_settings_file(save_path)
    initialization_params["save_path"] = save_path
    tagger = Tagger(initialization_params=initialization_params)
    data_name = save_path.split("monitor_")[1].split(".")[0]
    main_loop(tagger, data_name)