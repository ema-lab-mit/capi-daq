# run_scan.py
import sys
import os
import time
import queue
import argparse
import threading
from datetime import datetime, timedelta
import json
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from fastparquet import write, ParquetFile
import pandas as pd
from epics import PV
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

# os.system("influxd")

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
TRIGGER_LEVEL = modified_settings.get("trigger_level", -0.5)
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
    "refresh_rate": 5,
}

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def get_wnum(i=1):
    try:
        return round(float(wavenumbers_pvs[i - 1].get()), 5)
    except Exception as e:
        print(f"Error getting wavenumber: {e}")
        return 0.00000
def get_all_wavenumbers():
    return [get_wnum(i) for i in range(1, 5)]
    
def write_to_influxdb(data, data_name, wns):
    points = []
    for d in data:
        dt = datetime.fromtimestamp(d[4])
        points.append(Point("tagger_data").tag("type", data_name).field("bunch", d[0]).time(dt, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("n_events", d[1]).time(dt, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("channel", d[2]).time(dt, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("time_offset", float(d[3])).time(dt, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("timestamp", d[4]).time(dt, WritePrecision.NS))
        points += [Point("tagger_data").tag("type", data_name).field(f"wn_{i}", wns[i-1]).time(dt, WritePrecision.NS) for i in [1, 2, 3, 4]]
    write_api.write(bucket=INFLUXDB_BUCKET, record=points)

def write_to_parquet_direct(data, wnums, saving_path):
    schema = {
        'bunch': 'int64',
        'n_events': 'int64',
        'channels': 'int64',
        'time_offset': 'float64',
        'timestamp': 'float64',
        'wn1': 'float64',
        'wn2': 'float64',
        'wn3': 'float64',
        'wn4': 'float64',
    }
    df = pd.DataFrame(data, columns=['bunch', 'n_events', 'channels', 'time_offset', 'timestamp'])
    df['wn1'] = wnums[0]
    df['wn2'] = wnums[1]
    df['wn3'] = wnums[2]
    df['wn4'] = wnums[3]
    
    if os.path.exists(saving_path):
        pf = ParquetFile(saving_path)
        write(saving_path, df, file_scheme='hive', append=True)
    else:
        write(saving_path, df, file_scheme='hive', compression='SNAPPY', write_index=False)

def process_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_rate", type=float, default=0.5)
    parser.add_argument("--is_scanning", type=bool, default=False)
    args = parser.parse_args()
    return args.refresh_rate, args.is_scanning

def create_saving_path(folder_location, saving_format):
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    identifier = str(time_now).replace(":", "-").replace(" ", "_").replace("-", "_")
    if not os.path.exists(folder_location):
        os.makedirs(folder_location, exist_ok=True)
    name = "scan_" + identifier + "." + saving_format
    return os.path.join(folder_location, name)

def main_loop(tagger, data_name, saving_file_path=SAVING_FILE):
    tagger.set_trigger_falling()
    tagger.set_trigger_level(float(TRIGGER_LEVEL))
    tagger.start_reading()
    wavenumbers_pv_names = ["LaserLab:wavenumber_1", "LaserLab:wavenumber_2", "LaserLab:wavenumber_3", "LaserLab:wavenumber_4"]
    
    global wavenumbers_pvs
    wavenumbers_pvs = [PV(name) for name in wavenumbers_pv_names]
    
    print("Connecting to devices")
    starting_time = datetime.now()
    batch_data = []
    while True:
        data = tagger.get_data()
        if data:
            batch_data += data
            wns = get_all_wavenumbers()
            write_to_influxdb(data, data_name, wns)
            if datetime.now() - starting_time > timedelta(seconds=initialization_params["refresh_rate"]):
                write_to_parquet_direct(batch_data, wns, saving_file_path)
                batch_data = []
                starting_time = datetime.now()

if __name__ == "__main__":
    refresh_rate, is_scanning = process_input_args()
    initialization_params["refresh_rate"] = refresh_rate
    folder_location = load_path()["saving_folder"]
    save_path = create_saving_path(folder_location, SAVING_FORMAT)
    update_settings_file(save_path)
    metadata_writer(folder_location, save_path, initialization_params)
    initialization_params["save_path"] = save_path
    print(f"Saving data to {save_path}")
    tagger = Tagger(initialization_params=initialization_params)
    data_name = save_path.split("scan_")[1].split(".")[0]
    print("DATA NAME: ", data_name)
    main_loop(tagger, data_name, save_path)
