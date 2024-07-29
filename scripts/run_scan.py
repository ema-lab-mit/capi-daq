import sys
import os
import time
import pandas as pd
import queue
import argparse
import threading
from datetime import datetime
import json
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import serial
# import pyarrow as pa
# import pyarrow.parquet as pq
import fastparquet

this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from fast_tagger_gui.src.physics_utils import time_to_flops
from fast_tagger_gui.src.tag_interface import Tagger
from fast_tagger_gui.src.system_utils import (
    get_secrets,
    load_path,
    update_settings_file, 
)
from fast_tagger_gui.src.devices.multimeter import VoltageReader, HP_Multimeter
from fast_tagger_gui.src.devices.wavemeter import WavenumberReader

SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
POSTING_BATCH_SIZE = 2
db_token = get_secrets()
os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"

data_queue = queue.Queue()
stop_event = threading.Event()

def get_card_settings(settings_path=SETTINGS_PATH):
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        return {
            "tof_start": float(settings.get("tof_start", "1e-6")),
            "tof_end": float(settings.get("tof_end", "20e-6")),
            "channel_level": float(settings.get("channel_level", "-0.5")),
            "trigger_level": float(settings.get("trigger_level", "-0.5")),
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
    "refresh_rate": 0.1,
}

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_to_influxdb(data, data_name, voltage, wavenumbers, timestamp):
    points = []
    for d in data:
        points.append(Point("tagger_data").tag("type", data_name).field("bunch", d[0]).time(timestamp, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("n_events", d[1]).time(timestamp, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("channel", d[2]).time(timestamp, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("time_offset", float(d[3])).time(timestamp, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("timestamp", d[4]).time(timestamp, WritePrecision.NS))
        points.append(Point("tagger_data").tag("type", data_name).field("voltage", voltage).time(timestamp, WritePrecision.NS))
        points += [Point("tagger_data").tag("type", data_name).field(f"wn_{i}", wavenumbers[i-1]).time(timestamp, WritePrecision.NS) for i in range(1, 5)]
    write_api.write(bucket=INFLUXDB_BUCKET, record=points)

def process_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_rate", type=float, default=0.5)
    parser.add_argument("--is_scanning", type=bool, default=False)
    parser.add_argument("--voltage_port", type=int, default=16)
    args = parser.parse_args()
    return args.refresh_rate, args.is_scanning, args.voltage_port

def create_saving_path(folder_location, saving_format, label="scan_"):
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    identifier = str(time_now).replace(":", "-").replace(" ", "_").replace("-", "_")
    if not os.path.exists(folder_location):
        os.makedirs(folder_location, exist_ok=True)
    name = label + identifier + "." + saving_format
    return os.path.join(folder_location, name)

# def write_to_file(saving_file):
#     while not stop_event.is_set() or not data_queue.empty():
#         try:
#             data_batch = data_queue.get(timeout=1)
#             df = pd.DataFrame(data_batch, columns=["bunch", "n_events", "channel", "time_offset", "timestamp", "voltage", "wn_1", "wn_2", "wn_3", "wn_4"])
#             table = pa.Table.from_pandas(df)
#             if not os.path.exists(saving_file):
#                 pq.write_table(table, saving_file)
#             else:
#                 with pq.ParquetWriter(saving_file, table.schema, compression='snappy') as writer:
#                     writer.write_table(table)
#         except queue.Empty:
#             continue

def write_to_file(saving_file):
    while not stop_event.is_set() or not data_queue.empty():
        try:
            data_batch = data_queue.get(timeout=1)
            df = pd.DataFrame(data_batch, columns=["bunch", "n_events", "channel", "time_offset", "timestamp", "voltage", "wn_1", "wn_2", "wn_3", "wn_4"])

            if os.path.exists(saving_file):
                fastparquet.write(saving_file, df, append=True)
            else: 
                df.to_parquet(saving_file)
        except queue.Empty:
            continue

def metadata_writer(folder_location, data_complete_path, initialization_params):
    file_name = "metadata_tagger_monitor.csv"
    payload = {
        "refresh_rate": initialization_params["refresh_rate"],
        "stop_time_window": initialization_params["trigger"]["stops"],
        "init_time": initialization_params["trigger"]["starts"],
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_complete_path": data_complete_path,
        "format": initialization_params["save_path"].split('.')[-1],
    }
    metadata_path = os.path.join(folder_location, file_name)
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        # concatenate
        metadata_df = pd.concat([metadata_df, pd.DataFrame([payload])])
        metadata_df.drop_duplicates(subset=["started_at"], inplace=True)
    else:
        metadata_df = pd.DataFrame([payload])
    metadata_df.to_csv(metadata_path, index=False)  # overwrite existing file with new data

def main_loop(tagger, data_name, voltage_reader, wavenumber_reader):
    tagger.set_trigger_falling()
    tagger.set_trigger_level(float(TRIGGER_LEVEL))
    tagger.start_reading()
    i = 0
    batched_data = []
    while not stop_event.is_set():
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        data = tagger.get_data()
        if data is not None:
            for d in data:
                voltage = voltage_reader.get_voltage()
                wavenumbers = wavenumber_reader.get_wavenumbers()
                batched_data.append([
                    d[0], d[1], d[2], float(d[3]), d[4],
                    voltage, wavenumbers[0], wavenumbers[1], wavenumbers[2], wavenumbers[3]
                ])
            i += 1
            if i % POSTING_BATCH_SIZE == 0:
                data_queue.put(batched_data)
                write_to_influxdb(batched_data, data_name, voltage, wavenumbers, timestamp)
                batched_data = []
    # Ensure all remaining data is processed before exiting
    if batched_data:
        data_queue.put(batched_data)

if __name__ == "__main__":
    refresh_rate, is_scanning, voltage_port = process_input_args()
    initialization_params["refresh_rate"] = refresh_rate
    folder_location = load_path()["saving_folder"]
    save_path = create_saving_path(folder_location, SAVING_FORMAT, label="scan_")
    update_settings_file(save_path)
    initialization_params["save_path"] = save_path
    tagger = Tagger(initialization_params=initialization_params)
    data_name = save_path.split("scan_")[1].split(".")[0]
    multimeter = HP_Multimeter("COM" + str(voltage_port))
    voltage_reader = VoltageReader(multimeter, refresh_rate=refresh_rate)
    wavenumber_reader = WavenumberReader(refresh_rate=refresh_rate)
    voltage_reader.start()
    wavenumber_reader.start()
    
    metadata_writer(folder_location, save_path, initialization_params)
    
    writer_thread = threading.Thread(target=write_to_file, args=(save_path,))
    writer_thread.start()

    try:
        main_loop(tagger, data_name, voltage_reader, wavenumber_reader)
    except KeyboardInterrupt:
        stop_event.set()
        voltage_reader.stop()
        wavenumber_reader.stop()
        writer_thread.join()
