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
import pyarrow as pa
import pyarrow.parquet as pq
import logging

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
POSTING_BATCH_SIZE = 100
db_token = get_secrets()
os.environ["INFLUXDB_TOKEN"] = db_token
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = db_token
INFLUXDB_ORG = "EMAMIT"
INFLUXDB_BUCKET = "DAQ"

# Initialize a bounded queue to prevent memory issues
data_queue = queue.Queue(maxsize=10000)  # Adjust based on memory constraints
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

# Initialize InfluxDB Client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_to_influxdb(batch_data, measurement_name):
    """
    Writes each row in batch_data to InfluxDB with the correct time & fields.
    batch_data is a list of lists with columns:
      [bunch, n_events, channel, time_offset, timestamp_str, voltage, wn_1, wn_2, wn_3, wn_4]
    """
    points = []
    for row in batch_data:
        # row indices:
        #  0 -> bunch
        #  1 -> n_events
        #  2 -> channel
        #  3 -> time_offset
        #  4 -> string timestamp, e.g. 2025-01-17T14:33:42.123456Z
        #  5 -> voltage
        #  6,7,8,9 -> wavenumbers
        try:
            # Convert the string timestamp to Python datetime
            data_ingestion = datetime.strptime(row[4], "%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
            # Fallback to now if there's any parsing error
            data_ingestion = datetime.utcnow()

        # Build a single point or multiple fields in one point
        p = (Point("tagger_data")
             .tag("type", measurement_name)
             .field("bunch", row[0])
             .field("n_events", row[1])
             .field("channel", row[2])
             .field("time_offset", float(row[3]))
             .field("voltage", float(row[5]))
             .field("wn_1", float(row[6]))
             .field("wn_2", float(row[7]))
             .field("wn_3", float(row[8]))
             .field("wn_4", float(row[9]))
             # The "timestamp" string can also be stored as a field if desired:
             .field("timestamp_str", row[4])
             .time(data_ingestion, WritePrecision.NS))
        points.append(p)

    if len(points) > 0:
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

def write_to_file(saving_file):
    """
    Writes the queued data to Parquet. We remove CSV-based metadata saving
    and do not touch SQLite. Just Parquet + Influx is used.
    """
    schema = pa.schema([
        pa.field("bunch", pa.int64(), nullable=True),
        pa.field("n_events", pa.int64(), nullable=True),
        pa.field("channel", pa.int64(), nullable=True),
        pa.field("time_offset", pa.float64(), nullable=True),
        pa.field("timestamp", pa.string(), nullable=True),
        pa.field("voltage", pa.float64(), nullable=True),
        pa.field("wn_1", pa.float64(), nullable=True),
        pa.field("wn_2", pa.float64(), nullable=True),
        pa.field("wn_3", pa.float64(), nullable=True),
        pa.field("wn_4", pa.float64(), nullable=True),
    ])

    try:
        # Open file in binary write mode
        file = open(saving_file, 'wb')
        writer = pq.ParquetWriter(file, schema)
        print(f"Initialized ParquetWriter for {saving_file}")
    except Exception as e:
        print(f"Error initializing ParquetWriter: {e}")
        return

    while not stop_event.is_set() or not data_queue.empty():
        try:
            data_batch = data_queue.get(timeout=1)
            df = pd.DataFrame(data_batch, columns=[
                "bunch", "n_events", "channel", "time_offset",
                "timestamp", "voltage", "wn_1", "wn_2", "wn_3", "wn_4"
            ])

            # Enforce data types
            df = df.astype({
                "bunch": 'Int64',
                "n_events": 'Int64',
                "channel": 'Int64',
                "time_offset": 'float64',
                "timestamp": 'string',
                "voltage": 'float64',
                "wn_1": 'float64',
                "wn_2": 'float64',
                "wn_3": 'float64',
                "wn_4": 'float64'
            })

            # Convert DataFrame to PyArrow Table
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

            # Write the table as a new row group
            writer.write_table(table)

            file.flush()
            os.fsync(file.fileno())

            print(f"Wrote batch of size {len(data_batch)} to {saving_file}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error writing to Parquet file: {e}")

    # Final flush and close
    try:
        writer.close()
        file.close()
        print(f"Closed ParquetWriter for {saving_file}")
    except Exception as e:
        print(f"Error closing ParquetWriter: {e}")

def main_loop(tagger, measurement_name, voltage_reader, wavenumber_reader):
    """
    Read data from the Tagger, grab voltage and wavenumbers, and send them
    both to Parquet (via the queue) and to InfluxDB. 
    """
    tagger.set_trigger_falling()
    tagger.set_trigger_level(float(TRIGGER_LEVEL))
    tagger.start_reading()
    i = 0
    batched_data = []

    while not stop_event.is_set():
        # We'll store the string-based timestamp for each event
        # But we only generate a single "batch" timestamp per iteration
        now_str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        data = tagger.get_data()

        if data is not None:
            # We retrieve voltage and wavenumbers once per cycle
            # (Alternatively, you could measure them once per event, 
            #  but that might slow things down.)
            voltage = voltage_reader.get_voltage()
            wavenumbers = wavenumber_reader.get_wavenumbers()

            for d in data:
                # d is [bunch, n_events, channel, time_offset, raw_tagger_timestamp]
                # We'll keep the fifth field (tagger's raw timestamp) 
                # or we can ignore it. We'll store our 'now_str' as official.
                batched_data.append([
                    d[0],       # bunch
                    d[1],       # n_events
                    d[2],       # channel
                    float(d[3]),# time_offset
                    now_str,    # string timestamp (UTC)
                    voltage,
                    wavenumbers[0],
                    wavenumbers[1],
                    wavenumbers[2],
                    wavenumbers[3]
                ])

            i += 1
            if i % POSTING_BATCH_SIZE == 0:
                try:
                    # Put the entire batch into the queue for Parquet
                    data_queue.put(batched_data, timeout=1)
                    # Write the same batch to Influx
                    write_to_influxdb(batched_data, measurement_name)
                    batched_data = []
                except queue.Full:
                    print("Data queue is full. Dropping data or handle overflow.")
                if i % 100 == 0:
                    print(f"Processed {i} cycles. Queue size: {data_queue.qsize()}")

    # Final flush before exit
    if batched_data:
        try:
            data_queue.put(batched_data, timeout=1)
            write_to_influxdb(batched_data, measurement_name)
        except queue.Full:
            print("Data queue is full on final data put.")

if __name__ == "__main__":
    refresh_rate, is_scanning, voltage_port = process_input_args()
    initialization_params["refresh_rate"] = refresh_rate

    # Load the path from JSON
    folder_location = load_path()["saving_folder"]
    save_path = create_saving_path(folder_location, SAVING_FORMAT, label="scan_")
    update_settings_file(save_path)
    initialization_params["save_path"] = save_path

    tagger = Tagger(initialization_params=initialization_params)
    # Extract a unique measurement name from the newly created file
    # e.g. scan_2025_01_17_14_33_42.parquet => measurement_name=2025_01_17_14_33_42
    measurement_name = os.path.basename(save_path).split("scan_")[1].split(".")[0]

    multimeter = HP_Multimeter("COM" + str(voltage_port))
    voltage_reader = VoltageReader(multimeter, refresh_rate=refresh_rate)
    wavenumber_reader = WavenumberReader(refresh_rate=refresh_rate)
    voltage_reader.start()
    wavenumber_reader.start()

    # Start parquet writer in background
    writer_thread = threading.Thread(target=write_to_file, args=(save_path,), daemon=True)
    writer_thread.start()

    try:
        main_loop(tagger, measurement_name, voltage_reader, wavenumber_reader)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping DAQ.")
    finally:
        stop_event.set()
        voltage_reader.stop()
        wavenumber_reader.stop()
        writer_thread.join()
        print("DAQ stopped gracefully.")
