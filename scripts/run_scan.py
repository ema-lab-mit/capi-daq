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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("daq.log")  # Optional: Log to a file
    ]
)
logger = logging.getLogger(__name__)

# Import PyArrow removed as it's no longer needed for CSV
# import pyarrow as pa
# import pyarrow.parquet as pq

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
from fast_tagger_gui.src.devices.spectrometer import SpectrometreReader


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
            "data_format": settings.get("data_format", "csv"),  # Changed default to 'csv'
            "saving_file": settings.get("saving_file", "data.csv"),  # Changed default to 'data.csv'
        }
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return {}

modified_settings = get_card_settings()
STOP_TIME_WINDOW = modified_settings.get("tof_end", 20e-6)
INIT_TIME = modified_settings.get("tof_start", 1e-6)
CHANNEL_LEVEL = modified_settings.get("channel_level", -0.5)
TRIGGER_LEVEL = modified_settings.get("trigger_level", -0.5)
SAVING_FORMAT = modified_settings.get("data_format", "csv")  # Ensure it's 'csv'
SAVING_FILE = modified_settings.get("saving_file", "data.csv")  # Ensure it's 'csv'

initialization_params = {
    "trigger": {
        "channels": [True, True, True, True],
        "levels": [CHANNEL_LEVEL for _ in range(4)],
        "types": [False for _ in range(4)],
        "starts": [int(time_to_flops(INIT_TIME)) for _ in range(4)],
        "stops": [int(time_to_flops(STOP_TIME_WINDOW)) for _ in range(4)],
    },
    "refresh_rate": 0.5,
}

# Initialize InfluxDB Client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_to_influxdb(data, data_name, voltage, wavenumbers, spectr, trigger_rate, event_rate=None):
    points = []
    print("Read spectr", spectr, type(spectr))
    for d in data:
        data_ingestion = datetime.fromtimestamp(d[-1])#.strftime()
        points.append(Point("scan").tag("type", data_name).field("bunch", d[0]).time(data_ingestion, WritePrecision.NS))
        points.append(Point("scan").tag("type", data_name).field("n_events", d[1]).time(data_ingestion, WritePrecision.NS))
        points.append(Point("scan").tag("type", data_name).field("channel", d[2]).time(data_ingestion, WritePrecision.NS))
        points.append(Point("scan").tag("type", data_name).field("time_offset", float(d[3])).time(data_ingestion, WritePrecision.NS))
        points.append(Point("scan").tag("type", data_name).field("id_timestamp", d[4]).time(data_ingestion, WritePrecision.NS))
        points.append(Point("scan").tag("type", data_name).field("voltage", voltage).time(data_ingestion, WritePrecision.NS))
        points.append(Point("scan").tag("type", data_name).field("trigger_rate", trigger_rate).time(data_ingestion, WritePrecision.NS))
        points.append(Point("scan").tag("type", data_name).field("event_rate", event_rate).time(data_ingestion, WritePrecision.NS))
        points.append(Point("scan").tag("type", data_name).field(f"spectr_peak", str(spectr)).time(data_ingestion, WritePrecision.NS))
        points += [Point("scan").tag("type", data_name).field(f"wn_{i}", wavenumbers[i-1]).time(data_ingestion, WritePrecision.NS) for i in range(1, 5)]
    try:
        write_api.write(bucket=INFLUXDB_BUCKET, record=points)
    except Exception as e:
        print(f"Error writing to InfluxDB: {e}")

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
    Writes data batches from the queue to a CSV file efficiently and safely.
    Ensures that headers are written only once and handles file flushing to prevent corruption.
    """
    file_exists = os.path.isfile(saving_file)
    write_header = not file_exists or os.path.getsize(saving_file) == 0

    try:
        with open(saving_file, 'a', newline='', buffering=1) as file:
            while not stop_event.is_set() or not data_queue.empty():
                try:
                    data_batch = data_queue.get(timeout=1)
                    if not data_batch:
                        continue

                    df = pd.DataFrame(data_batch)

                    # Write to CSV
                    df.to_csv(
                        file,
                        header=write_header,
                        index=False,
                        mode='a',
                        lineterminator='\n'
                    )

                    if write_header:
                        write_header = False

                    file.flush()
                    os.fsync(file.fileno())

                    logger.info(f"Wrote batch of size {len(data_batch)} to {saving_file}")
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error writing to CSV file: {e}")

            # Handle any remaining data after stop_event is set
            while not data_queue.empty():
                try:
                    data_batch = data_queue.get_nowait()
                    if not data_batch:
                        continue

                    df = pd.DataFrame(data_batch)
                    df = df

                    df.to_csv(
                        file,
                        header=write_header,
                        index=False,
                        mode='a',
                        lineterminator='\n'
                    )

                    if write_header:
                        write_header = False

                    file.flush()
                    os.fsync(file.fileno())

                    logger.info(f"Wrote final batch of size {len(data_batch)} to {saving_file}")
                except Exception as e:
                    logger.error(f"Error writing final data to CSV file: {e}")

    except Exception as e:
        logger.error(f"Error initializing CSV file writer: {e}")

    logger.info(f"Closed CSV writer for {saving_file}")

def main_loop(tagger, measurement_name, voltage_reader, wavenumber_reader, initialization_params=initialization_params):
    """
    Read data from the Tagger, grab voltage and wavenumbers, and send them
    both to CSV (via the queue) and to InfluxDB.
    """
    tagger.set_trigger_falling()
    tagger.set_trigger_level(float(TRIGGER_LEVEL))
    tagger.start_reading()
    i = 0
    alpha = 0.1
    event_rate = 0.0
    batched_data = []
    i_time = time.time()
    while not stop_event.is_set():
        now_str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        try:
            data, new_triggers, new_events = tagger.get_data(return_splitted=True)
        except Exception as e:
            logger.error(f"Error getting data from Tagger: {e}")
            data, new_triggers, new_events = [], [], []
        len_triggers = len(new_triggers)
        time_now = time.time()
        if len_triggers > 0:
            print(f"Event rate: {event_rate:.2f} Hz")
            try:
                voltage = voltage_reader.get_voltage()
            except Exception as e:
                logger.error(f"Error getting voltage: {e}")
                voltage = 0.0
            try:
                wavenumbers = wavenumber_reader.get_wavenumbers()
            except Exception as e:
                logger.error(f"Error getting wavenumbers: {e}")
                wavenumbers = [0.0, 0.0, 0.0, 0.0]
            try:
                spectr = (spectrometer_reader.get_spec())
            except Exception as e:
                logger.error(f"Error getting spectrometer data: {e}")
                spectr = "0.0"
            delta_t_total = time_now - i_time
            try:
                trigger_rate = new_triggers[-1][0] / (delta_t_total)
            except ZeroDivisionError:
                trigger_rate = 0.000
            
            event_rate = ((1 - alpha) * ((len(new_events) * trigger_rate) / len_triggers) + alpha * event_rate) 

            write_to_influxdb(new_events, measurement_name, voltage, wavenumbers, spectr, trigger_rate=trigger_rate, event_rate=event_rate)
            if len(new_events):
                # Append timestamp to each event if not already present
                # Assuming 'new_events' is a list of lists or tuples
                for event in new_events:
                    if len(event) < 5:
                        event.append(now_str)  # Append timestamp if missing
                    else:
                        event[4] = now_str  # Update existing timestamp
                batched_data += new_events
                i += 1
                if i % POSTING_BATCH_SIZE == 0:
                    try:
                        data_queue.put(batched_data, timeout=1)
                        batched_data = []
                    except queue.Full:
                        logger.warning("Data queue is full on batch data put.")
        time.sleep(initialization_params["refresh_rate"])

    # Final flush before exit
    try:
        if batched_data:
            data_queue.put(batched_data, timeout=1)
            write_to_influxdb(batched_data, measurement_name, voltage, wavenumbers, trigger_rate=trigger_rate)
    except queue.Full:
        logger.warning("Data queue is full on final data put.")
    except Exception as e:
        logger.error(f"Error during final data put: {e}")

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
    # e.g. scan_2025_01_17_14_33_42.csv => measurement_name=2025_01_17_14_33_42
    measurement_name = os.path.basename(save_path).split("scan_")[1].split(".")[0]

    try:
        multimeter = HP_Multimeter("COM" + str(voltage_port))
    except Exception as e:
        logger.error(f"Error initializing multimeter: {e}")
    try:
        voltage_reader = VoltageReader(multimeter, refresh_rate=refresh_rate)
        wavenumber_reader = WavenumberReader(refresh_rate=refresh_rate)
        spectrometer_reader = SpectrometreReader(refresh_rate=refresh_rate)
        voltage_reader.start()
        wavenumber_reader.start()
        spectrometer_reader.start()
    except Exception as e:
        logger.error(f"Error initializing readers: {e}")


    # Start CSV writer in background (non-daemon)
    writer_thread = threading.Thread(target=write_to_file, args=(save_path,))
    writer_thread.start()

    try:
        main_loop(tagger, measurement_name, voltage_reader, wavenumber_reader, initialization_params)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping DAQ.")
    finally:
        stop_event.set()
        voltage_reader.stop()
        wavenumber_reader.stop()
        spectrometer_reader.stop()
        writer_thread.join()  # Wait for writer_thread to finish
        voltage_reader.join()  # Wait for VoltageReader to finish
        wavenumber_reader.join()  # Wait for WavenumberReader to finish
        spectrometer_reader.join()
        logger.info("DAQ stopped gracefully.")
