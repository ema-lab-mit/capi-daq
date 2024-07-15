import sys
import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import queue
import argparse
import ntplib
import threading
from datetime import datetime, timedelta
import traceback
import json
from confluent_kafka import Producer, Consumer, KafkaException

# Kafka configuration
KAFKA_TOPIC = "daq_data"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# Kafka producer setup
producer_conf = {
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
}
producer = Producer(producer_conf)

# Kafka consumer setup
consumer_conf = {
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
    "group.id": "daq_group",
    "auto.offset.reset": "earliest",
}
consumer = Consumer(consumer_conf)
consumer.subscribe([KAFKA_TOPIC])

this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from beamline_tagg_gui.utils.physics_tools import time_to_flops, flops_to_time

N_CHANNELS = ["A", "B", "C", "D"]
SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
MAX_DF_LENGTH_IN_MEMORY_IF_SCANNING = 1_000_000

# Read the configuration file for the initialization parameters


def get_card_settings():
    try:
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        return {
            "tof_start": float(settings.get("tof_start", "1e-6")),
            "tof_end": float(settings.get("tof_end", "20e-6")),
            "channel_level": float(settings.get("channel_level", "-0.5")),
            "trigger_level": float(settings.get("trigger_level", "-0.5")),
            "pv_name": settings.get("pv_name", "LaserLab:wavenumber_1"),
            "data_format": settings.get("data_format", "parquet"),
        }
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}


modified_settings = get_card_settings()
STOP_TIME_WINDOW = modified_settings.get("tof_end", 20e-6)
INIT_TIME = modified_settings.get("tof_start", 1e-6)
SAVING_FORMAT = modified_settings.get("data_format", "parquet")
LASER_PVNAME = modified_settings.get("pv_name", "LaserLab:wavenumber_1")
CHANNEL_LEVEL = modified_settings.get("channel_level", -0.5)

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
        response = c.request("pool.ntp.org", timeout=5)
        ntp_time = datetime.fromtimestamp(response.tx_time)
        local_time = datetime.now()
        ntp_time_offset = (ntp_time - local_time).total_seconds()
        last_sync_time = local_time
        ntp_sync_queue.put(("success", ntp_time_offset))
    except Exception as e:
        ntp_sync_queue.put(("error", str(e)))


def write_settings(saving_path):
    settings_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
    settings = {"save_path": saving_path}
    with open(settings_path, "w") as f:
        json.dump(settings, f)


def get_synced_time():
    return datetime.now() + timedelta(seconds=ntp_time_offset)


def time_sync_thread():
    while True:
        sync_time()
        time.sleep(time_sync_interval)


class Tagger:
    def __init__(self, index=0):
        self.index = index
        self.trigger_level = 0
        self.trigger_type = True
        self.channels = initialization_params["trigger"]["channels"]
        self.levels = initialization_params["trigger"]["levels"]
        self.type = initialization_params["trigger"]["types"]
        self.starts = initialization_params["trigger"]["starts"]
        self.stops = initialization_params["trigger"]["stops"]
        self.flops_to_time = flops_to_time
        self.card = None
        self.started = False
        self.init_card()
        print("card initialized")

    def set_trigger_level(self, level):
        self.trigger_level = level

    def set_trigger_rising(self):
        self.set_trigger_type(type="rising")

    def set_trigger_falling(self):
        self.set_trigger_type(type="falling")

    def set_trigger_type(self, type="falling"):
        self.trigger_type = type == "rising"

    def enable_channel(self, channel):
        self.channels[channel] = True

    def disable_channel(self, channel):
        self.channels[channel] = False

    def set_channel_level(self, channel, level):
        self.levels[channel] = level

    def set_channel_rising(self, channel):
        self.set_type(channel, type="rising")

    def set_channel_falling(self, channel):
        self.set_type(channel, type="falling")

    def set_type(self, channel, type="falling"):
        self.type[channel] = type == "rising"

    def set_channel_window(self, channel, start=0, stop=600000):
        self.starts[channel] = start
        self.stops[channel] = stop

    def init_card(self):
        kwargs = {}
        kwargs["trigger_level"] = self.trigger_level
        kwargs["trigger_rising"] = self.trigger_type
        for i, info in enumerate(
            zip(self.channels, self.levels, self.type, self.starts, self.stops)
        ):
            ch, l, t, st, sp = info
            kwargs["channel_{}_used".format(i)] = ch
            kwargs["channel_{}_level".format(i)] = l
            kwargs["channel_{}_rising".format(i)] = t
            kwargs["channel_{}_start".format(i)] = st
            kwargs["channel_{}_stop".format(i)] = sp
        kwargs["index"] = self.index
        if self.card is not None:
            self.stop()
        self.card = tg(**kwargs)

    def start_reading(self):
        self.started = True
        self.card.startReading()
        print("started reading")

    def get_data(self, timeout=2):
        start = time.time()
        while time.time() - start < timeout:
            status, data = self.card.getPackets()
            if status == 0:  # trigger detected, so there is data
                if data == []:
                    print("no data")
                    return None
                else:
                    return data
            elif (
                status == 1
            ):  # no trigger seen yet, go to sleep for a bit and try again
                time.sleep(0.01)
            else:
                raise ValueError
        return None

    def stop(self):
        if self.card is not None:
            if self.started:
                self.card.stopReading()
                self.started = False
            self.card.stop()
            self.card = None

    def build_dataframe(self, data, wnum):
        df = pd.DataFrame(
            data=data, columns=["bunch", "n_events", "channels", "timestamp"]
        )
        df.timestamp = df.timestamp.apply(self.flops_to_time)
        df["synced_time"] = [get_synced_time() for _ in range(len(df))]
        df["wavenumber"] = wnum
        return df


def get_wnum():
    return round(float(wavenumber.get()), 5)


def time_converter(value):
    return datetime.timedelta(milliseconds=value)


def main_loop(tagger, is_scanning=False, pv_name=LASER_PVNAME):
    tagger.set_trigger_falling()
    tagger.set_trigger_level(-0.5)
    tagger.start_reading()
    numbers_of_trigs = []
    sync_thread = threading.Thread(target=time_sync_thread, daemon=True)
    sync_thread.start()
    global wavenumber
    wavenumber = PV("LaserLab:wavenumber_1")

    while True:
        # Check for NTP sync results
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
            print(current_wnum)
            current_df = tagger.build_dataframe(data, current_wnum)
            for _, row in current_df.iterrows():
                producer.produce(
                    KAFKA_TOPIC, key=str(row["bunch"]), value=row.to_json()
                )
                producer.flush()

        time.sleep(initialization_params["refresh_rate"])


def load_settings():
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


def metadata_writer(folder_location, data_complete_path):
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

    folder_location = load_settings()["saving_folder"]
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
    tagger = Tagger()
    main_loop(tagger, args.is_scanning)
