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
from pylablib.devices import M2
from epics import PV
import json

this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from beamline_tagg_gui.utils.physics_tools import time_to_flops, flops_to_time

N_CHANNELS = ["A", "B", "C", "D"]
STOP_TIME_WINDOW = 20e-6 # 200us
INIT_TIME = 1e-6
MAX_DF_LENGTH_IN_MEMORY_IF_SCANNING = 100_000
SAVING_FORMAT = "csv"

initialization_params = {
    "trigger": {
        "channels": [True, True, True, True],
        "levels": [-0.5 for _ in range(4)],
        "types": [False for _ in range(4)],
        "starts": [int(time_to_flops(INIT_TIME)) for _ in range(4)],
        "stops": [int(time_to_flops(STOP_TIME_WINDOW)) for _ in range(4)],
    },
    "refresh_rate": 0.5,
}
ntp_time_offset = 0.0
last_sync_time = None
time_sync_interval = 30  # Sync every .5 minutes
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
        
def write_settings(saving_path):
    settings_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
    settings = {"save_path": saving_path}
    with open(settings_path, 'w') as f:
        json.dump(settings, f)

def get_synced_time():
    return datetime.now() + timedelta(seconds=ntp_time_offset)

def time_sync_thread():
    while True:
        sync_time()
        time.sleep(time_sync_interval)

class Tagger():
    def __init__(self,index = 0):
        self.index = index
        self.trigger_level = 0
        self.trigger_type = True
        self.channels = initialization_params['trigger']['channels']
        self.levels = initialization_params['trigger']['levels']
        self.type = initialization_params['trigger']['types']
        self.starts = initialization_params['trigger']['starts']
        self.stops = initialization_params['trigger']['stops']
        self.flops_to_time = flops_to_time
        self.card = None
        self.started = False
        self.init_card()
        print('card initialized')

    def set_trigger_level(self,level):
        self.trigger_level = level

    def set_trigger_rising(self):
        self.set_trigger_type(type='rising')

    def set_trigger_falling(self):
        self.set_trigger_type(type='falling')

    def set_trigger_type(self,type='falling'):
        self.trigger_type = type == 'rising'

    def enable_channel(self, channel):
        self.channels[channel] = True

    def disable_channel(self, channel):
        self.channels[channel] = False

    def set_channel_level(self,channel,level):
        self.levels[channel] = level

    def set_channel_rising(self,channel):
        self.set_type(channel,type='rising')

    def set_channel_falling(self,channel):
        self.set_type(channel,type='falling')

    def set_type(self,channel,type='falling'):
        self.type[channel] = type == 'rising'

    def set_channel_window(self,channel,start=0,stop=600000):
        self.starts[channel] = start
        self.stops[channel] = stop

    def init_card(self):
        kwargs = {}
        kwargs['trigger_level'] = self.trigger_level
        kwargs['trigger_rising'] = self.trigger_type
        for i,info in enumerate(zip(self.channels,self.levels,self.type,self.starts,self.stops)):
            ch,l,t,st,sp = info
            kwargs['channel_{}_used'.format(i)] = ch
            kwargs['channel_{}_level'.format(i)] = l
            kwargs['channel_{}_rising'.format(i)] = t
            kwargs['channel_{}_start'.format(i)] = st
            kwargs['channel_{}_stop'.format(i)] = sp
        kwargs['index'] = self.index
        if self.card is not None:
            self.stop()
        self.card = tg(**kwargs)

    def start_reading(self):
        self.started = True
        self.card.startReading()
        print('started reading')

    def get_data(self,timeout = 2):
        start = time.time()
        while time.time()-start < timeout:
            status, data = self.card.getPackets()
            if status == 0: # trigger detected, so there is data
                if data == []:
                    print('no data')
                    return None
                else:
                    return data
            elif status == 1: # no trigger seen yet, go to sleep for a bit and try again
                time.sleep(0.01)
            else:
                raise ValueError
        return None

    def stop(self):
        if not self.card is None:
            if self.started:
                self.card.stopReading()
                self.started = False
            self.card.stop()
            self.card = None

    def build_dataframe(self, data, wnum):
        df = pd.DataFrame(data=data, columns=["bunch", "n_events", "channels", "timestamp"])
        df.timestamp = df.timestamp.apply(self.flops_to_time)
        df['synced_time'] = [get_synced_time() for _ in range(len(df))]
        df['wavenumber'] = wnum
        return df

def get_wnum():
    return round(float(wavenumber.get()), 5)

def time_converter(value):
    return datetime.timedelta(milliseconds=value)

def main_loop(tagger, saving_path=None, is_scanning=False):
    df = pd.DataFrame(columns=["bunch", "n_events", "channels", "timestamp", "synced_time", "wavenumber"])
    tagger.set_trigger_falling()
    tagger.set_trigger_level(-0.5)
    tagger.start_reading()
    numbers_of_trigs = []
    sync_thread = threading.Thread(target=time_sync_thread, daemon=True)
    sync_thread.start()

    laser = M2.Solstis("192.168.1.222", 39933)
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
            df = pd.concat([df, current_df])
            numbers_of_trigs.append(len(data))
            if not is_scanning and len(df) > MAX_DF_LENGTH_IN_MEMORY_IF_SCANNING:
                df = df.iloc[-MAX_DF_LENGTH_IN_MEMORY_IF_SCANNING:]
            if saving_path:
                if SAVING_FORMAT == "csv":
                    df.to_csv(saving_path, index=False)
                elif SAVING_FORMAT == "parquet":
                    df.to_parquet(saving_path)

        time.sleep(initialization_params["refresh_rate"])

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
    tagger = Tagger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--refresh_rate", type=float, default=0.5)
    parser.add_argument("--is_scanning", type=bool, default=False)
    args = parser.parse_args()
    initialization_params["refresh_rate"] = args.refresh_rate
    
    save_path = args.save_path
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    identifier = str(time_now).replace(":", "-").replace(" ", "_").replace("-", "_")
    folder_location = save_path if os.path.isdir(save_path) else os.path.dirname(save_path)
    if not os.path.exists(folder_location):
        os.makedirs(folder_location, exist_ok=True)
        
    name = "tagger_monitor_data_" + identifier + "." + SAVING_FORMAT if os.path.isdir(save_path) else os.path.basename(save_path).split(".")[0] + "_" + identifier+ "." + SAVING_FORMAT
    save_path = os.path.join(folder_location, name)
    write_settings(save_path)
    metadata_writer(folder_location, save_path)
    initialization_params["save_path"] = save_path
    print(f"Saving data to {save_path}")
    main_loop(tagger, save_path, args.is_scanning)
