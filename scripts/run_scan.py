import sys
import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from beamline_tagg_gui.utils.physics_tools import time_to_flops, flops_to_time
import queue
import numpy as np
import time
import pandas as pd
import argparse
import ntplib
from time import ctime
import threading
from datetime import datetime, timedelta

N_CHANNELS = ["A", "B", "C", "D"]
STOP_TIME_WINDOW = 20e-6 # 200us
INIT_TIME = 1e-6
MAX_DF_LENGTH_IN_MEMORY = 100_000

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

def get_synced_time():
    return datetime.now() + timedelta(seconds=ntp_time_offset)

def time_sync_thread():
    while True:
        sync_time()
        time.sleep(time_sync_interval)

def sync_time():
    global ntp_time_offset, last_sync_time
    c = ntplib.NTPClient()
    try:
        response = c.request('pool.ntp.org', timeout=5)
        ntp_time = datetime.fromtimestamp(response.tx_time)
        local_time = datetime.now()
        ntp_time_offset = (ntp_time - local_time).total_seconds()
        last_sync_time = local_time
        print(f"Time synced. Offset: {ntp_time_offset}")
    except Exception as e:
        print(f"Time sync error: {str(e)}")

def get_synced_time():
    return datetime.now() + timedelta(seconds=ntp_time_offset)


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
                time.sleep(0.001)
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
        
    def build_dataframe(self, data):
        df = pd.DataFrame(data=data, columns=["bunch", "n_events", "channels", "timestamp"])
        df.timestamp = df.timestamp.apply(self.flops_to_time)
        df.timestamp = df.timestamp + initialization_params["trigger"]["starts"][0]
        df['synced_time'] = [get_synced_time() for _ in range(len(df))]
        return df


def main_loop(tagger: object, saving_path: str = None, is_scanning: bool = False):
    df = pd.DataFrame(columns=["bunch", "n_events", "channels", "timestamp", "synced_time"])
    tagger.set_trigger_falling()
    tagger.set_trigger_level(-0.5)
    tagger.start_reading()
    numbers_of_trigs = []
    starting_time = time.time()

    # Start the time sync thread
    sync_thread = threading.Thread(target=time_sync_thread, daemon=True)
    sync_thread.start()

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
            current_df = tagger.build_dataframe(data)
            df = pd.concat([df, current_df])
            numbers_of_trigs.append(len(data))
            
            if not is_scanning and len(df) > MAX_DF_LENGTH_IN_MEMORY:
                df = df.iloc[-MAX_DF_LENGTH_IN_MEMORY:]
            
            if saving_path:
                # df.to_parquet(saving_path)
                df.to_csv(saving_path)
        
        time.sleep(initialization_params["refresh_rate"])


if __name__ == "__main__":
    tagger = Tagger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--refresh_rate", type=float, default=0.5)
    parser.add_argument("--is_scanning", type=bool, default=False)
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        folder_path = os.path.dirname(args.save_path)
        os.makedirs(folder_path, exist_ok=True)
    
    initialization_params["save_path"] = args.save_path
    initialization_params["refresh_rate"] = args.refresh_rate
    main_loop(tagger, args.save_path)