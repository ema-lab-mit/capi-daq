import sys
import os
import time

this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from beamline_tagg_gui.utils.physics_tools import flops_to_time


class Tagger():
    def __init__(self, index=0, initialization_params:dict = {}):
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

    def set_trigger_level(self, level):
        self.trigger_level = level

    def set_trigger_rising(self):
        self.set_trigger_type(type='rising')

    def set_trigger_falling(self):
        self.set_trigger_type(type='falling')

    def set_trigger_type(self, type='falling'):
        self.trigger_type = type == 'rising'

    def enable_channel(self, channel):
        self.channels[channel] = True

    def disable_channel(self, channel):
        self.channels[channel] = False

    def set_channel_level(self, channel, level):
        self.levels[channel] = level

    def set_channel_rising(self, channel):
        self.set_type(channel, type='rising')

    def set_channel_falling(self, channel):
        self.set_type(channel, type='falling')

    def set_type(self, channel, type='falling'):
        self.type[channel] = type == 'rising'

    def set_channel_window(self, channel, start=0, stop=600000):
        self.starts[channel] = start
        self.stops[channel] = stop

    def init_card(self):
        kwargs = {}
        kwargs['trigger_level'] = self.trigger_level
        kwargs['trigger_rising'] = self.trigger_type
        for i, info in enumerate(zip(self.channels, self.levels, self.type, self.starts, self.stops)):
            ch, l, t, st, sp = info
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

    def get_data(self, timeout=2):
        start = time.time()
        while time.time() - start < timeout:
            status, data = self.card.getPackets()
            if status == 0:  # trigger detected, so there is data
                if data == []:
                    print('no data')
                    return None
                else:
                    return data
            elif status == 1:  # no trigger seen yet, go to sleep for a bit and try again
                time.sleep(0.001)
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
