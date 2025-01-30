import threading
import time 
from epics import PV
pv_names = ["SPECTROMETER:SPECTRUM"]

global wavenumbers_pvs

class SpectrometreReader(threading.Thread):
    def __init__(self, refresh_rate=0.2):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.spectrum = [0.0, 0.0, 0.0, 0.0]
        self.stop_event = threading.Event()
        self.pv = PV(pv_names[0])
        if not self.pv.connected:
            print(f"Error connecting to PV: {pv_names[0]}")
            self.stop()

    def run(self):
        while not self.stop_event.is_set():
            self.spectrum = self.get_spec()
            time.sleep(self.refresh_rate)
    
    def stop(self):
        self.stop_event.set()
    
    def get_spec(self, patience=0.1, max_tries=10):
        try:
            spec = self.pv.get()
            tries = 0
            while spec is None and tries < max_tries:
                time.sleep(patience)
                spec = self.pv.get()
                tries += 1
            return spec
        except Exception as e:
            print(f"Error getting spectrum: {e}")
            return 0.00000