import threading
import time 
from epics import PV
wavenumbers_pv_names = ["LaserLab:wavenumber_1", "LaserLab:wavenumber_2", "LaserLab:wavenumber_3", "LaserLab:wavenumber_4"]

global wavenumbers_pvs
wavenumbers_pvs = [PV(name) for name in wavenumbers_pv_names]

class WavenumberReader(threading.Thread):
    def __init__(self, refresh_rate=0.5):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.wavenumbers = [0.0, 0.0, 0.0, 0.0]
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.wavenumbers = [self.get_wnum(i) for i in range(1, 5)]
            time.sleep(self.refresh_rate)
    
    def stop(self):
        self.stop_event.set()
    
    def get_wnum(self, i=1):
        try:
            return round(float(wavenumbers_pvs[i - 1].get()), 5)
        except Exception as e:
            print(f"Error getting wavenumber: {e}")
            return 0.00000
    
    def get_wavenumbers(self):
        return self.wavenumbers