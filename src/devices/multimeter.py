import serial
import time
import threading

class HP_Multimeter:
    def __init__(self, port):
        self.device = serial.Serial(port, baudrate=9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_TWO, timeout=1)
        self.reset()
        time.sleep(0.25)
        self.setRemote()
        time.sleep(0.25)
    
    def reset(self):
        self.device.write(b'*RST\n')
        self.device.readline()
    
    def setRemote(self):
        self.device.write(b'SYSTEM:REMOTE\n')
        self.device.readline()
    
    def identity(self):
        self.device.write(b'*IDN?\n')
        response = self.device.readline()
        return response
    
    def getVoltage(self):
        self.device.write(b"MEAS:VOLT:DC?\n")
        try:
            response = self.device.readline().decode('utf-8').strip('\r\n')
            response = float(response)
        except Exception as expn:
            print('uh oh, exception occurred reading the voltage', expn)
            response = 0.0
        return response

class VoltageReader(threading.Thread):
    def __init__(self, multimeter, refresh_rate=0.5):
        super().__init__()
        self.multimeter = multimeter
        self.refresh_rate = refresh_rate
        self.voltage = 0.0
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.voltage = self.multimeter.getVoltage()
            time.sleep(self.refresh_rate)
    
    def stop(self):
        self.stop_event.set()
    
    def get_voltage(self):
        return self.voltage