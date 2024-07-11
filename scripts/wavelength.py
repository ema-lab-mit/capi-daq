# The code by Sherlock
import os
import time
import datetime
import traceback
import pandas as pd
from pylablib.devices import M2
from epics import PV


def get_wnum():
    return round(float(wavenumber.get()),5)

def time_converter(value):
    return datetime.timedelta(milliseconds = value)

def write_data(xlist, ylist, wnum, rate):
    xlist.append(xlist[-1] + time_converter(rate))    
    ylist.append(wnum)

try:
    now = datetime.datetime.now()
    laser = M2.Solstis("192.168.1.222", 39933)
    wavenumber = PV("LaserLab:wavenumber_1")
    xDat, yDat = [], []
    rate = float(input("Reading rate:"))
    filename = input("Filename:")
    filepath = input("Enter the filepath to save the data:")
    filename = f"{filename}.csv"
    filepath = os.path.join(filepath, filename)

    while True:
        print(xDat)
        print(yDat)
        current_wnum = get_wnum()
        if len(xDat) == 0:
            xDat.append(now)
            yDat.append(current_wnum)
        else:
            write_data(xDat, yDat, current_wnum, rate)
            # xDat.append(xDat[-1] + time_converter(rate))
            # yDat = np.append(yDat, current_wnum)
        # print(xDat)
        # print(yDat)

        time.sleep(rate)

except Exception as e:
    print(f"An error occurred: {e} \n {traceback.format_exc()}")

finally:
    df = pd.DataFrame({'Time': xDat, 'Wavenumber': yDat})
    df.to_csv(filepath, index=False, mode = "x")
    print(f"x:{len(xDat)}, y:{len(yDat)}")

    

