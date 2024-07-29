import sys
import os
import subprocess
import signal
import time

def run_commands():
    # Start the server
    server_process = subprocess.Popen(["influxd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Give the server some time to start up
    time.sleep(2)
    
    # Start the GUI
    activate_env = "conda activate envNTT"
    change_dir = "cd D:\\NewData"
    run_gui = "python C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\main.py"
    full_command = f"{activate_env} && {change_dir} && {run_gui}"
    
    # Run the command directly in the terminal
    gui_process = subprocess.Popen(full_command, shell=True)
    
    return server_process, gui_process

def stop_processes(server_process, gui_process):
    # Stop the server process
    if server_process:
        server_process.terminate()
        server_process.wait()
    
    # Stop the GUI process
    if gui_process:
        gui_process.terminate()
        gui_process.wait()

def signal_handler(signum, frame):
    print("Interrupt received, stopping processes...")
    stop_processes(server_process, gui_process)
    sys.exit(0)

if __name__ == "__main__":
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the commands
    server_process, gui_process = run_commands()

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping processes...")
        stop_processes(server_process, gui_process)
