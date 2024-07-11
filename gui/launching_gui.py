import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QLabel, QFrame, QSizePolicy)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QTimer
from threading import Thread
import subprocess
import psutil
SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"


class CustomButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(50)
        self.setFont(QFont('Arial', 12))
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)

    def set_running(self, running):
        if running:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #FF0000;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #CC0000;
                }
                QPushButton:pressed {
                    background-color: #990000;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3e8e41;
                }
            """)

class SimpleGUI(QWidget):

    def __init__(self, plotting_script, tagger_script, scan_script, scanning_plotter):
        super().__init__()
        self.save_path = ""
        self.plotting_script = plotting_script
        self.tagger_script = tagger_script
        self.scan_script = scan_script
        self.scanning_plotter = scanning_plotter
        self.processes = {}
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title_label = QLabel('CAPI')
        title_label.setFont(QFont('Arial', 36, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel('Comprehensive Acquisition and Processing Interface\n EMA LAB')
        subtitle_label.setFont(QFont('Arial', 12))
        subtitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        # Buttons
        self.plot_button = CustomButton('Launch Tagger Monitor')
        self.plot_button.clicked.connect(self.toggle_tagger_monitor)
        main_layout.addWidget(self.plot_button)

        self.run_scripts_button = CustomButton('Run Scan')
        self.run_scripts_button.clicked.connect(self.toggle_scan)
        main_layout.addWidget(self.run_scripts_button)

        self.save_data_button = CustomButton('Set Saving Directory')
        self.save_data_button.clicked.connect(self.save_data)
        main_layout.addWidget(self.save_data_button)

        # Status
        self.status_label = QLabel('Ready')
        self.status_label.setFont(QFont('Arial', 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self.setWindowTitle('CAPI - DAQ System')
        self.setGeometry(300, 300, 500, 450)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
        """)

        # Timer to update process status
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_process_status)
        self.timer.start(1000)  # Update every second

    def toggle_tagger_monitor(self):
        if 'streamlit' in self.processes and 'tagger' in self.processes:
            self.kill_process('streamlit')
            self.kill_process('tagger')
            self.plot_button.set_running(False)
            self.status_label.setText('Tagger Monitor Terminated')
        else:
            self.launch_tagger_monitor()

    def launch_tagger_monitor(self):
        self.status_label.setText('Launching Tagger Monitor...')
        thread_tagger = Thread(target=self.launch_tagger_thread)
        thread_streamlit = Thread(target=self.launch_streamlit_thread)
        thread_tagger.start()
        thread_streamlit.start()

    def launch_streamlit_thread(self):
        command = f"streamlit run {self.plotting_script}"
        process = subprocess.Popen(command, shell=True)
        self.processes['streamlit'] = psutil.Process(process.pid)
        self.plot_button.set_running(True)

    def launch_tagger_thread(self):
        command = f"python {self.tagger_script}"
        process = subprocess.Popen(command, shell=True)
        self.processes['tagger'] = psutil.Process(process.pid)
        self.status_label.setText('Tagger Monitor Launched')

    def toggle_scan(self):
        if 'scan' in self.processes and 'scanning_plotter' in self.processes:
            self.kill_process('scan')
            self.kill_process('scanning_plotter')
            self.run_scripts_button.set_running(False)
            self.status_label.setText('Scan Terminated')
        else:
            self.run_scan()

    def run_scan(self):
        self.status_label.setText('Running Scan...')
        thread_scan = Thread(target=self.run_scan_thread)
        thread_plotter = Thread(target=self.run_scanning_plotter_thread)
        thread_scan.start()
        thread_plotter.start()

    def run_scan_thread(self):
        command = f"python {self.scan_script}"
        process = subprocess.Popen(command, shell=True)
        self.processes['scan'] = psutil.Process(process.pid)
        self.run_scripts_button.set_running(True)

    def run_scanning_plotter_thread(self):
        command = f"python {self.scanning_plotter}"
        process = subprocess.Popen(command, shell=True)
        self.processes['scanning_plotter'] = psutil.Process(process.pid)
        self.status_label.setText('Scan Running')

    def save_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.save_path, _ = QFileDialog.getSaveFileName(self, "Configure Saving Path", "", "CSV Files (*.csv)", options=options)
        if self.save_path:
            self.write_settings(self.save_path)
            self.status_label.setText(f'Data Saved: {self.save_path}')
        else:
            self.status_label.setText('Save Cancelled')

    def write_settings(self, path):
        settings = {"save_path": path}
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(settings, f)

    def closeEvent(self, event):
        # Prevent the widget from closing immediately
        event.ignore()

        # Terminate all processes
        for process_name in list(self.processes.keys()):
            self.kill_process(process_name)

        # Use a single-shot timer to delay the actual closing
        QTimer.singleShot(1000, self.finalClose)

    def finalClose(self):
        # This method will be called after a delay, ensuring all processes have been terminated
        QApplication.quit()

    def kill_process(self, process_name):
        if process_name in self.processes:
            try:
                process = self.processes[process_name]
                for child in process.children(recursive=True):
                    child.terminate()
                    child.wait()  # Wait for the child to actually terminate
                process.terminate()
                process.wait()  # Wait for the process to actually terminate
                self.processes.pop(process_name)
                self.status_label.setText(f'{process_name.capitalize()} process terminated')
            except psutil.NoSuchProcess:
                self.status_label.setText(f'{process_name.capitalize()} process not found')


    def update_process_status(self):
        for process_name, process in list(self.processes.items()):
            if not process.is_running():
                self.processes.pop(process_name)
                if process_name in ['streamlit', 'tagger']:
                    if 'streamlit' not in self.processes and 'tagger' not in self.processes:
                        self.plot_button.set_running(False)
                elif process_name in ['scan', 'scanning_plotter']:
                    if 'scan' not in self.processes and 'scanning_plotter' not in self.processes:
                        self.run_scripts_button.set_running(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Example usage with script paths and parameters
    plotting_script = "path/to/plotting_script.py"
    tagger_script = "path/to/tagger_script.py"
    scan_script = "path/to/scan_script.py"
    scanning_plotter = "path/to/scanning_plotter.py"
    
    ex = SimpleGUI(plotting_script, tagger_script, scan_script, scanning_plotter)
    ex.show()
    sys.exit(app.exec_())
