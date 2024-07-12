import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QFrame, QSizePolicy, QMessageBox, QDialog, QFormLayout, QLineEdit, QDialogButtonBox)
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
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f639e;
            }
        """)

    def set_running(self, running):
        if running:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #1f639e;
                }
            """)

from PyQt5.QtWidgets import (QDialog, QFormLayout, QLineEdit, QDialogButtonBox, 
                             QComboBox, QVBoxLayout, QGroupBox, QLabel)

class ParameterDialog(QDialog):
    def __init__(self, settings_path, parent=None):
        super().__init__(parent)
        self.settings_path = settings_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Edit Parameters')
        self.setGeometry(100, 100, 400, 300)

        main_layout = QVBoxLayout(self)

        # Group for PV Name
        pv_group = QGroupBox("PV Name")
        pv_layout = QFormLayout()
        self.pv_combo = QComboBox()
        self.pv_combo.addItems(['wavenumber_1', 'wavenumber_2', 'wavenumber_3', 'wavenumber_4'])
        pv_layout.addRow("PV Name:", self.pv_combo)
        pv_group.setLayout(pv_layout)
        main_layout.addWidget(pv_group)

        # Group for Data Format
        format_group = QGroupBox("Data Format")
        format_layout = QFormLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(['parquet', 'csv'])
        format_layout.addRow("Data Format:", self.format_combo)
        format_group.setLayout(format_layout)
        main_layout.addWidget(format_group)

        # Group for Other Parameters
        other_group = QGroupBox("Other Parameters")
        other_layout = QFormLayout()
        self.parameters = self.load_parameters()
        self.line_edits = {}
        
        for key, value in self.parameters.items():
            if key not in ['pv_name', 'data_format']:
                line_edit = QLineEdit(str(value))
                other_layout.addRow(key, line_edit)
                self.line_edits[key] = line_edit
        
        other_group.setLayout(other_layout)
        main_layout.addWidget(other_group)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.save_parameters)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        # Set initial values
        self.pv_combo.setCurrentText(self.parameters.get('pv_name', 'wavenumber_1'))
        self.format_combo.setCurrentText(self.parameters.get('data_format', 'parquet'))

    def load_parameters(self):
        if os.path.exists(self.settings_path):
            with open(self.settings_path, 'r') as f:
                return json.load(f)
        return {}

    def save_parameters(self):
        self.parameters['pv_name'] = self.pv_combo.currentText()
        self.parameters['data_format'] = self.format_combo.currentText()
        
        for key, line_edit in self.line_edits.items():
            self.parameters[key] = line_edit.text()

        with open(self.settings_path, 'w') as f:
            json.dump(self.parameters, f, indent=4)

        self.accept()

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
        self.plot_button.clicked.connect(self.verify_tagger_monitor)
        main_layout.addWidget(self.plot_button)

        self.run_scripts_button = CustomButton('Run Scan')
        self.run_scripts_button.clicked.connect(self.verify_scan)
        main_layout.addWidget(self.run_scripts_button)

        # Create a horizontal layout for the set saving directory and set parameters buttons
        button_layout = QHBoxLayout()

        self.save_data_button = CustomButton('Set Saving Directory')
        self.save_data_button.setIcon(QIcon.fromTheme("folder"))
        self.save_data_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.save_data_button.clicked.connect(self.save_data)
        button_layout.addWidget(self.save_data_button)

        self.set_parameters_button = CustomButton('Set Parameters')
        self.set_parameters_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.set_parameters_button.clicked.connect(self.set_parameters)
        button_layout.addWidget(self.set_parameters_button)

        main_layout.addLayout(button_layout)

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
                background-color: #ecf0f1;
            }
            QLabel {
                color: #2c3e50;
            }
        """)

        # Timer to update process status
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_process_status)
        self.timer.start(1000)  # Update every second

    def verify_tagger_monitor(self):
        if not self.save_path:
            QMessageBox.warning(self, "Warning", "Please set the saving directory before launching the Tagger Monitor.")
            self.save_data()
        else:
            self.toggle_tagger_monitor()

    def verify_scan(self):
        if not self.save_path:
            QMessageBox.warning(self, "Warning", "Please set the saving directory before running the scan.")
            self.save_data()
        else:
            self.toggle_scan()

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
        command = f"streamlit run {self.scanning_plotter}"
        process = subprocess.Popen(command, shell=True)
        self.processes['scanning_plotter'] = psutil.Process(process.pid)
        self.status_label.setText('Scan Running')

    def save_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.save_path = QFileDialog.getExistingDirectory(self, "Select Saving Directory", options=options)
        if self.save_path:
            self.update_settings(self.save_path)
            self.status_label.setText(f'Saving Directory: {self.save_path}')
        else:
            self.status_label.setText('Save Cancelled')

    def update_settings(self, path):
        path = path.replace('/', '\\')#.replace('\\', '/
        settings = {}
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                settings = json.load(f)
        settings["saving_folder"] = path
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(settings, f)

    def set_parameters(self):
        dialog = ParameterDialog(SETTINGS_PATH, self)
        dialog.exec_()

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
