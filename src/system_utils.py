# utils.py
import os
import json
import pandas as pd
from datetime import datetime
import argparse

SETTINGS_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\fast_tagger_gui\\settings.json"
TOKEN_PATH = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ\\secrets.json"

def get_secrets(token_path=TOKEN_PATH):
    try:
        with open(token_path, 'r') as f:
            secrets = json.load(f)
        return secrets.get("token", "")
    except Exception as e:
        print(f"Error loading secrets: {e}")
        return ""

def load_path(settings_path=SETTINGS_PATH):
    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)
        return {
            "saving_folder": settings.get("saving_folder", ""),
            "saving_file": settings.get("saving_file", ""),
        }
    except Exception as e:
        print(f"Error loading settings: {e}")
        return ""

def update_settings_file(saving_file, settings_path=SETTINGS_PATH):
    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)
        settings["saving_file"] = saving_file
        with open(settings_path, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        print(f"Error updating settings file: {e}")

def metadata_writer(folder_location, data_complete_path, initialization_params):
    file_name = "metadata_tagger_monitor.csv"
    payload = {
        "refresh_rate": initialization_params["refresh_rate"],
        "stop_time_window": initialization_params.get("tof_end"),
        "init_time": initialization_params.get("tof_start"),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_complete_path": data_complete_path,
        "format": initialization_params.get("data_format"),
    }
    metadata_path = os.path.join(folder_location, file_name)
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        # concatenate
        metadata_df = pd.concat([metadata_df, pd.DataFrame([payload])])
        metadata_df.drop_duplicates(subset=["started_at"], inplace=True)
        metadata_df.to_csv(metadata_path, index=False)  # overwrite existing file with new data
    else:
        metadata_df = pd.DataFrame([payload])
    metadata_df.to_csv(metadata_path, index=False)

