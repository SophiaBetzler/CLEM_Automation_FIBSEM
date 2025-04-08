import subprocess
import os
from pathlib import Path
from datetime import datetime
import time
import json
from fibsem import structures, microscope, utils
import numpy as np
from tkinter import messagebox
import sys

def error_message(text):
    messagebox.showerror("Error", text)

def create_temp_folder():
    """
    This script creates a folder with the current date on the desktop. Inside a temp folder to
    temporarily store data.
    """
    current_date = datetime.now().strftime("%Y%m%d")
    desktop_path = Path.home() / "Desktop/"
    folder_path = os.path.join(desktop_path, 'TestImages/', current_date)
    acquisition_time = datetime.now().strftime("%H-%M")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(os.path.join(folder_path + '/Temp')):
        os.makedirs(folder_path + '/Temp')
    return os.path.join(folder_path, 'Temp')

def execute_external_script(script, dir_name):
    """
    This function executes a 'script.py' function from a different directory.
    script: script name as str
    dir_name: dir_name as str
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    python_path_ml = os.path.join(project_root, dir_name, 'venv/bin/python')
    venv_path_ml = os.path.join(project_root, dir_name)
    ml_model_python = os.path.join(python_path_ml)
    identify_fiducial = os.path.join(venv_path_ml, script)
    subprocess.run([ml_model_python, identify_fiducial],
                            capture_output=True, text=True)

def connect_to_microscope():
    project_root = Path(__file__).resolve().parent.parent

    try:
        # for hydra microscope use:
        config_path = os.path.join(project_root, 'config', 'czii-tfs-hydra-configuration.yaml')
        # for arctis microscope use:
        # config_path = os.path.join(self.project_root, 'config', 'tfs-arctis-configuration.yaml')
        # self.microscope, self.settings = utils.setup_session(manufacturer='Thermo', ip_address='192.168.0.1',
        #                                                      config_path=config_path)
        #
        fib_microscope, fib_settings = utils.setup_session(manufacturer='Demo', ip_address='localhost',
                                                             config_path=config_path)
        return fib_microscope, fib_settings
    except Exception as e:
        error_message(f"Connection to microscope failed: {e}")
        sys.exit()

def get_stage_position():
    current_stage_position = fib_microscope.get_stage_position()
    current_stage_position_adjusted_units = [current_stage_position.x*1e3,
                                              current_stage_position.y*1e3,
                                              current_stage_position.z*1e3,
                                              np.rad2deg(current_stage_position.r),
                                              np.rad2deg(current_stage_position.t)]
    print(f"The current stage position is {current_stage_position_adjusted_units}.")

temp_folder = create_temp_folder()
execute_external_script('Identify_Fiducial.py', 'Ultralytics')
fib_microscope, settings = connect_to_microscope()

while not os.path.exists(os.path.join(temp_folder, 'stage_move.json')):
    time.sleep(1)

with open(os.path.join(temp_folder, 'stage_move.json'), 'r') as file:
    stage_move = json.load(file)

if os.path.exists(os.path.join(temp_folder, 'stage_move.json')):
    os.remove(os.path.join(temp_folder, 'stage_move.json'))

get_stage_position()
beam_type = structures.BeamType.ELECTRON
fib_microscope.stable_move(stage_move['move_X'], stage_move['move_Y'], beam_type)
get_stage_position()
