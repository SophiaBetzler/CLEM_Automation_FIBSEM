import subprocess
import os
from pathlib import Path
from datetime import datetime
from fibsem import utils, structures
from tkinter import messagebox
import sys
import ast
import yaml
import numpy as np


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
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(os.path.join(folder_path + '/Temp')):
        os.makedirs(folder_path + '/Temp')
    return os.path.join(folder_path, 'Temp'), folder_path


class BasicFunctions:
    """
    This class summarizes the basic functions needed as foundation for many other modules.
    It includes the read-in of the textfiles, allows to execute scripts from another
    virtual environment, connects to the microscope.
    """
    temp_folder_path, folder_path = create_temp_folder()

    def __init__(self, pc_type='windows', manufacturer='Demo', ip='localhost', tool='Hydra'):
        create_temp_folder()
        self.project_root = Path(__file__).resolve().parent.parent
        self.python_root = Path(__file__).resolve().parent.parent.parent.parent
        self.manufacturer = manufacturer
        self.ip = ip
        self.tool = tool
        self.pc_type = pc_type

    def execute_external_script(self, script, dir_name, parameter=None):
        """
        This function executes a 'script.py' function from a different directory.
        script: script name as str
        dir_name: dir_name as str
        """
        if self.pc_type == 'mac':
            python_path = os.path.join(self.python_root, dir_name, 'venv/bin/python')
        elif self.pc_type == 'windows':
            python_path = os.path.join(self.python_root, dir_name, 'venv\Scripts\python')
        venv_path = os.path.join(self.python_root, dir_name)
        script_path = os.path.join(venv_path, script)
        if parameter is not None:
            result = subprocess.run(
                [python_path, script_path, '--temp_folder_path', self.temp_folder_path, '--parameter', str(parameter)],
                capture_output=True,
                text=True
            )
        else:
            result = subprocess.run([python_path, script_path],
                                        capture_output=True,
                                        text=True)
        return result

    def connect_to_microscope(self):
        """
        Establish connection to the microscope.
        manufacturer: 'Demo', 'Thermo', 'Tescan'
        ip: 'localhost', '192.168.0.1'
        tool: 'Hydra', 'Arctis'
        """
        if self.tool == 'Hydra':
            config_path = os.path.join(self.project_root, 'config', 'czii-tfs-hydra-configuration.yaml')
        elif self.tool == 'Arctis':
            config_path = os.path.join(self.project_root, 'config', 'tfs-arctis-configuration.yaml')
        else:
            raise ValueError("No valid tool selected. Options are Hydra and Arctis")

        try:
            fib_microscope, fib_settings = utils.setup_session(manufacturer=self.manufacturer,
                                                               ip_address=self.ip,
                                                               config_path=config_path)
            return fib_microscope, fib_settings
        except Exception as e:
            error_message(f"Connection to microscope failed: {e}")
            sys.exit()
    def read_from_yaml(self, filename):
        """
        User generated yaml file with the basic settings for imaging with the ion/electron beam.
        """
        with open(os.path.join(self.project_root, 'modules_czii', filename + '.yaml')) as file:
            dictionary = yaml.safe_load(file)
        # if 'reduced_area' in dictionary or dictionary['reduced_area'] is not None:
        #     dictionary['reduced_area'] = {
        #                                     "left": dictionary['reduced_area'][0],
        #                                     "top": dictionary['reduced_area'][1],
        #                                     "width": dictionary['reduced_area'][2],
        #                                     "height": dictionary['reduced_area'][3],
        #                                 }


        return dictionary

    def read_from_dict(self, filename):
        """
        The user should create txt files for his experiment conditions which are then converted to dicts and used
        to set the milling/imaging conditions.
        """
        dictionary = {}
        with open(os.path.join(self.folder_path , filename + '.txt'), 'r') as file:
            for line in file:
                if ":" in line:  # Ensure it's a key-value pair
                    key, value = line.strip().split(":", 1)  # Split on first ":"
                    dictionary[key.strip()] = value.strip()
        keys_convert_to_float = ['milling_current', 'milling_voltage', 'line_integration', 'frame_integration',
                                 'spacing',
                                 'spot_size', 'rate', 'milling_voltage', 'dwell_time', 'hfw', 'voltage',
                                 'working_distance', 'beam_current', 'center_x', 'center_y',
                                 'depth', 'rotation', 'width', 'height', 'passes', 'time']
        keys_convert_to_int = ['frame_integration', 'line_integration']
        keys_convert_to_bool = ['autocontrast', 'autogamma', 'save', 'drift_correction', 'reduced_area',
                                'is_exclusion'
                                'aquire_image']
        keys_convert_to_points = ['stigmation', 'shift']
        str_to_bool = {"true": True, "false": False, "none": None}
        for key in keys_convert_to_bool:
            if key in dictionary:
                dictionary[key] = str_to_bool.get(dictionary[key].lower(), None)
        for key in keys_convert_to_float:
            if key in dictionary:
                if dictionary[key] is not None:
                    dictionary[key] = float(dictionary[key])
        for key in keys_convert_to_points:
            if any(f"{key}{suffix}" in dictionary for suffix in ["X", "Y"]):
                if key == 'Center':
                    dictionary['Point'] = {'x': float(dictionary.pop(f"{key}X")),
                                           'y': float(dictionary.pop(f"{key}Y"))}
                else:
                    dictionary[key] = {'x': float(dictionary.pop(f"{key}X")), 'y': float(dictionary.pop(f"{key}Y"))}
        if 'resolution' in dictionary:
            dictionary['resolution'] = ast.literal_eval(dictionary['resolution'])
        for key in keys_convert_to_int:
            if key in dictionary:
                dictionary[key] = int(dictionary[key])
        if 'scan_rotation' in dictionary:
            dictionary['scan_rotation'] = np.deg2rad(float(dictionary['scan_rotation']))
        return dictionary