from fibsem import acquire, utils, microscope, structures, milling, calibration, gis
from fibsem.structures import BeamType, FibsemStagePosition, FibsemDetectorSettings
from tkinter import messagebox
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import ast
import inspect
import time
import re
import yaml
import cv2
import pandas as pd

def error_message(text):
    messagebox.showerror("Error", text)

class Fibsemcontrol():
    """
    Class which incorporates every function needed for FIB SEM Control.
    Class calls functions from openFIBSEM to control the microscope.

    """
    def __init__(self, folder_path):
        """
        Establish connection to the microscope.
        Possible inputs for manufacturer:
                        Demo (for offline use)
                        Establish connection to the microscope.
                        "Thermo", "Thermo Fisher Scientific", "Thermo Fisher Scientific"
                        "Tescan", "TESCAN"
        session path: Directory to store the data.
        config_path: Directory to the microscope configuration file to be used.
        """
        self.project_root = Path(__file__).resolve().parent.parent
        self.folder_path = folder_path
        self.imaging_settings = self.read_from_yaml()
        self.imaging_settings.beam_type = BeamType.ELECTRON
        self.imaging_settings.filename = 'Default'
        self.imaging_settings.path = self.folder_path
        try:
            #for hydra microscope use:
            config_path = os.path.join(self.project_root, 'config', 'czii-tfs-hydra-configuration.yaml')
            #for arctis microscope use:
            #config_path = os.path.join(self.project_root, 'config', 'tfs-arctis-configuration.yaml')
            #self.microscope, self.settings = utils.setup_session(manufacturer='Thermo', ip_address='192.168.0.1',
            #                                                      config_path=config_path)
            #
            self.microscope, self.settings = utils.setup_session(manufacturer='Demo', ip_address='localhost',
                                                                  config_path=config_path)
            print(f"The settings are {self.settings}.")
            print(f"The microscope is {self.microscope}")
        except Exception as e:
            error_message(f"Connection to microscope failed: {e}")
            sys.exit()

    def read_from_yaml(self):
        """
        Read data from the yaml file. Return imaging setting object.
        """
        def get_all_keys(obj):
            keys = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    keys.append(key)
                    keys.extend(get_all_keys(value))
            elif isinstance(obj, list):
                for item in obj:
                    keys.extend(get_all_keys(item))
            return keys

        project_root = Path(__file__).resolve().parent.parent
        config_file = os.path.join(project_root, 'config', 'EucentricHeight')
        with open(f"{config_file}.yaml", 'r') as file:
            data = yaml.safe_load(file)

        parameters = get_all_keys(data)
        imaging_settings = structures.ImageSettings()
        for parameter in parameters:
            if parameter == 'scan_rotation':
                setattr(imaging_settings, parameter, np.deg2rad(data[parameter]))
            else:
                setattr(imaging_settings, parameter, data[parameter])
        return imaging_settings

    def measure_image_shift(self, image0, image):
        img1 = np.float32(image0.data)
        img2 = np.float32(image.data)
        shift, response = cv2.phaseCorrelate(img1, img2)
        return shift

    def tilting_imaging(self, angle):
        before_stage_position = self.microscope.get_stage_position()
        self.imaging_settings.save = True
        ms = datetime.now().microsecond // 1000
        self.imaging_settings.filename = datetime.now().strftime("%H-%M-%S")+ f"-{ms:03d}"
        image0 = acquire.new_image(self.microscope, self.imaging_settings)
        stage_movement = FibsemStagePosition(x=float(0.0),
                                             y=float(0.0),
                                             z=float(0.0),
                                             r=np.deg2rad(0.0),
                                             t=np.deg2rad(angle))
        self.microscope.move_stage_relative(stage_movement)
        image = acquire.new_image(self.microscope, self.imaging_settings)
        shift_tilt = self.measure_image_shift(image0, image)
        self.microscope.move_stage_absolute(before_stage_position)
        image_after = acquire.new_image(self.microscope, self.imaging_settings)
        shift_before = self.measure_image_shift(image0, image_after)
        return shift_tilt, shift_before

    def vertical_move(self, relative_z):
        stage_movement = FibsemStagePosition(x=float(0.0),
                                             y=float(0.0),
                                             z=float(relative_z),
                                             r=np.deg2rad(0.0),
                                             t=np.deg2rad(0.0))
        self.microscope.move_stage_relative(stage_movement)

if __name__ == "__main__":
    #create a folder for the experiment
    current_date = datetime.now().strftime("%Y%m%d")
    desktop_path = Path.home() / "Desktop/"
    folder_path = os.path.join(desktop_path, 'TestImages/', current_date)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fibsem = Fibsemcontrol(folder_path)
    shiftY_tilt = []
    shiftY_before = []
    shiftX_tilt = []
    shiftX_before = []
    indices = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['tilt', 'z-shift'])
    df_tilt = pd.DataFrame(columns=['shiftX', 'shiftY'], index=indices)
    df_zero = pd.DataFrame(columns=['shiftX', 'shiftY'], index=indices)
    z_shifts = [float(0.5e-3), float(1.0e-3), float(1.5e-3)]
    for z_shift in z_shifts:
        fibsem.vertical_move(z_shift)
        angles = [2, 1, -1, -2, -5]
        for angle in angles:
            shift_tilt, shift_before = fibsem.tilting_imaging(angle)
            print(shift_tilt[0])
            print(shift_before[1])
            shiftX_tilt.append(shift_tilt[0])
            shiftX_before.append(shift_before[0])
            shiftY_tilt.append(shift_tilt[1])
            shiftY_before.append(shift_before[1])
            new_row_tilt = pd.DataFrame(
                {'shiftX': [shift_tilt[0]], 'shiftY': [shift_tilt[1]]},
                index=pd.MultiIndex.from_tuples([(str(angle), str(z_shift))], names=['tilt', 'z-shift']))
            new_row_zero = pd.DataFrame(
                {'shiftX': [shift_before[0]], 'shiftY': [shift_before[1]]},
                index=pd.MultiIndex.from_tuples([(str(angle), str(z_shift))], names=['tilt', 'z-shift']))
            df_tilt = pd.concat([df_tilt, new_row_tilt])
            df_zero = pd.concat([df_zero, new_row_zero])
    with pd.ExcelWriter(folder_path + '/EucentricHeight.xlsx', engine='xlsxwriter') as writer:
        df_tilt.to_excel(writer, sheet_name='Tilted_Image_Comparison', index=False)
        df_zero.to_excel(writer, sheet_name='Zero_Tile_Image_Comparison', index=False)
    print(df_tilt)
    print(df_zero)

#### MAKE SURE TO INSTALL PANDAS AND xlsxwriter!

