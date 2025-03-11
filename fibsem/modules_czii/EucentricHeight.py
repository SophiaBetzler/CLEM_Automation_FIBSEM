from fibsem import acquire, utils, microscope, structures, milling, calibration, gis
from fibsem.structures import BeamType, FibsemStagePosition, FibsemDetectorSettings
from tkinter import messagebox
from Basic_Functions import BasicFunctions
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
import json
from Imaging import Imaging

def error_message(text):
    messagebox.showerror("Error", text)

class EucentricHeight():
    """
    Class which incorporates every function needed for FIB SEM Control.
    Class calls functions from openFIBSEM to control the microscope.

    """
    def __init__(self, fib_microscope, bf):
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
        self.bf = bf
        self.fib_microscope = fib_microscope
        self.project_root = Path(__file__).resolve().parent.parent
        self.imaging_settings, self.imaging_dict = self.bf.read_from_yaml\
            (os.path.join(self.project_root, 'config', 'EucentricHeight'))
        self.imaging = Imaging(self.fib_microscope, bf=self.bf, imaging_settings=self.imaging_settings)
        self.beam_settings = structures.BeamSettings.from_dict(self.imaging_dict)
        self.fib_microscope.set_beam_settings(self.beam_settings)
        self.imaging_settings.beam_type = BeamType.ELECTRON
        self.beam_settings = self.bf.read_from_yaml(os.path.join(self.project_root, 'config', 'EucentricHeight'))
        self.folder_path = self.bf.folder_path
        self.imaging_settings.path = self.folder_path
        self.imaging_settings.hfw = 600.0e-6
        self.imaging_settings.save = True
        self.temp_folder_path = self.bf.temp_folder_path

    def measure_image_shift(self, image0, image):
        img1 = np.float32(image0.data)
        img2 = np.float32(image.data)
        shift, response = cv2.phaseCorrelate(img1, img2)
        return shift

    def measure_fiducial_position(self, image0, image1):
        image0.save(os.path.join(self.temp_folder_path, 'image0.tif'))
        self.bf.execute_external_script(script='Identify_Fiducial_Remote.py',
                                        dir_name='Ultralytics',
                                        parameter=1)

        while not os.path.exists(os.path.join(self.temp_folder_path, 'fiducial_id_result.json')):
            time.sleep(1)

        with open(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'), 'r') as file:
            fiducial_id_result = json.load(file)

        fiducial_image0 = [float(fiducial_id_result['centerX']),
                           float(fiducial_id_result['centerY'])]

        os.remove(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'))

        image1.save(os.path.join(self.temp_folder_path, 'image.tif'))

        self.bf.execute_external_script(script='Identify_Fiducial_Remote.py',
                                        dir_name='Ultralytics',
                                        parameter=1)

        while not os.path.exists(os.path.join(self.temp_folder_path, 'fiducial_id_result.json')):
            time.sleep(1)

        with open(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'), 'r') as file:
            fiducial_id_result = json.load(file)

        fiducial_image1 = [float(fiducial_id_result['centerX']),
                           float(fiducial_id_result['centerY'])]

        os.remove(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'))

        shift=[fiducial_image0[0]-fiducial_image1[0], fiducial_image0[1]-fiducial_image1[1]]

        return shift

    def tilting_imaging(self, angle):
        before_stage_position = self.fib_microscope.get_stage_position()
        ms = datetime.now().microsecond // 1000
        self.imaging_settings.filename = datetime.now().strftime("%H-%M-%S")+f"-{ms:03d}_before"
        image0 = acquire.new_image(self.fib_microscope, self.imaging_settings)
        stage_movement = FibsemStagePosition(x=float(0.0),
                                             y=float(0.0),
                                             z=float(0.0),
                                             r=np.deg2rad(0.0),
                                             t=np.deg2rad(angle))
        self.fib_microscope.move_stage_relative(stage_movement)
        ms = datetime.now().microsecond // 1000
        self.imaging_settings.filename = datetime.now().strftime("%H-%M-%S") + f"-{ms:03d}_{angle}"
        image = acquire.acquire_image(self.fib_microscope, self.imaging_settings)
        shift_tilt = self.measure_image_shift(image0, image)
        fiducial_shift_tilt = self.measure_fiducial_position(image0, image)
        self.fib_microscope.move_stage_absolute(before_stage_position)
        ms = datetime.now().microsecond // 1000
        self.imaging_settings.filename = datetime.now().strftime("%H-%M-%S") + f"-{ms:03d}_after"
        image_after = acquire.acquire_image(self.fib_microscope, self.imaging_settings)
        shift_before = self.measure_image_shift(image0, image_after)
        fiducial_shift_before = self.measure_fiducial_position(image0, image_after)
        return shift_tilt, shift_before, fiducial_shift_tilt, fiducial_shift_before

    def vertical_move(self, relative_z):
        stage_movement = FibsemStagePosition(x=float(0.0),
                                             y=float(0.0),
                                             z=float(relative_z),
                                             r=np.deg2rad(0.0),
                                             t=np.deg2rad(0.0))
        self.fib_microscope.move_stage_relative(stage_movement)
        print(f"The current stage position is {self.fib_microscope.get_stage_position()}")

    def eucentric_height_tilt_series(self, z_shifts, tilts):
        shiftY_tilt = []
        shiftY_before = []
        shiftX_tilt = []
        shiftX_before = []
        fiducial_shiftX_tilt = []
        fiducial_shiftY_tilt = []
        fiducial_shiftX_before = []
        fiducial_shiftY_before = []
        indices = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['tilt', 'z-shift'])
        df_tilt = pd.DataFrame(columns=['shiftX', 'shiftY', 'fiducial_shiftX', 'fiducial_shiftY'], index=indices)
        df_zero = pd.DataFrame(columns=['shiftX', 'shiftY', 'fiducial_shiftX', 'fiducial_shiftY'], index=indices)
        for z_shift in z_shifts:
            self.vertical_move(z_shift)
            angles = tilts
            for angle in angles:
                shift_tilt, shift_before, fiducial_shift_tilt, fiducial_shift_before = self.tilting_imaging(angle)
                shiftX_tilt.append(shift_tilt[0])
                shiftX_before.append(shift_before[0])
                shiftY_tilt.append(shift_tilt[1])
                shiftY_before.append(shift_before[1])
                fiducial_shiftX_tilt.append(fiducial_shift_tilt[0])
                fiducial_shiftY_tilt.append(fiducial_shift_tilt[1])
                fiducial_shiftX_before.append(fiducial_shift_before[0])
                fiducial_shiftY_before.append(fiducial_shift_before[1])
                new_row_tilt = pd.DataFrame(
                    {'shiftX': [shift_tilt[0]], 'shiftY': [shift_tilt[1]],
                     'fiducial_shiftX': [fiducial_shift_tilt[0]], 'fiducial_shiftY': [fiducial_shift_tilt[1]]},
                    index=pd.MultiIndex.from_tuples([(str(angle), str(z_shift))], names=['tilt', 'z-shift']))
                new_row_zero = pd.DataFrame(
                    {'shiftX': [shift_before[0]], 'shiftY': [shift_before[1]],
                     'fiducial_shiftX': [fiducial_shift_before[0]], 'fiducial_shiftY': [fiducial_shift_before[1]]},
                    index=pd.MultiIndex.from_tuples([(str(angle), str(z_shift))], names=['tilt', 'z-shift']))
                df_tilt = pd.concat([df_tilt, new_row_tilt])
                df_zero = pd.concat([df_zero, new_row_zero])
            self.vertical_move(-z_shift)
        with pd.ExcelWriter(self.folder_path + '/EucentricHeight.xlsx', engine='xlsxwriter') as writer:
            df_tilt.to_excel(writer, sheet_name='Tilted_Image_Comparison', index=True)
            df_zero.to_excel(writer, sheet_name='Zero_Tile_Image_Comparison', index=True)












#### MAKE SURE TO INSTALL PANDAS AND xlsxwriter!

