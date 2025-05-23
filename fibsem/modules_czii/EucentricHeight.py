from fibsem import acquire, structures
from fibsem.structures import BeamType, FibsemStagePosition
from tkinter import messagebox
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import time
import cv2
import pandas as pd
import json
from Imaging import Imaging
import matplotlib.pyplot as plt

def error_message(text):
    messagebox.showerror("Error", text)

class EucentricHeight:
    """
    Class which containes functions related to the setup of the eucentric height and the beam coincidence.

    """
    def __init__(self, fib_microscope, bf):
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
        """
        Function measures the image shift between two images using the cross-correlation function implemented in openCV.
        """
        img1 = np.float32(image0.data)
        img2 = np.float32(image.data)
        shift, response = cv2.phaseCorrelate(img1, img2)
        return shift

    def measure_fiducial_position(self, image0, image1):
        """
        Function utilizes trained machine learning model (external python environment) to measure the shift between two
        images. It utilizes the current version of the trained model.
        """
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

    def shift_measurement_from_tilt(self, angle):
        """
        Function measures the shift between images recorded at different tilts. It uses both cross-correlation and
        fiducial position to do the measurement.
        Three images are recorded before, after tilt and after returning to the inital tilt.
        angle: relative tilt angle in degrees
        returns: shifts induced by the tilt and comparison
        """
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
        stage_movement_back = FibsemStagePosition(x=float(0.0),
                                             y=float(0.0),
                                             z=float(0.0),
                                             r=np.deg2rad(0.0),
                                             t=np.deg2rad(-angle))
        self.fib_microscope.move_stage_absolute(stage_movement_back)
        ms = datetime.now().microsecond // 1000
        self.imaging_settings.filename = datetime.now().strftime("%H-%M-%S") + f"-{ms:03d}_after"
        image_after = acquire.acquire_image(self.fib_microscope, self.imaging_settings)
        shift_before = self.measure_image_shift(image0, image_after)
        fiducial_shift_before = self.measure_fiducial_position(image0, image_after)
        return shift_tilt, shift_before, fiducial_shift_tilt, fiducial_shift_before

    def vertical_move(self, relative_z):
        """
        Function moves stage vertically.
        relative_z: requested relative move in z in micrometer.
        """
        stage_movement = FibsemStagePosition(x=float(0.0),
                                             y=float(0.0),
                                             z=float(relative_z/1.0e6),
                                             r=np.deg2rad(0.0),
                                             t=np.deg2rad(0.0))
        self.fib_microscope.move_stage_relative(stage_movement)

    def eucentric_height_tilt_series(self, z_shifts, tilts):
        """
        Automatic screening of the sample shifts for different z-heights and tilt ranges, to create reference data which
        ideally allow a more targeted determination of the eucentric height. Ideally, the eucentric height should be
        determined manually before running the script and then different Z-shifts are performed to determine the impact
        of the sample tilt on the sample shift.
        z_shifts: targeted relative z_shifts in micrometer
        tilts: relative tilt angles in degrees
        """
        shift_y_tilt = []
        shift_y_before = []
        shift_x_tilt = []
        shift_x_before = []
        fiducial_shift_x_tilt = []
        fiducial_shift_y_tilt = []
        fiducial_shift_x_before = []
        fiducial_shift_y_before = []
        indices = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['tilt', 'z-shift'])
        df_tilt = pd.DataFrame(columns=['shiftX', 'shiftY', 'fiducial_shiftX', 'fiducial_shiftY'], index=indices)
        df_zero = pd.DataFrame(columns=['shiftX', 'shiftY', 'fiducial_shiftX', 'fiducial_shiftY'], index=indices)
        for z_shift in z_shifts:
            self.vertical_move(z_shift)
            angles = tilts
            for angle in angles:
                shift_tilt, shift_before, fiducial_shift_tilt, fiducial_shift_before = self.shift_measurement_from_tilt(angle)
                shift_x_tilt.append(shift_tilt[0])
                shift_x_before.append(shift_before[0])
                shift_y_tilt.append(shift_tilt[1])
                shift_y_before.append(shift_before[1])
                fiducial_shift_x_tilt.append(fiducial_shift_tilt[0])
                fiducial_shift_y_tilt.append(fiducial_shift_tilt[1])
                fiducial_shift_x_before.append(fiducial_shift_before[0])
                fiducial_shift_y_before.append(fiducial_shift_before[1])
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

    def determine_eucentric_plane(self, measurements):
        """
        Function determines mathematical equation of the eucentric plane. The inputs are the x, y, z stage positions
        on the grid in mm. It outputs the equation for the plane which can be used to calculate the eucentric heights on
        the grid.
        measurements: array of [x, y, z] coordinates. At least three points are required.
        """
        positions_x = measurements[:, 0]
        positions_y = measurements[:, 1]
        positions_z = measurements[:, 2]
        position_matrix = np.column_stack((positions_x, positions_y, np.ones(measurements.shape[0])))
        coeff, residuals, rank, s = np.linalg.lstsq(position_matrix, positions_z, rcond=None)
        a, b, c = coeff
        positions_z_predicted = a * positions_x + b * positions_y + c
        squared_residuals = np.sum((positions_z - positions_z_predicted)**2)
        r_squared = 1 - squared_residuals / np.sum((positions_z - np.mean(positions_z))**2)
        print(f"The R^2 value for the fit of the eucentric plane is: {r_squared}.")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions_x, positions_y, positions_z, color='red')
        x_range = np.linspace(positions_x.min(), positions_x.max(), 10)
        y_range = np.linspace(positions_y.min(), positions_y.max(), 10)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = a * xx + b * yy + c
        ax. plot_surface(xx, yy, zz, alpha=0.5, color='blue')
        plt.show()
        return [a, b, c], r_squared


#### MAKE SURE TO INSTALL PANDAS AND xlsxwriter!

