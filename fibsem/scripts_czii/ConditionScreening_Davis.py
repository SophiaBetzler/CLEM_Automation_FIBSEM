from fibsem import acquire, utils, structures, calibration
from fibsem.microscope import ThermoMicroscope
from fibsem.structures import BeamType, FibsemStagePosition, FibsemDetectorSettings
from tkinter import messagebox
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import yaml
import pandas as pd
import sys
from PyQt5 import QtWidgets

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
        self.imaging_settings = structures.ImageSettings(
                                                autocontrast=False,
                                                resolution=[1536, 1024],
                                                beam_type=BeamType.ELECTRON,
                                                autogamma=False,
                                                save=True,
                                                filename='start',
                                                path=self.folder_path)
        project_root = Path(__file__).resolve().parent.parent
        config_file = os.path.join(self.project_root, 'config', 'ConditionScreening')
        self.imaging_settings.hfw = self.read_from_yaml(config_file, 'hfw')
        self.imaging_settings.dwell_time = self.read_from_yaml(config_file, 'dwell_time')
        self.imaging_settings.current = self.read_from_yaml(config_file, 'current')

    def read_from_yaml(self, filename, name):
        with open(f"{filename}.yaml", 'r') as file:
            data = yaml.safe_load(file)
        entry = float(data[name])
        if entry:
            return entry

    def set_starting_conditions(self, brightness, contrast):
        """
        Set the starting conditions for the experiment. It will first expose the sample to the electron beam for a
        pre-defined time to neutralize charges.
        brightness: pre-set brightness value which will not be changed during the experiment
        contrast: pre-set contrast value which will not be changed during the experiment
        return: the contrast of the 'initial image' and saves this image as 'Before.tiff'
        """
        calibration.auto_focus_beam(self.microscope, self.settings, beam_type=BeamType.ELECTRON)
        calibration.auto_charge_neutralisation(n_iterations=10, microscope=self.microscope, image_settings=self.imaging_settings)
        new_detector_settings = FibsemDetectorSettings(
            type='ETD',
            mode='SecondaryElectrons',
            brightness=brightness,
            contrast=contrast
        )
        self.microscope.set_detector_settings(detector_settings=new_detector_settings)

        image = acquire.new_image(self.microscope, self.imaging_settings)
        return np.std(image.data)

    def screen_conditions(self, voltages, stage_tilts, stage_biases):
        """
        Function to screen the conditions, defined to be most important for charging.
        Inputs are defined in the user-interface. It is expecting lists.
        """
        print(self.microscope.get_microscope_state())
        index = pd.MultiIndex.from_product(
            [voltages, stage_tilts, stage_biases],
            names=['voltage', 'stage_tilt', 'stage_bias']
        )
        def measure_contrast(voltage, stage_tilt, stage_bias):
            self.imaging_settings.voltage = voltage
            #Thermo = ThermoMicroscope()
            #Thermo.connection.beams.electron_beam.beam_deceleration.stage_bias.limits
            #Thermo.connection.beams.electron_beam.beam_deceleration.stage_bias.value = stage_bias
            stage_movement = FibsemStagePosition(x=float(0.0),
                                                 y=float(0.0),
                                                 z=float(0.0),
                                                 r=np.deg2rad(0.0),
                                                 t=np.deg2rad(stage_tilt))
            self.imaging_settings.save=False
            self.microscope.move_stage_relative(stage_movement)
            image = acquire.new_image(self.microscope, self.imaging_settings)
            return np.std(image.data)

        contrast_values = []
        prev_t = None
        for v, t, b in index:
            if t != prev_t:
                calibration.auto_focus_beam(self.microscope, self.settings, beam_type=BeamType.ELECTRON)  # Replace this with your actual function call
                prev_t = t  # Update the previous value
            contrast = measure_contrast(v, t, b)
            contrast_values.append(contrast)

        df_contrast_values = pd.DataFrame({'contrast': contrast_values}, index=index)
        max_index = df_contrast_values['contrast'].idxmax()
        print("RESULT---------------------------")
        print(f"The indices leading to the maximum contrast are a voltage of {max_index[0]}, "
              f"a relative stage tilt of {max_index[1]} and a stage_bias of {max_index[2]}.")
        print(df_contrast_values)
        return df_contrast_values

    def create_surface_plots(self, df_contrast_values, voltages, stage_tilts, stage_biases):
        for bias in stage_biases:
            df_subset = df_contrast_values.xs(bias, level='stage_bias')
            df_reset = df_subset.reset_index()

            # Pivot the table so that:
            # - Rows are stage_tilt values,
            # - Columns are voltage values,
            # - Values are contrast.
            df_pivot = df_reset.pivot(index='stage_tilt', columns='voltage', values='contrast')
            voltage_vals = df_pivot.columns.values
            stage_tilt_vals = df_pivot.index.values
            X, Y = np.meshgrid(voltage_vals, stage_tilt_vals)
            Z = df_pivot.values
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
            ax.set_xlabel('Voltage')
            ax.set_ylabel('Stage Tilt')
            ax.set_zlabel('Contrast')
            ax.set_title(f'Surface Plot at stage_bias = {bias}')
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()

class ParameterWindow(QtWidgets.QWidget):
    def __init__(self, fibsemcontrol):
        super().__init__()
        self.initUI()
        self.fibsem = fibsemcontrol

    def initUI(self):
        # Main layout for the window
        main_layout = QtWidgets.QVBoxLayout()
        # Display Settings group for brightness and contrast
        display_group = QtWidgets.QGroupBox("Display Settings")
        display_layout = QtWidgets.QFormLayout()
        self.brightness_input = QtWidgets.QLineEdit()
        self.contrast_input = QtWidgets.QLineEdit()
        display_layout.addRow("Brightness:", self.brightness_input)
        display_layout.addRow("Contrast:", self.contrast_input)
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)
        # Tilts group with min, max, and steps
        tilt_group = QtWidgets.QGroupBox("Stage Tilt Range")
        tilt_layout = QtWidgets.QFormLayout()
        self.tilt_min_input = QtWidgets.QLineEdit()
        self.tilt_max_input = QtWidgets.QLineEdit()
        self.tilt_steps_input = QtWidgets.QLineEdit()
        tilt_layout.addRow("Minimum:", self.tilt_min_input)
        tilt_layout.addRow("Maximum:", self.tilt_max_input)
        tilt_layout.addRow("Steps:", self.tilt_steps_input)
        tilt_group.setLayout(tilt_layout)

        # Biases group with min, max, and steps
        bias_group = QtWidgets.QGroupBox("Stage Bias Range")
        bias_layout = QtWidgets.QFormLayout()
        self.bias_min_input = QtWidgets.QLineEdit()
        self.bias_max_input = QtWidgets.QLineEdit()
        self.bias_steps_input = QtWidgets.QLineEdit()
        bias_layout.addRow("Minimum:", self.bias_min_input)
        bias_layout.addRow("Maximum:", self.bias_max_input)
        bias_layout.addRow("Steps:", self.bias_steps_input)
        bias_group.setLayout(bias_layout)

        # Voltages group with min, max, and steps
        voltage_group = QtWidgets.QGroupBox("Voltage Range")
        voltage_layout = QtWidgets.QFormLayout()
        self.voltage_min_input = QtWidgets.QLineEdit()
        self.voltage_max_input = QtWidgets.QLineEdit()
        self.voltage_steps_input = QtWidgets.QLineEdit()
        voltage_layout.addRow("Minimum:", self.voltage_min_input)
        voltage_layout.addRow("Maximum:", self.voltage_max_input)
        voltage_layout.addRow("Steps:", self.voltage_steps_input)
        voltage_group.setLayout(voltage_layout)

        # Submit button
        self.submit_button = QtWidgets.QPushButton("Submit")
        self.submit_button.clicked.connect(self.onSubmit)

        # Add groups and button to the main layout
        main_layout.addWidget(tilt_group)
        main_layout.addWidget(bias_group)
        main_layout.addWidget(voltage_group)
        main_layout.addWidget(self.submit_button)

        self.setLayout(main_layout)
        self.setWindowTitle("Parameter Setup")
        self.show()

    def onSubmit(self):
        # Retrieve values
        brightness = self.brightness_input.text()
        contrast = self.contrast_input.text()
        tilt_min = self.tilt_min_input.text()
        tilt_max = self.tilt_max_input.text()
        tilt_steps = self.tilt_steps_input.text()
        bias_min = self.bias_min_input.text()
        bias_max = self.bias_max_input.text()
        bias_steps = self.bias_steps_input.text()
        voltage_min = self.voltage_min_input.text()
        voltage_max = self.voltage_max_input.text()
        voltage_steps = self.voltage_steps_input.text()

        voltages = np.linspace(float(voltage_min), float(voltage_max), int(voltage_steps)).tolist()
        biases = np.linspace(float(bias_min), float(bias_max), int(bias_steps)).tolist()
        tilts = np.linspace(float(tilt_min), float(tilt_max), int(tilt_steps)).tolist()
        try:
            fibsem.set_starting_conditions(brightness=float(brightness), contrast=float(contrast))
            data_frame_screening = fibsem.screen_conditions(voltages, tilts, biases)
            fibsem.create_surface_plots(data_frame_screening, voltages, tilts, biases)
        except Exception as e:
            print(f"The screening failed: {e}")

if __name__ == "__main__":
    #create a folder for the experiment
    current_date = datetime.now().strftime("%Y%m%d")
    desktop_path = Path.home() / "Desktop/"
    folder_path = os.path.join(desktop_path, 'TestImages/', current_date)
    acquisition_time = datetime.now().strftime("%H-%M")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(os.path.join(folder_path + '/Temp')):
        os.makedirs(folder_path + '/Temp')
    app = QtWidgets.QApplication(sys.argv)
    fibsem = Fibsemcontrol(folder_path)
    gui = ParameterWindow(fibsem)
    fibsem.set_starting_conditions(brightness=0.7, contrast=0.7)
    df_contrast_values = fibsem.screen_conditions(voltages=[1500, 1750, 2000, 2250, 2500], stage_tilts=[-1, -0.5, 0.0, +0.5, +1.0], stage_biases=[0.5, 1.0, 1.5])

    sys.exit(app.exec())