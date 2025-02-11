from fibsem import acquire, utils, microscope, structures, milling
from fibsem.structures import BeamType, FibsemStagePosition
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

    def convert_txt_to_yaml(self, filename_milling, milling_protocol):

        def insert_nested_dict(data, key_path, value):
            keys = key_path.split(".")
            d = data
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value
            print(f"The value is {d}.")

        def txt_to_yaml(txt_filename, yaml_filename):
            dictionary = {}

            keys_convert_to_float = ['width', 'height', 'depth', 'rotation', 'centre_x', 'centre_y', 'milling_current',
                                     'milling_voltage', 'hfw']
            keys_convert_to_str = ['scan_direction', 'cross_section', 'patterning_mode', 'application_file']

            with open(txt_filename, "r") as txt_file:
                for line in txt_file:
                    if ":" in line:
                        key_path, value = line.strip().split(":", 1)
                        value = value.strip()
                        if "." in key_path:
                            key = key_path.split(".", 1)[1]
                        else:
                            key = key_path
                        if key in keys_convert_to_float:
                            insert_nested_dict(dictionary, key_path.strip(), float(value))
                        elif key in keys_convert_to_str:
                            insert_nested_dict(dictionary, key_path.strip(), str(value))
                        else:
                            insert_nested_dict(dictionary, key_path.strip(), value)

            with open(yaml_filename, "w") as yaml_file:
                yaml.dump(dictionary, yaml_file, default_flow_style=False, sort_keys=False)

        txt_to_yaml(os.path.join(self.folder_path + '/' + filename_milling), os.path.join(self.folder_path +  '/Temp/' + milling_protocol))

    def read_from_dict(self, filename):
        """
        The user should create txt files for his experiment conditions which are then converted to dicts and used
        to set the milling/imaging conditions.
        """
        dictionary = {}
        with open(os.path.join(self.folder_path + '/' + filename), 'r') as file:
            for line in file:
                if ":" in line:  # Ensure it's a key-value pair
                    key, value = line.strip().split(":", 1)  # Split on first ":"
                    dictionary[key.strip()] = value.strip()
        keys_convert_to_float = ['milling_current', 'line_integration', 'frame_integration', 'spacing',
                                 'spot_size', 'rate', 'milling_voltage', 'dwell_time', 'hfw', 'voltage',
                                 'working_distance', 'beam_current', 'scan_rotation', 'centre_x', 'centre_y',
                                 'depth', 'rotation', 'width', 'height', 'passes', 'time']
        keys_convert_to_int = ['frame_integration', 'line_integration']
        keys_convert_to_bool = ['autocontrast', 'autogamma', 'save', 'drift_correction', 'reduced_area', 'is_exclusion']
        keys_convert_to_points = ['stigmation', 'shift']
        keys_convert_to_object = ['cross_section']

        ###########I AM LOOKING AT THIS PROBLEM HERE #####################
        dict_pattern_type = {
            'Rectangle': structures.CrossSectionPattern.Rectangle.name,
            'Regular': structures.CrossSectionPattern.RegularCrossSection.name,
            'CleaningCrossSection': structures.CrossSectionPattern.CleaningCrossSection.name
        }
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
                dictionary[key] = {'x': float(dictionary.pop(f"{key}X")), 'y': float(dictionary.pop(f"{key}Y"))}
        if 'resolution' in dictionary:
            dictionary['resolution'] = ast.literal_eval(dictionary['resolution'])
        for key in keys_convert_to_object:
            if key in dictionary:
                dictionary[key] = dict_pattern_type[dictionary[key]]
        for key in keys_convert_to_int:
            if key in dictionary:
                dictionary[key] = int(dictionary[key])
        return dictionary

    def acquire_image(self, key):
        '''
        This function connects to the buttons in the GUI. It allows to take electron beam, ion beam and electron and ion
        beam images. TO DO: Add ability to take fluorescence images.
        Currently, no auto-focusing is done. But should be added in the future.
        key = 'electron', 'ion', 'both'
        Data are saved to the hard drive.
        '''

        dict_beamtypes = {
                'electron': BeamType.ELECTRON,
                'ion': BeamType.ION,
                'both': None,
                #'fl': BeamType.FL
                        }
        self.settings.image.path = self.folder_path
        plt.ion()  # needed to avoid the QCoreApplication::exec: The event loop is already running error
        if key == 'electron' or key == 'ion':
            fixed_parameters = {
                'filename': acquisition_time,
                'beam_type': dict_beamtypes[key],
                'path': self.folder_path,
            }
            try:
                filename_imaging = rf"imaging_{key}.txt"
                beam_settings = structures.BeamSettings.from_dict(self.read_from_dict(filename_imaging),
                                                                  beam_type=fixed_parameters['beam_type'])
                beam_system_settings = structures.BeamSystemSettings(beam_type=fixed_parameters['beam_type'],
                                                                     beam=getattr(self.settings.system, key).beam,
                                                                     detector=getattr(self.settings.system,
                                                                                      key).detector,
                                                                     eucentric_height=getattr(self.settings.system,
                                                                                              key).eucentric_height,
                                                                     column_tilt=getattr(self.settings.system,
                                                                                         key).column_tilt,
                                                                     enabled=getattr(self.settings.system, key).enabled
                                                                     )
                beam_system_settings.plasma = self.read_from_dict(filename_imaging)['plasma']
                beam_system_settings.plasma_gas = self.read_from_dict(filename_imaging)['plasma_source']
                imaging_settings = structures.ImageSettings.from_dict(self.read_from_dict(filename_imaging))

                for key1, value in fixed_parameters.items():
                    setattr(imaging_settings, key1, value)
                image = acquire.new_image(self.microscope, imaging_settings)
                plt.imshow(image.data, cmap='gray')
            except Exception as e:
                print(f"Image acquisition failed: {e}")
        elif key == 'both':
            try:
                image_eb, image_ion = acquire.take_reference_images(self.microscope, self.settings.image)
                self.settings.image.filename = acquisition_time + '_' + 'ebeam_ion'
                fig, ax = plt.subplots(1, 2, figsize=(10, 7))
                ax[0].imshow(image_eb.data, cmap="gray")
                ax[0].set_title("Electron Image")
                ax[1].imshow(image_ion.data, cmap="gray")
                ax[1].set_title("Ion Image")
            except Exception as e:
                print(f"Image acquisition failed: {e}.")
        else:
                return
        plt.show()
        return image


    def get_stage_position(self):
        return  self.microscope.get_stage_position()

    def move_stage(self, new_stage_position=None, mode=None, preset_stage_position=None):
        """
        Move the stage, either to user defined values or to stored stage positions which can be found in the yaml file.
        new_stage_position: list of values for the new stage position
        mode: either 'absolute' or 'relative'
        preset_stage_position: is the name of the preset stage position it is always automatically connected to
                                mode='absolute'
        """
        global stage_movement
        if new_stage_position and preset_stage_position is not None:
            raise ValueError(
                "Only one of 'new_stage_position' or 'preset_stage_position' can be provided, not both.")
        elif new_stage_position is None and preset_stage_position is None:
            raise ValueError(
                "Either 'new_stage_position' or 'preset_stage_position' must be provided.")
        else:
            if new_stage_position is not None:
                if mode is None:
                    raise ValueError("Mode can not be none, but must be absolute or relative.")
                else:
                    stage_movement = FibsemStagePosition(x=float(new_stage_position[0]) * 1e-3,
                                                         y=float(new_stage_position[1]) * 1e-3,
                                                         z=float(new_stage_position[2]) * 1e-3,
                                                         r=np.deg2rad(float(new_stage_position[3])),
                                                         t=np.deg2rad(float(new_stage_position[4])))
            elif preset_stage_position is not None:
                mode = 'absolute'
                with open(os.path.join(self.project_root, 'config', 'czii-stored-stage-positions.yaml'), "r") as file:
                    data = yaml.safe_load(file)
                preset_stage_position_values = next((entry for entry in data if entry["name"] == preset_stage_position),
                                                    None)
                stage_movement = FibsemStagePosition(x=float(preset_stage_position_values['x']),
                                                     y=float(preset_stage_position_values['y']),
                                                     z=float(preset_stage_position_values['z']),
                                                     r=np.deg2rad(float(preset_stage_position_values['r'])),
                                                     t=np.deg2rad(float(preset_stage_position_values['t'])))
            try:
                 if mode == 'relative':
                    self.microscope.move_stage_relative(stage_movement)
                 elif mode == 'absolute':
                     self.microscope.move_stage_absolute(stage_movement)
                 else:
                        return
            except Exception as e:
                print(f"The stage movement failed: {e}")
        print(f"The current stage position is {self.microscope.get_stage_position()}.")

    def create_fiducials(self, centerX=0, centerY=0):
        """
        Creates a fiducial based on the settings stored in the milling.txt file
        """
        filename_milling = 'milling.txt'
        self.convert_txt_to_yaml(filename_milling, 'milling_protocol.yaml')
        PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), os.path.join(self.folder_path + "/Temp/milling_protocol.yaml"))
        _, milling_settings = utils.setup_session(protocol_path=PROTOCOL_PATH)
        self.move_stage(preset_stage_position=milling_settings.protocol['fiducial']['stage_position'])
        try:
            rectangle_pattern_1 = structures.FibsemRectangleSettings.from_dict(milling_settings.protocol['fiducial'])
            rectangle_pattern_2 = rectangle_pattern_1
            rectangle_pattern_2.rotation = -rectangle_pattern_1.rotation
        except Exception as e:
            print(f"The fiducial creation failed because of {e}.")
        image = self.acquire_image('ion')
        milling_settings = structures.FibsemMillingSettings.from_dict(self.read_from_dict(filename_milling))
        ####I AM HERE ###############



        try:
            milling_stage = milling.FibsemMillingStage(
                name="Custom Milling Stage",
                num=1,
                alignment=milling.MillingAlignment(enabled=True),
                milling=milling_settings,
                imaging=imaging_settings,
                pattern=rectangle_pattern_1,
            )
        except Exception as e:
            print(f"Setting the milling stages failed: {e}")
        try:
            milling.patterning.plotting.draw_milling_patterns(image, [milling_stage])
            print('Successfully drew the milling pattern.')
        except Exception as e:
            print(f"The milling pattern was not set: {e}")
        try:
            filename_milling = 'milling.txt'
            milling_settings = structures.FibsemMillingSettings.from_dict(self.read_from_dict(filename_milling))

            print(f"Milling setup finished.")
        except Exception as e:
            print(f"The milling setup failed: {e}")
        try:

            milling.setup_milling(self.microscope, milling_settings)

            #milling.run_milling(self.microscope, milling_current=milling_settings.milling_current,
            #                                     milling_voltage=milling_settings.milling_voltage)
            self.acquire_image("ion")
        except Exception as e:
            print(f"The milling failed: {e}")

        # print(f"The estimated milling time is {milling.estimate_milling_time(self.microscope, [rectangle_pattern_1, rectangle_pattern_2])}.")
        # milling.setup_milling(self.microscope, ion_milling_settings)
        # milling.run_milling(self.microscope, ion_milling_settings.milling_voltage, ion_milling_settings.milling_current)
        # print(f"Milling finished.")
        # self.acquire_image('ion')
        # milling.finish_milling(self.microscope, imaging_current=ion_imaging_settings.milling_current,
        #                        imaging_voltage=ion_imaging_settings.milling_voltage)

class Gui(QWidget):
    '''
    GUI to allow the user to access simple functions like image acquisition and stage movement.
    Developed for testing purposes.
    Available functions:
            -   acquire and save ebeam and ibeam images folder is created on desktop (date) and files are named based on
                acquisition time
            -   stage movement to absolute and relative coordinates in the RAW coordinate system
    '''

    def __init__(self, fibsemcontrol):
        super().__init__()
        self.fibsem = fibsemcontrol

        # Create a vertical layout of buttons to acquire images
        acquire_button_layout = QVBoxLayout()
        self.button_eb = QPushButton("Electron Beam Image")
        self.button_eb.clicked.connect(lambda: self.fibsem.acquire_image('electron'))
        acquire_button_layout.addWidget(self.button_eb)
        self.button_ion = QPushButton("Ion Beam Image")
        self.button_ion.clicked.connect(lambda: self.fibsem.acquire_image('ion'))
        acquire_button_layout.addWidget(self.button_ion)
        self.button_both = QPushButton("Electron and Ion Beam Image")
        self.button_both.clicked.connect(lambda: self.fibsem.acquire_image('both'))
        acquire_button_layout.addWidget(self.button_both)
        self.button_fl = QPushButton("FL Image")
        self.button_fl.setEnabled(False)
        self.button_fl.clicked.connect(lambda: self.fibsem.acquire_image('fl'))
        acquire_button_layout.addWidget(self.button_fl)

        # Create a horizontal layout of input boxes for stage movement
        relative_input_box_layout = QHBoxLayout()
        column_descriptors = ["X (mm)", "Y (mm)", "Z (mm)", "Rotation (deg)", "Tilt (deg)"]
        self.relative_move_inputs = []
        for desc in column_descriptors:
            column_layout = QVBoxLayout()
            label = QLabel(desc, self)
            #label.setAlignment(Qt.AlignCenter)  # Center-align the text above the box
            column_layout.addWidget(label)
            input_box = QLineEdit(self)
            input_box.setPlaceholderText("Enter value")
            column_layout.addWidget(input_box)
            relative_input_box_layout.addLayout(column_layout)
            self.relative_move_inputs.append(input_box)
        relative_move_button = QPushButton("Relative", self)
        relative_move_button.clicked.connect(self.relative_stage_movement)
        relative_input_box_layout.addWidget(relative_move_button)
        absolute_input_box_layout = QHBoxLayout()
        self.absolute_move_inputs = []
        for _ in range(5):
            input_box = QLineEdit(self)
            input_box.setPlaceholderText("Enter value")
            absolute_input_box_layout.addWidget(input_box)
            self.absolute_move_inputs.append(input_box)

        absolute_move_button = QPushButton("Absolute", self)
        absolute_move_button.clicked.connect(self.absolute_stage_movement)
        absolute_input_box_layout.addWidget(absolute_move_button)
        acquire_button_layout.addLayout(relative_input_box_layout)
        acquire_button_layout.addLayout(absolute_input_box_layout)

        milling_layout = QHBoxLayout()
        fiducial_button = QPushButton("Fiducial", self)
        fiducial_button.clicked.connect(fibsem.create_fiducials)
        milling_layout.addWidget(fiducial_button)
        stage_position_button = QPushButton("Stage Position", self)
        stage_position_button.clicked.connect(fibsem.get_stage_position)
        milling_layout.addWidget(stage_position_button)
        acquire_button_layout.addLayout(milling_layout)

        # Set the main layout for the widget
        self.setLayout(acquire_button_layout)

    def relative_stage_movement(self):
        # Collect the settings for the relative stage movement
        values = [input_box.text() for input_box in self.relative_move_inputs]
        fibsem.move_stage(values, 'relative')

    def absolute_stage_movement(self):
        # Collect data from the second row
        values = [input_box.text() for input_box in self.absolute_move_inputs]
        fibsem.move_stage(values, 'absolute')

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
    app = QApplication(sys.argv)
    fibsem = Fibsemcontrol(folder_path)
    gui = Gui(fibsem)
    gui.show()
    sys.exit(app.exec())

