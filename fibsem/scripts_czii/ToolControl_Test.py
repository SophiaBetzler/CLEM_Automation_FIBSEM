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
        keys_convert_to_float = ['milling_current', 'milling_voltage', 'line_integration', 'frame_integration', 'spacing',
                                 'spot_size', 'rate', 'milling_voltage', 'dwell_time', 'hfw', 'voltage',
                                 'working_distance', 'beam_current', 'center_x', 'center_y',
                                 'depth', 'rotation', 'width', 'height', 'passes', 'time']
        keys_convert_to_int = ['frame_integration', 'line_integration']
        keys_convert_to_bool = ['autocontrast', 'autogamma', 'save', 'drift_correction', 'reduced_area', 'is_exclusion'
                                'aquire_image']
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
                if key == 'Center':
                    dictionary['Point'] = {'x': float(dictionary.pop(f"{key}X")), 'y': float(dictionary.pop(f"{key}Y"))}
                else:
                    dictionary[key] = {'x': float(dictionary.pop(f"{key}X")), 'y': float(dictionary.pop(f"{key}Y"))}
        if 'resolution' in dictionary:
            dictionary['resolution'] = ast.literal_eval(dictionary['resolution'])
        for key in keys_convert_to_object:
            if key in dictionary:
                dictionary[key] = dict_pattern_type[dictionary[key]]
        for key in keys_convert_to_int:
            if key in dictionary:
                dictionary[key] = int(dictionary[key])
        if 'scan_rotation' in dictionary:
            dictionary['scan_rotation'] = np.deg2rad(float(dictionary['scan_rotation']))
        return dictionary

    def read_from_yaml(self, filename, name):
        with open(f"{filename}.yaml", 'r') as file:
            data = yaml.safe_load(file)
        entry = next((item for item in data if item['name'] == name), None)
        if entry:
            return entry

    def auto_focus_image(self):
        calibration.auto_focus_beam(self.microscope, self.settings, beam_type=BeamType.ELECTRON)

    def acquire_image(self, key):
        '''
        This function connects to the buttons in the GUI. It allows to take electron beam, ion beam and electron and ion
        beam images. TO DO: Add ability to take fluorescence images.
        Currently, no auto-focusing is done. But should be added in the future.
        key = 'electron', 'ion', 'both'
        Data are saved to the hard drive.
        '''

        new_detector_settings = FibsemDetectorSettings(
            type='ETD',
            mode='SecondaryElectrons',
            brightness=0.8,
            contrast=0.7
        )

        self.microscope.set_detector_settings(detector_settings=new_detector_settings)
        print(self.microscope.get_microscope_state())

        dict_beamtypes = {
                'electron': BeamType.ELECTRON,
                'ion': BeamType.ION,
                'both': BeamType.ION,
                'multiple': BeamType.ELECTRON,
                'tiles': BeamType.ELECTRON,
                #'fl': BeamType.FL
                        }

        self.settings.image.path = self.folder_path
        plt.ion()  # needed to avoid the QCoreApplication::exec: The event loop is already running error
        filename_milling = rf"milling_base.txt"
        fixed_parameters = {
            'filename': acquisition_time,
            'beam_type': dict_beamtypes[key],
            'path': self.folder_path,
        }
        if key == 'electron' or key == 'ion':
            try:
                filename_imaging = rf"imaging_{key}.txt"
                beam_settings = structures.BeamSettings.from_dict(self.read_from_dict(filename_imaging),
                                                                  beam_type=fixed_parameters['beam_type'])
                self.microscope.set("plasma_gas",self.read_from_dict(filename_milling)['plasma_source'], beam_type=BeamType.ION)
                self.microscope.set("plasma", self.read_from_dict(filename_milling)['plasma'], beam_type=BeamType.ION)
                self.microscope.set_beam_settings(beam_settings)
                imaging_settings = structures.ImageSettings.from_dict(self.read_from_dict(filename_imaging))
                for key1, value in fixed_parameters.items():
                    setattr(imaging_settings, key1, value)
                image = acquire.new_image(self.microscope, imaging_settings)
                plt.imshow(image.data, cmap='gray')
            except Exception as e:
                print(f"Image acquisition failed: {e}")
        elif key == 'both':
            try:
                key_both = 'ion'
                beam_settings = structures.BeamSettings.from_dict(self.read_from_dict(rf"imaging_{key_both}.txt"),
                                                                  beam_type=BeamType.ION)
                self.microscope.set("plasma_gas",self.read_from_dict(filename_milling)['plasma_source'], beam_type=fixed_parameters['beam_type'])
                self.microscope.set("plasma", self.read_from_dict(filename_milling)['plasma'], beam_type=fixed_parameters['beam_type'])
                self.microscope.set_beam_settings(beam_settings)
                imaging_settings = structures.ImageSettings.from_dict(self.read_from_dict(rf"imaging_{key_both}.txt"))
                imaging_settings.filename = acquisition_time
                imaging_settings.path = self.folder_path
                image_eb, image_ion = acquire.take_reference_images(self.microscope, imaging_settings)
                fig, ax = plt.subplots(1, 2, figsize=(10, 7))
                ax[0].imshow(image_eb.data, cmap="gray")
                ax[0].set_title("Electron Image")
                ax[1].imshow(image_ion.data, cmap="gray")
                ax[1].set_title("Ion Image")
            except Exception as e:
                print(f"Image acquisition failed: {e}.")
        elif key == 'multiple':
            hfws = [float(80e-6), float(150e-6), float(400e-6), float(900e-6)]
            key_multiple = 'electron'
            try:
                contrast = {}
                imaging_settings = structures.ImageSettings.from_dict(self.read_from_dict(rf"imaging_{key_multiple}.txt"))
                for i, hfw in enumerate(hfws):
                    imaging_settings.hfw = hfw
                    imaging_settings.filename = f"{acquisition_time}_{i}_{hfw*1000000}_um."
                    imaging_settings.path = self.folder_path
                    image = acquire.new_image(self.microscope, imaging_settings)
                    contrast[str(hfw)] = np.std(image.data)
                print(contrast)
            except Exception as e:
                print(f"The image acquisition failed: {e}")
        elif key == 'tiles':
            self.microscope.move_flat_to_beam(BeamType.ELECTRON)
            image_settings = self.settings.image
            image_settings.hfw = 600e-6
            image_settings.resolution = [1024, 1024]
            image_settings.beam_type = BeamType.ELECTRON
            image_settings.save = True
            image_settings.path = self.folder_path
            image_settings.filename='Tiles'
            # tile settings
            dx, dy = image_settings.hfw, image_settings.hfw
            nrows, ncols = 3, 3

            # tile
            initial_position = self.microscope.get_stage_position()
            for i in range(nrows):

                # restore position
                self.microscope.move_stage_absolute(initial_position)
                # stable movement dy
                self.microscope.stable_move(dx=0, dy=dy * i, beam_type=BeamType.ELECTRON)

                for j in range(ncols):
                    # stable movement dx
                    self.microscope.stable_move(dx=dx, dy=0, beam_type=BeamType.ELECTRON)
                    # acquire images with both beams
                    image_settings.filename = f"tile_{i:03d}_{j:03d}"
                    ib_image = acquire.new_image(self.microscope, image_settings)
            import glob
            filenames = sorted(glob.glob(os.path.join(image_settings.path, "tile*.tif")))

            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
            for i, fname in enumerate(filenames):
                image = structures.FibsemImage.load(fname)
                ax = axes[-(i // ncols +1)][i % ncols]
                ax.imshow(image.data, cmap="gray")
                ax.axis("off")

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.01, wspace=0.01)
            plt.savefig(os.path.join(image_settings.path, "tiles.png"), dpi=300)
            plt.show()
        else:
                return
        plt.show()


    def get_stage_position(self):
        current_stage_position = self.microscope.get_stage_position()
        current_stage_position_adjusted_unitis = [current_stage_position.x*1e3,
                                                  current_stage_position.y*1e3,
                                                  current_stage_position.z*1e3,
                                                  np.rad2deg(current_stage_position.r),
                                                  np.rad2deg(current_stage_position.t)]
        print(f"The current stage position is {current_stage_position_adjusted_unitis}.")
        return current_stage_position

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
                 elif mode == 'stable_ebeam':
                     self.microscope.stable_move(stage_movement.x, stage_movement.y, beam_type=BeamType.ELECTRON)
                 elif mode == 'stable_ion':
                     self.microscope.stable_move(stage_movement.x, stage_movement.y, beam_type=BeamType.ION)
                 elif mode == 'safe':
                     self.microscope.safe_absolute_stage_movement(stage_movement)
                 else:
                        return
            except Exception as e:
                print(f"The stage movement failed: {e}")
        print(f"The current stage position is {self.microscope.get_stage_position()}.")

    def create_fiducials(self):
        """
        Creates a fiducial based on the settings stored in the milling_base.txt file
        """
        rect_settings = structures.FibsemRectangleSettings(width=10e-6, height=10e-6, depth=1e-6, centre_x=0, centre_y=0)

        rectangle_pattern = milling.patterning.patterns2.RectanglePattern(
            width=10e-6,
            height=10e-6,
            depth=1e-6,
            point=structures.Point(0, 0),
        )


        milling_settings = structures.FibsemMillingSettings(
            milling_current=1e-9,
            milling_voltage=30e3,
            hfw=80e-6,
            application_file="Si",
            patterning_mode="Serial",
        )

        milling_alignment = milling.MillingAlignment(
            enabled=False,
        )

        milling_stage = milling.FibsemMillingStage(
            name="Milling Stage",
            milling=milling_settings,
            pattern=rectangle_pattern,
            alignment=milling_alignment,
        )

        milling.draw_patterns(self.microscope, milling_stage.pattern.define())

        # 3. run milling
        milling.run_milling(self.microscope, milling_stage.milling.milling_current, milling_stage.milling.milling_voltage)

        # 4. finish milling (restore imaging beam settings, clear shapes, ...)
        milling.finish_milling(self.microscope)


        #filename_milling = 'milling_base.txt'
        #pattern_1 = milling.patterning.patterns2.FiducialPattern.from_dict(self.read_from_dict(filename_milling))
        # self.microscope.set("plasma_gas", self.read_from_dict(filename_milling)['plasma_source'],
        #                     beam_type=BeamType.ION)
        # self.microscope.set("plasma", self.read_from_dict(filename_milling)['plasma'], beam_type=BeamType.ION)
        # milling_settings = structures.FibsemMillingSettings.from_dict(self.read_from_dict(filename_milling))
        # milling_alignment = milling.MillingAlignment(enabled=False)
        # milling_stages = milling.FibsemMillingStage(
        #     name="Fiducial",
        #     milling = milling_settings,
        #     pattern = pattern_1,
        #     alignment = milling_alignment,
        # )
        # milling.setup_milling(self.microscope, milling_stages)
        # print(f"The milling time is {self.microscope.estimate_milling_time()}")
        # milling.run_milling(self.microscope, milling_current=milling_stages.milling.milling_current, milling_voltage=milling_stages.milling.milling_voltage)
        # print(f"Running the milling finished.")
        # milling.finish_milling(self.microscope)
        # print('Milling finished.')

    def GIS_Coating(self, gridnumber):
        #GIS_Stage_Position = self.read_from_yaml(os.path.join(self.project_root, 'config', 'positions'), rf"grid0{gridnumber}-GIS")
        #self.move_stage([GIS_Stage_Position['x'], GIS_Stage_Position['y'], GIS_Stage_Position['z'], GIS_Stage_Position['r'], GIS_Stage_Position['t']], mode='absolute')
        #gis.gis_protocol = {"time": 10, "hfw": 900.0e-6, "gas": "Pt cryo", "length": None}
        #gis.deposit_platinum(self.microscope, gis.gis_protocol)
        print("Gas Injection finished")

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
        self.button_multiple = QPushButton("Series of Electron Beam Images")
        self.button_multiple.clicked.connect(lambda: self.fibsem.acquire_image('multiple'))
        acquire_button_layout.addWidget(self.button_multiple)
        self.button_tiles = QPushButton("Tiles")
        self.button_tiles.clicked.connect(lambda: self.fibsem.acquire_image('tiles'))
        acquire_button_layout.addWidget(self.button_tiles)
        self.button_fl = QPushButton("FL Image")
        self.button_fl.setEnabled(False)
        self.button_fl.clicked.connect(lambda: self.fibsem.acquire_image('fl'))
        acquire_button_layout.addWidget(self.button_fl)

        # Create a horizontal layout of input boxes for stage movement
        move_input_box_layout = QHBoxLayout()
        column_descriptors = ["X (mm)", "Y (mm)", "Z (mm)", "Rotation (deg)", "Tilt (deg)"]
        self.move_inputs = []
        for desc in column_descriptors:
            column_layout = QVBoxLayout()
            label = QLabel(desc, self)
            #label.setAlignment(Qt.AlignCenter)  # Center-align the text above the box
            column_layout.addWidget(label)
            input_box = QLineEdit(self)
            input_box.setPlaceholderText("Enter value")
            column_layout.addWidget(input_box)
            move_input_box_layout.addLayout(column_layout)
            self.move_inputs.append(input_box)

        absolute_input_box_layout = QHBoxLayout()
        self.absolute_move_inputs = []
        for _ in range(5):
            input_box = QLineEdit(self)
            input_box.setPlaceholderText("Enter value")
            absolute_input_box_layout.addWidget(input_box)
            self.absolute_move_inputs.append(input_box)

        acquire_button_layout.addLayout(move_input_box_layout)


        move_button_layout = QHBoxLayout()
        relative_move_button = QPushButton("Relative", self)
        relative_move_button.clicked.connect(self.relative_stage_movement)
        move_button_layout.addWidget(relative_move_button)
        absolute_move_button = QPushButton("Absolute", self)
        absolute_move_button.clicked.connect(self.absolute_stage_movement)
        move_button_layout.addWidget(absolute_move_button)
        stable_ebeam_move_button = QPushButton("Stable Ebeam", self)
        stable_ebeam_move_button.clicked.connect(self.stable_stage_movement_ebeam)
        move_button_layout.addWidget(stable_ebeam_move_button)
        stable_ion_move_button = QPushButton("Stable Ion", self)
        stable_ion_move_button.clicked.connect(self.stable_stage_movement_ion)
        move_button_layout.addWidget(stable_ion_move_button)
        safe_move_button = QPushButton("Safe", self)
        safe_move_button.clicked.connect(self.safe_stage_movement)
        move_button_layout.addWidget(safe_move_button)
        acquire_button_layout.addLayout(move_button_layout)

        milling_layout = QHBoxLayout()
        fiducial_button = QPushButton("Fiducial", self)
        fiducial_button.clicked.connect(fibsem.create_fiducials)
        milling_layout.addWidget(fiducial_button)
        stage_position_button = QPushButton("Stage Position", self)
        stage_position_button.clicked.connect(fibsem.get_stage_position)
        milling_layout.addWidget(stage_position_button)
        acquire_button_layout.addLayout(milling_layout)
        GIS_1_button = QPushButton("GIS Grid1", self)
        GIS_1_button.clicked.connect(lambda: fibsem.GIS_Coating(1))
        milling_layout.addWidget(GIS_1_button)
        # Set the main layout for the widget
        self.setLayout(acquire_button_layout)

    def relative_stage_movement(self):
        # Collect the settings for the relative stage movement
        values = [input_box.text() for input_box in self.move_inputs]
        fibsem.move_stage(values, 'relative')

    def absolute_stage_movement(self):
        # Collect data from the second row
        values = [input_box.text() for input_box in self.move_inputs]
        fibsem.move_stage(values, 'absolute')

    def stable_stage_movement_ebeam(self):
        values = [input_box.text() for input_box in self.move_inputs]
        fibsem.move_stage(values, 'stable_ebeam')

    def stable_stage_movement_ion(self):
        values = [input_box.text() for input_box in self.move_inputs]
        fibsem.move_stage(values, 'stable_ion')

    def safe_stage_movement(self):
        values = [input_box.text() for input_box in self.move_inputs]
        fibsem.move_stage(values, 'safe')



if __name__ == "__main__":
    #create a folder for the experiment
    print('Application started.')
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

