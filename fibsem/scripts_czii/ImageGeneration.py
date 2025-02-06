from fibsem import acquire, utils, microscope, structures, milling
from fibsem.structures import BeamType, FibsemStagePosition, FibsemImageMetadata
from tkinter import messagebox
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import time

def error_message(text):
    messagebox.showerror("Error", text)


class Fibsemcontrol():
    """
    Class which incorporates every function needed for FIB SEM Control.
    Class calls functions from openFIBSEM to control the microscope.

    """
    def __init__(self):
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
        try:
            #for hydra microscope use:
            config_path = os.path.join(self.project_root, 'config', 'czii-tfs-hydra-configuration.yaml')
            #for arctis microscope use:
            #config_path = os.path.join(self.project_root, 'config', 'tfs-arctis-configuration.yaml')
            # self.microscope, self.settings = utils.setup_session(manufacturer='Thermo', ip_address='192.168.0.1',
            #                                                      config_path=config_path)
            #
            self.microscope, self.settings = utils.setup_session(manufacturer='Demo', ip_address='localhost',
                                                                  config_path=config_path)
            print(f"The settings are {self.settings}.")
            print(f"The microscope is {self.microscope}")
        except Exception as e:
            error_message(f"Connection to microscope failed: {e}")
            sys.exit()

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
        current_date = datetime.now().strftime("%Y%m%d")
        desktop_path = Path.home() / "Desktop/"
        folder_path = os.path.join(desktop_path, 'TestImages/', current_date)
        acquisition_time = datetime.now().strftime("%H-%M")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            print('Directory already exists')
        self.settings.image.path = folder_path
        plt.ion()  # needed to avoid the QCoreApplication::exec: The event loop is already running error
        imaging_settings = structures.ImageSettings(
            autocontrast=True,
            autogamma=False,
            resolution=[1536, 1024],
            dwell_time=3e-6,
            filename=acquisition_time,
            beam_type=dict_beamtypes[key],
            save=True,
            hfw=150.0e-6,
            path=folder_path,
        )
        ion_imaging_settings = milling.FibsemMillingSettings(
            milling_voltage=3000,
            milling_current=60e-12,
        )
        if key == 'electron':
            try:
                image = acquire.new_image(self.microscope, imaging_settings)
                plt.imshow(image.data, cmap='gray')
            except Exception as e:
                print(f"Image acquisition failed: {e}")
        elif key == 'ion':
            try:
                ion_imaging_settings = milling.FibsemMillingSettings(
                    milling_current = 60e-12,
                    milling_voltage = 30000,
                )
                #milling.set("current", ion_imaging_settings.milling_current, ion_imaging_settings.milling_channel)
                #milling.set("voltage", ion_imaging_settings.milling_voltage, ion_imaging_settings.milling_channel)
                #time.sleep(15)
                #print("Imaging current set.")
                image = acquire.new_image(self.microscope, imaging_settings)
                plt.imshow(image.data, cmap='gray')
            except Exception as e:
                print(f"Image acquisition failed {e}.")
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

    def move_stage(self, new_stage_position, mode):
        print(f"The current stage position is {self.microscope.get_stage_position()}.")
        try:
            stage_movement = FibsemStagePosition(x=float(new_stage_position[0])*1e-3,
                                y=float(new_stage_position[1])*1e-3,
                                z=float(new_stage_position[2])*1e-3,
                                r=np.deg2rad(float(new_stage_position[3])),
                                t=np.deg2rad(float(new_stage_position[4])))
            print(f"The current stage position is {self.microscope.get_stage_position()}.")
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
        rectangle_pattern_1 = structures.FibsemRectangleSettings(
            rotation=30,
            width=10.0e-6,
            height=50.0e-6,
            centre_x=centerX,
            centre_y=centerY,
            depth=3e-6,
        )

        rectangle_pattern_2 = structures.FibsemRectangleSettings(
            rotation=-30,
            width=10.0e-6,
            height=100.0e-6,
            centre_x=centerX,
            centre_y=centerY,
            cleaning_cross_section=True,
            depth=3e-6,
        )

        ion_milling_settings = milling.FibsemMillingSettings(
            milling_voltage=30000,
            milling_current=15e-9,
            patterning_mode='Serial',
        )
        ion_imaging_settings = milling.FibsemMillingSettings(
            milling_voltage=3000,
            milling_current=60e-12,
        )
        self.acquire_image('ion')
        milling.draw_patterns(self.microscope, [rectangle_pattern_1, rectangle_pattern_2])
        #milling.FibsemMillingSettings.set("current", ion_milling_settings.milling_current, ion_milling_settings.milling_channel)
        #milling.FibsemMillingSettings.set("voltage", ion_milling_settings.milling_voltage, ion_milling_settings.milling_channel)
        print(f"The milling current and milling voltage are set.")
        print(f"The estimated milling time is {milling.estimate_milling_time(self.microscope, [rectangle_pattern_1, rectangle_pattern_2])}.")
        milling.setup_milling(self.microscope, ion_milling_settings)
        milling.run_milling(self.microscope, ion_milling_settings.milling_voltage, ion_milling_settings.milling_current)
        print(f"Milling finished.")
        self.acquire_image('ion')
        milling.finish_milling(self.microscope, imaging_current=ion_imaging_settings.milling_current,
                               imaging_voltage=ion_imaging_settings.milling_voltage)

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
        #self.button_eb.clicked.connect(lambda: self.fibsem.acquire_image('electron'))
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
    app = QApplication(sys.argv)
    fibsem = Fibsemcontrol()
    gui = Gui(fibsem)
    gui.show()
    sys.exit(app.exec())

