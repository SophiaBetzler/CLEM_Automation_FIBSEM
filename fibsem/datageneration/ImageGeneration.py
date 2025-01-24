from fibsem import acquire, utils
from fibsem.structures import BeamType, FibsemStagePosition
from tkinter import messagebox
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
from datetime import datetime
import numpy as np

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
            self.microscope, self.settings = utils.setup_session(manufacturer='Thermo', ip_address='192.168.0.1',
                                                                 config_path=config_path)

            # self.microscope, self.settings = utils.setup_session(manufacturer='Demo', ip_address='localhost',
            #                                                      config_path=config_path)
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
        try:
            plt.ion()  # needed to avoid the QCoreApplication::exec: The event loop is already running error
            if key == 'electron' or key == 'ion':
                self.settings.image.beam_type = dict_beamtypes[key]
                self.settings.image.filename = acquisition_time + '_' + str(key)
                print(self.settings.image)
                image = acquire.new_image(self.microscope, self.settings.image)
                print(self.settings.image)
                plt.imshow(image.data, cmap='gray')
            elif key == 'both':
                image_eb, image_ion = acquire.take_reference_images(self.microscope, self.settings.image)
                self.settings.image.filename = acquisition_time + '_' + 'ebeam_ion'
                fig, ax = plt.subplots(1, 2, figsize=(10, 7))
                ax[0].imshow(image_eb.data, cmap="gray")
                ax[0].set_title("Electron Image")
                ax[1].imshow(image_ion.data, cmap="gray")
                ax[1].set_title("Ion Image")
            else:
                return
            plt.show()
        except Exception as e:
            error_message(f"Image acquisition failed: {e}")



    def move_stage(self, new_stage_position, mode):
        print(f"The current stage position is {self.microscope.get_stage_position()}.")
        try:
            stage_movement = FibsemStagePosition(x=float(new_stage_position[0])*1e-6,
                                y=float(new_stage_position[1])*1e-6,
                                z=float(new_stage_position[2])*1e-6,
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



class Gui(QWidget):
    def __init__(self, fibsemcontrol):
        super().__init__()
        self.fibsem = fibsemcontrol
        # Create a vertical layout of buttons to acquire images
        buttonlayout = QVBoxLayout()
        self.button_eb = QPushButton("Electron Beam Image")
        #self.button_eb.clicked.connect(lambda: self.fibsem.acquire_image('electron'))
        self.button_eb.clicked.connect(lambda: self.fibsem.acquire_image('electron'))
        buttonlayout.addWidget(self.button_eb)
        self.button_ion = QPushButton("Ion Beam Image")
        self.button_ion.clicked.connect(lambda: self.fibsem.acquire_image('ion'))
        buttonlayout.addWidget(self.button_ion)
        self.button_both = QPushButton("Electron and Ion Beam Image")
        self.button_both.clicked.connect(lambda: self.fibsem.acquire_image('both'))
        buttonlayout.addWidget(self.button_both)
        self.button_fl = QPushButton("FL Image")
        self.button_fl.setEnabled(False)
        self.button_fl.clicked.connect(lambda: self.fibsem.acquire_image('fl'))
        buttonlayout.addWidget(self.button_fl)
        self.setLayout(buttonlayout)
        # Create a horizontal layout of input boxes for stage movement
        firstrowboxlayout = QHBoxLayout()
        column_descriptors = ["X (um)", "Y (um)", "Z (um)", "Rotation (deg)", "Tilt (deg)"]
        self.first_row_inputs = []
        for desc in column_descriptors:
            column_layout = QVBoxLayout()
            label = QLabel(desc, self)
            #label.setAlignment(Qt.AlignCenter)  # Center-align the text above the box
            column_layout.addWidget(label)
            input_box = QLineEdit(self)
            input_box.setPlaceholderText("Enter value")
            column_layout.addWidget(input_box)
            firstrowboxlayout.addLayout(column_layout)
            self.first_row_inputs.append(input_box)
        first_row_button = QPushButton("Relative", self)
        first_row_button.clicked.connect(self.collect_first_row_data)
        firstrowboxlayout.addWidget(first_row_button)
        secondrowboxlayout = QHBoxLayout()
        self.second_row_inputs = []
        for _ in range(5):
            input_box = QLineEdit(self)
            input_box.setPlaceholderText("Enter value")
            secondrowboxlayout.addWidget(input_box)
            self.second_row_inputs.append(input_box)

        second_row_button = QPushButton("Absolute", self)
        second_row_button.clicked.connect(self.collect_second_row_data)
        secondrowboxlayout.addWidget(second_row_button)
        buttonlayout.addLayout(firstrowboxlayout)
        buttonlayout.addLayout(secondrowboxlayout)

        # Add a label to display results
        self.result_label = QLabel("Result: ", self)
        buttonlayout.addWidget(self.result_label)

        # Set the main layout for the widget
        self.setLayout(buttonlayout)

    def collect_first_row_data(self):
        # Collect data from the first row
        values = [input_box.text() for input_box in self.first_row_inputs]
        fibsem.move_stage(values, 'relative')

    def collect_second_row_data(self):
        # Collect data from the second row
        values = [input_box.text() for input_box in self.second_row_inputs]
        fibsem.move_stage(values, 'absolute')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fibsem = Fibsemcontrol()
    gui = Gui(fibsem)
    gui.show()
    sys.exit(app.exec())

