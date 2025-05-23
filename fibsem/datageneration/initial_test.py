from fibsem import acquire, utils
from fibsem.structures import BeamType
import tkinter as tk
from tkinter import messagebox
#from PyQt5 import
import matplotlib
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

def error_message(text):
    messagebox.showerror("Error", text)


class fibsemcontrol():
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
            self.microscope, self.settings = utils.setup_session(manufacturer='Demo', ip_address='localhost',
                                                                 config_path=config_path)
            print(f"The settings are {self.settings}.")
            print(f"The microscope is {self.microscope}")
        except:
            error_message('Connection to microscope failed.')
            sys.exit()

    def acquire_image(self, key):
        dict_beamtypes = {
                'electron': BeamType.ELECTRON,
                'ion': BeamType.ION
                #'fl': BeamType.FL
                        }
        self.settings.image.beam_type = dict_beamtypes[key]
        image = acquire.new_image(self.microscope, self.settings.image)
        plt.imshow(image.data)
        plt.show()


# b


if __name__ == "__main__":
    fibsem = fibsemcontrol()
    fibsem.acquire_image('electron')

