from Basic_Functions import BasicFunctions
from Imaging import Imaging
from fibsem import structures
import argparse
import os
import time
import json

class FiducialID:
    """
    This class loads an external script to allow ML based identification of fiducials in the image
    and then moves the stage either so that the fiducial is in the center (1) or the center between the two
    fiducials is in the center of the image (2).
    number_fiducials: 1 or 2 (currently, maybe this will be expanded?)
    fib_microscope: the fib_microscope object created in the Basic_Functions class
    beam: either 'ion' or 'electron'
    """
    def __init__(self, bf, imaging, number_fiducials, fib_settings):
        self.number_fiducials = number_fiducials
        self.imaging = imaging
        self.fib_microscope = self.imaging.fib_microscope
        self.beam = self.imaging.beam
        self.beam_type = getattr(structures.BeamType, self.beam.upper())
        self.bf = bf
        self.folder_path = self.bf.folder_path
        self.temp_folder_path = self.bf.temp_folder_path
        self.imaging_settings, self.imaging_settings_dict = bf.read_from_yaml(filename=f"imaging_{self.beam}")
        self.beam_settings = structures.BeamSettings.from_dict(self.imaging_settings_dict)
        self.fib_microscope.set_beam_settings(self.beam_settings)
        self.fib_settings = fib_settings

    def data_generation(self, hfw):
        """
        Image acquisition.
        """
        imaging = Imaging(bf=self.bf, fib_microscope=self.fib_microscope, beam=self.beam, autofocus=True,
                          fib_settings=self.fib_settings)
        imaging.acquire_image(hfw=hfw, folder_path=self.temp_folder_path, save=True)

    def fiducial_identification(self):
        """
        Function to identify the fiducials in the images. It is iterative at lower and higher
        magnification.
        """
        self.data_generation(hfw=150e-6)
        self.bf.execute_external_script(script='Identify_Fiducial_Remote.py',
                                        dir_name='Ultralytics',
                                        parameter=self.number_fiducials)

        while not os.path.exists(os.path.join(self.temp_folder_path, 'fiducial_id_result.json')):
            time.sleep(1)

        with open(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'), 'r') as file:
            required_move = json.load(file)

        os.remove(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'))

        print(self.fib_microscope.get_stage_position())
        print(f"The move required in X direction is {required_move['moveX']}, the move in Y direction is {required_move['moveY']}")
        self.fib_microscope.stable_move(required_move['moveX'], required_move['moveY'], self.beam_type)
        print(self.fib_microscope.get_stage_position())
        if self.number_fiducials == 1:
            hfw_2 = 80e-6
        elif self.number_fiducials == 2:
            hfw_2 = 150e-6
        else:
            hfw_2 = 300e-6
        self.data_generation(hfw=hfw_2)
        self.bf.execute_external_script('Identify_Fiducial_Remote.py',
                                        'Ultralytics',
                                        parameter=self.number_fiducials)

        while not os.path.exists(os.path.join(self.temp_folder_path, 'fiducial_id_result.json')):
            time.sleep(1)

        with open(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'), 'r') as file:
            required_move = json.load(file)

        os.remove(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'))

        print(self.fib_microscope.get_stage_position())
        print(
            f"The move required in X direction is {required_move['moveX']}, the move in Y direction is {required_move['moveY']}")
        self.fib_microscope.stable_move(required_move['moveX'], required_move['moveY'], self.beam_type)
        print(self.fib_microscope.get_stage_position())

