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
    def __init__(self, bf, imaging, fib_settings):
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
        imaging = Imaging(bf=self.bf, fib_microscope=self.fib_microscope, beam=self.beam)
        imaging.acquire_image(hfw=hfw, folder_path=self.temp_folder_path, save=True, autofocus=True,
                              fib_settings=self.fib_settings)

    def fiducial_identification(self, number_fiducials):
        """
        Function to identify the fiducials in the images. It is iterative at lower and higher
        magnification, by connecting to the ML scripts. The ML scripts returns the correct stage position required
        to move either the fiducial or the center between the two fiducials into the center of the image.
        number_fiducials: currently 1 or 2.
        """
        if number_fiducials == 1:
            hfw_1 = 150e-6
            hfw_2 = 80e-6
        elif number_fiducials == 2:
            hfw_1 = 300e-6
            hfw_2 = 150e-6
        else:
            hfw_1 = 300e-6
            hfw_2 = 150e-6

        self.data_generation(hfw=hfw_1)
        self.bf.execute_external_script(script='Identify_Fiducial_Remote.py',
                                        dir_name='Ultralytics',
                                        parameter=number_fiducials)

        while not os.path.exists(os.path.join(self.temp_folder_path, 'fiducial_id_result.json')):
            time.sleep(0.1)

        with open(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'), 'r') as file:
            required_move = json.load(file)

        os.remove(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'))

        self.fib_microscope.stable_move(required_move['moveX'], required_move['moveY'], self.beam_type)

        self.data_generation(hfw=hfw_2)
        self.bf.execute_external_script('Identify_Fiducial_Remote.py',
                                        'Ultralytics',
                                        parameter=number_fiducials)

        while not os.path.exists(os.path.join(self.temp_folder_path, 'fiducial_id_result.json')):
            time.sleep(1)

        with open(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'), 'r') as file:
            required_move = json.load(file)

        os.remove(os.path.join(self.temp_folder_path, 'fiducial_id_result.json'))

        self.fib_microscope.stable_move(required_move['moveX'], required_move['moveY'], self.beam_type)

