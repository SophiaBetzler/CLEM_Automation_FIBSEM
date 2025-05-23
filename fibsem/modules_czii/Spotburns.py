
from fibsem import acquire, utils, microscope, structures, milling, calibration
from fibsem.structures import BeamType, FibsemStagePosition, FibsemDetectorSettings
import yaml
import os
from pathlib import Path
import numpy as np
import time



class SpotburnPatternBurning:

    def __init__(self, bf, fib_microscope):
        self.fib_microscope = fib_microscope
        self.bf = bf
        self.temp_folder_path = self.bf.temp_folder_path
        self.project_root = Path(__file__).resolve().parent.parent
        with open(os.path.join(self.project_root, 'config', 'czii-stored-stage-positions_hydra.yaml'), "r") as file:
            self.saved_stage_positions = yaml.safe_load(file)
        self.thermo_microscope = self.fib_microscope.connection
        self.list_spotburn_positions = [[0.0625, 713],
                                     [0.1719, 0.687],
                                     [0.1719, 0.739],
                                     [0.2813, 0.713],
                                     [0.2813, 0.766],
                                     [0.391, 0.739],
                                     [0.5, 0.713],
                                     [0.5, 0.766],
                                     [0.609, 0.739],
                                     [0.719, 0.713],
                                     [0.719, 0.766],
                                     [0.828, 0.687],
                                     [0.828, 0.739],
                                     [0.9375, 0.713],
                                     [0.172, 0.792],
                                     [0.281, 0.818],
                                     [0.391, 0.792],
                                     [0.609, 0.792],
                                     [0.828, 0.792],
                                     [0.719, 0.818]]




    def identify_trench(self):
        self.fib_microscope.set("plasma_gas", 'Xenon',
                                beam_type=BeamType.ION)
        self.fib_microscope.set("plasma", True,
                                beam_type=BeamType.ION)
        ion_beam_imaging_settings_dict = {'beam_current': 25.0e-12,
                                        'voltage': 30000,
                                        'working_distance': 16.5e-3,
                                        'scan_rotation': np.deg2rad(float(180.0)),
                                        'dwell_time': 100e-12,
                                        'resolution': (1536, 1024),
                                        'save': True,
                                        'filename': 'FIB_Image.tif',
                                        'path': self.temp_folder_path,
                                        'line_integration': 5 }

        imaging_settings = structures.ImageSettings.from_dict(ion_beam_imaging_settings_dict)
        calibration.auto_focus_beam(self.fib_microscope, beam_type=BeamType.ION)
        acquire.new_image(self.fib_microscope, imaging_settings)
        self.bf.execute_external_script(script='Identify_Trench.py',
                                        dir_name='Ultralytics')

    def burn_spotburns(self):
        self.thermo_microscope.beams.ion_beam.beam_current.value = 0.1e-9
        self.thermo_microscope.beams.ion_beam.high_voltage.value = 30000
        self.thermo_microscope.beams.ion_beam.horizontal_field_width.value = 40e-6
        self.thermo_microscope.beams.ion_beam.blank()
        for spotburn in self.list_spotburn_positions:
            self.thermo_microscope.beams.ion_beam.scanning.mode.set_spot(spotburn[0], spotburn[1])
            self.thermo_microscope.beams.ion_beam.unblank()
            time.sleep(10)
            self.thermo_microscope.beams.ion_beam.blank()
