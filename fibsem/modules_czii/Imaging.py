from Basic_Functions import BasicFunctions
from fibsem import structures, acquire, calibration
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Imaging:
    def __init__(self, fib_microscope, beam='electron', autofocus=False, fib_settings=None):
        bf = BasicFunctions()
        #self.imaging_settings_dict = bf.read_from_dict(filename=f"imaging_{beam}")
        self.imaging_settings_dict = bf.read_from_yaml(filename=f"imaging_{beam}")
        self.imaging_settings = structures.ImageSettings.from_dict(self.imaging_settings_dict)
        self.beam = beam
        self.beam_type = getattr(structures.BeamType, self.beam.upper())
        self.folder_path = BasicFunctions.folder_path
        self.imaging_settings.path = self.folder_path
        self.imaging_settings.beam_type = self.beam_type
        self.fib_microscope = fib_microscope
        self.beam_settings = structures.BeamSettings.from_dict(self.imaging_settings_dict, self.beam_type)
        self.fib_microscope.set_beam_settings(self.beam_settings)
        if autofocus is True and fib_settings is None:
            raise ValueError("optional_input is required when use_optional is True")
        self.autofocus = autofocus
        self.fib_settings = fib_settings

    def acquire_image(self, hfw=None, folder_path=None, save=None):
        """
        This function connects to the buttons in the GUI. It allows to take electron beam, ion beam and electron and ion
        beam images. TO DO: Add ability to take fluorescence images.
        key = 'electron', 'ion', 'both'
        Data are saved to the hard drive.
        """
        plt.ion()  # needed to avoid the QCoreApplication::exec: The event loop is already running error
        acquisition_time = datetime.now().strftime("%H-%M")
        self.imaging_settings.filename = acquisition_time
        if hfw is not None:
            self.imaging_settings.hfw = hfw
        if folder_path is not None:
            self.imaging_settings.path = folder_path
            print(f"Imaging path {self.imaging_settings.path}")
        if save is not None:
            self.imaging_settings.save = save

        if self.beam == 'ion':
            self.fib_microscope.set("plasma_gas", self.imaging_settings_dict['plasma_source'],
                                beam_type=self.beam_type)
            self.fib_microscope.set("plasma", self.imaging_settings_dict['plasma'],
                                beam_type=self.beam_type)
        try:
            if self.autofocus == True:
                calibration.auto_focus_beam(self.fib_microscope, self.fib_settings, self.beam_type)
            image = acquire.new_image(self.fib_microscope, self.imaging_settings)
            plt.imshow(image.data, cmap='gray')
            plt.show()
        except Exception as e:
            print(f"The image acquisition failed because of {e}.")

    def acquire_multiple(self, dict_parameters):
        """
        Uses lists of parameters stored in a dictionary to screen multiple imaging conditions.
        The keys of the dictionary must match attributes of the structures.ImageSettings class.
        """
        imaging_settings = self.imaging_settings
        imaging_settings.save = True
        filename = self.imaging_settings.filename
        for key, values in dict_parameters.items():
            for value in values:
                imaging_settings.key = value
                if key == 'hfw':
                    value = value*1e6
                imaging_settings.filename = f"{filename}_{key}_{value}"
                acquire.new_image(self.fib_microscope, imaging_settings)

    def acquire_tileset(self):
        """
        Acquire a tileset of either the ion-beam or the electron-beam image. A preset magnification and
        image resolution are used.
        """
        imaging_settings = self.imaging_settings
        imaging_settings.hfw = 600e-6
        imaging_settings.resolution = [2048, 2048]
        imaging_settings.save = True
        imaging_settings.save = True
        imaging_settings.path = self.folder_path
        imaging_settings.filename = 'Tiles'

        dx, dy = imaging_settings.hfw, imaging_settings.hfw
        nrows, ncols = 3, 3
        initial_position = self.fib_microscope.get_stage_position()
        for i in range(nrows):
            self.fib_microscope.move_stage_absolute(initial_position)
            self.fib_microscope.stable_move(dx=0, dy=dy * i, beam_type=self.beam_type)
            for j in range(ncols):
                self.fib_microscope.stable_move(dx=dx, dy=0, beam_type=self.beam_type)
                imaging_settings.filename = f"tile_{i:03d}_{j:03d}"
                acquire.new_image(self.fib_microscope, imaging_settings)
        import glob
        if self.beam == 'electron':
            filenames = sorted(glob.glob(os.path.join(imaging_settings.path, "tile*_eb.tif")))
        elif self.beam == 'ion':
            filenames = sorted(glob.glob(os.path.join(imaging_settings.path, "tile*_ib.tif")))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
        for i, fname in enumerate(filenames):
            image = structures.FibsemImage.load(fname)
            ax = axes[-(i // ncols + 1)][i % ncols]
            ax.imshow(image.data, cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.001, wspace=0.001)
        plt.savefig(os.path.join(imaging_settings.path, "tiles.png"), dpi=300)
        plt.show()
