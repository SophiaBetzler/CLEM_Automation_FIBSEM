from Basic_Functions import BasicFunctions
from fibsem import structures, acquire, calibration
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import math


class Imaging:
    def __init__(self, fib_microscope, bf, beam='electron', imaging_settings = None, autofocus=False, fib_settings=None):
        self.bf = bf
        if autofocus is True and fib_settings is None:
            raise ValueError("optional_input is required when use_optional is True")
        self.autofocus = autofocus
        self.fib_settings = fib_settings
        #self.imaging_settings_dict = bf.read_from_dict(filename=f"imaging_{beam}")
        if imaging_settings is None:
            self.imaging_settings, self.imaging_settings_dict = bf.read_from_yaml(filename=f"imaging_{beam}")
        else:
            self.imaging_settings = imaging_settings
            _, self.imaging_settings_dict = bf.read_from_yaml(filename=f"imaging_{beam}")
        self.beam = beam
        self.beam_type = getattr(structures.BeamType, self.beam.upper())
        self.beam_settings = structures.BeamSettings.from_dict(self.imaging_settings_dict, self.beam_type)
        self.folder_path = bf.folder_path
        self.imaging_settings.path = self.folder_path
        self.imaging_settings.beam_type = self.beam_type
        self.fib_microscope = fib_microscope
        self.fib_microscope.set_beam_settings(self.beam_settings)

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
            return image
        except Exception as e:
            print(f"The image acquisition failed because of {e}.")

    def acquire_multiple(self, dict_parameters):
        """
        Uses lists of parameters stored in a dictionary to screen multiple imaging conditions.
        The keys of the dictionary must match attributes of the structures.ImageSettings class.
        """
        imaging_settings = self.imaging_settings
        beam_settings = self.beam_settings
        imaging_settings.save = True
        filename = self.imaging_settings.filename
        for key, values in dict_parameters.items():
            for value in values:
                imaging_settings.key = value
                if key == 'hfw':
                    imaging_settings.hfw = value
                    value = value * 1e6
                elif key == 'current':
                    beam_settings.beam_current = value
                    value = value * 1e12
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

    def fast_acquire(self, number_frames, line_acquisition=False):
        logging.disable(logging.CRITICAL)
        if line_acquisition is True:
            image_resolution = self.imaging_settings.resolution
            self.imaging_settings.save = False
            self.imaging_settings.reduced_area = structures.FibsemRectangle(top=0.3,
                                                                            left=0.0,
                                                                            width=1.0,
                                                                            height=float(1/image_resolution[1]))
            #To use fast readout switch to ThermoFisher
            self.fib_microscope.connection.beams.electron_beam.scanning.mode.set_reduced_area = \
                (self.imaging_settings.reduced_area.top,
                 self.imaging_settings.reduced_area.left,
                 self.imaging_settings.reduced_area.width,
                 self.imaging_settings.reduced_area.height)
            self.fib_microscope.connection.beams.electron_beam.beam_current.value = self.imaging_settings.current
            self.fib_microscope.connection.beams.electron_beam.high_voltage.value = self.beam_settings.voltage
            self.fib_microscope.connection.beams.electron_beam.horizontal_field_width.value = self.beam_settings.hfw
            self.fib_microscope.connection.beams.electron_beam.scanning.resolution.value = (
                self.fib_microscope.connection.ScanningResolution.PRESET_1536x1024)
            self.fib_microscope.connection.beams.electron_beam.scanning.dwell_time.value = self.imaging_settings.dwell_time
            settings = self.fib_microscope.connection.GetImageSettings(wait_for_frame=True)
            image = self.fib_microscope.connection.microscope.imaging.get_image(settings)
            now = datetime.now()
            ms = now.strftime("%f")[:3]
            array_timeseries = [image.data[0, :]]
            now = datetime.now()
            ms = now.strftime("%f")[:3]
            list_timestamps = [f"{0}_{now:%H-%M-%S}-{ms}"]
            k = 0
            while self.fib_microscope.connection.microscope.imaging.state == self.fib_microscope.connection.ImagingState.ACQUIRING \
                and k < number_frames:
                settings = self.fib_microscope.connection.GetImageSettings(wait_for_frame=True)
                image = self.fib_microscope.connection.microscope.imaging.get_image(settings)
                now = datetime.now()
                ms = now.strftime("%f")[:3]
                array_timeseries.append([image.data[0, :]])
                now = datetime.now()
                ms = now.strftime("%f")[:3]
                list_timestamps.append([f"{0}_{now:%H-%M-%S}-{ms}"])
                k=k+1
            np.savetxt(f"{self.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition.txt", array_timeseries, fmt='%.3f')
            np.save(f"{self.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition", array_timeseries)
            plt.imshow(array_timeseries, cmap='gray')
            plt.savefig(f"{self.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition.png", dpi=600, bbox_inches="tight")
            plt.show()
            with open(f"{self.folder_path}/{now:%H-%M-%S}-{ms}_fast_acquisition_timestamps.txt", "w") as f:
                for item in list_timestamps:
                    f.write(str(item) + "\n")
        else:
            self.imaging_settings.save = True
            self.imaging_settings.reduced_area = None
            for i in range(number_frames):
                now = datetime.now()
                ms = now.strftime("%f")[:3]
                self.imaging_settings.filename = f"{i}_{now:%H-%M-%S}-{ms}"
                acquire.acquire_image(self.fib_microscope, self.imaging_settings)
        logging.disable(logging.NOTSET)

