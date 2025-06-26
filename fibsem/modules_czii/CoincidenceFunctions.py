
from fibsem.structures import FibsemStagePosition, BeamType

from GIS_Sputter_Setup import GisSputterAutomation
from Imaging import Imaging

import sys
import time
import copy
import queue
import platform
import json
import os
import threading
import statistics
from datetime import datetime, date
from pathlib import Path
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor
import re

import tifffile
import cv2
import numpy as np

pc_type = platform.system()
import matplotlib
if pc_type == 'Windows':
    matplotlib.use('Qt5Agg')
elif pc_type == 'Darwin':
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


sys.path.append('C:\Program Files\Thermo Scientific Autoscript')
sys.path.append('C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages')

#from autoscript_sdb_microscope_client import SdbMicroscopeClient
#from autoscript_sdb_microscope_client.enumerations import *
#from autoscript_sdb_microscope_client.structures import GetImageSettings
#from autoscript_sdb_microscope_client import SdbMicroscopeClient



class CoincidenceFunctions:
    def __init__(self, oa, mode, on_experiment_stopped=None):
        self.results = None
        self.oa = oa
        if self.oa.tool != 'Arctis':
            raise RuntimeError("This is not the right tool to run the automated tricoincidence routine.")
        self.imaging = Imaging(self.oa)
        self.mode = mode
        #self.oa.autoloader_control()
        self.position_data = []
        self.data_file = os.path.join(self.oa.folder_path, 'position_data.json')
        self.hfw = 120.0e-6
        self.auto_gis = GisSputterAutomation(self.oa)
        self.lock  = threading.Lock()
        self.coin_stop_event = threading.Event()
        self.pause_not_stop = False
        self.data_callback = None
        self.on_experiment_stopped = on_experiment_stopped


#######################################################################################################################
#####       Functions controlling the Arctis
#######################################################################################################################

    def grab_fl_live_image(self, save=True):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            image = self.oa.thermo_microscope.imaging.get_image()
            if save is True:
                image.save(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
            img = image.data
        else:
            sim = self.fl_data_simulation()
            image = sim()
            if save is True:
                tifffile.imwrite(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"), image)
            img = image
            print('[INFO] Collecting live FL image.')
        return img

    def grab_reflection_image(self, row=None):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.camera_settings.filter.type.value = CameraFilterType.REFLECTION
            self.oa.thermo_microscope.detector.brightness.value = 0.01
            self.oa.thermo_microscope.detector.camera_settings.binning.value = 4
            self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = 0.001
            emission_color = self.oa.thermo_microscope.detector.camera_settings.emission.type.value
            self.oa.thermo_microscope.detector.camera_settings.emission.start(emission_type=
                                                                              emission_color)
            reflection_image = self.grab_fl_live_image()
            self.oa.thermo_microscope.detector.camera_settings.emission.stop()
            if self.mode == 'auto':
                reflection_image.save(os.path.join(self.oa.folder_path, f"{row}-Dataset", f"{row}_reflection_image.tif"))
            else:
                reflection_image.save(os.path.join(self.manual_folder_path, f"reflection_image.tif"))

        else:
            sim = self.fl_data_simulation()
            image = sim()
            tifffile.imwrite(os.path.join(self.manual_folder_path, f"reflection_image.tif"), image)
            print('[INFO] Collecting reflection FL image.')

    def grab_fluorescence_image(self, add_on, row=None):
        if self.mode == 'auto' and self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.camera_settings.filter.type.value = CameraFilterType.FLUORESCENCE
            self.oa.thermo_microscope.detector.camera_settings.binning.value = self.position_data[row]['fl_settings']
            ['binning']
            self.oa.thermo_microscope.detector.brightness.value = self.position_data[row]['brightness']
            self.oa.thermo_microscope.detector.camera_settings.exposure_time.value \
                = self.position_data[row]['fl_settings']['exposure_time']
            self.oa.thermo_microscope.detector.camera_settings.filter.type.value \
                = self.position_data[row]['fl_settings']['filter_setting']
            self.oa.thermo_microscope.detector.camera_settings.emission.type \
                = self.position_data[row]['fl_settings']['emission_color']
            self.oa.thermo_microscope.detector.camera_settings.focus.value \
                = self.position_data[row]['fl_settings']['objective_focus']
            image = self.oa.thermo_microscope.imaging.grab_frame(save=False)
            image.save(os.path.join(self.oa.folder_path, f"{row}-Dataset", f"{row}_fl_image_{add_on}.tif"))
            img = image.data
        elif self.mode == 'manual' and self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            if self.oa.thermo_microscope.detector.camera_settings.filter.type.value == CameraFilterType.REFLECTION and hasattr(self.manual_binning):
                pass
            else:
                self.manual_binning = self.oa.thermo_microscope.detector.camera_settings.binning.value
                self.manual_brightness = self.oa.thermo_microscope.detector.brightness.value
                self.manual_exposure_time = self.oa.thermo_microscope.detector.camera_settings.exposure_time.value
                self.manual_filter_settings = self.oa.thermo_microscope.detector.camera_settings.filter.type.value
                self.manual_emission_color = self.oa.thermo_microscope.detector.camera_settings.emission.type
                self.manual_objective_focus =  self.oa.thermo_microscope.detector.camera_settings.focus.value
            image = self.oa.thermo_microscope.imaging.grab_frame()
            image.save(os.path.join(self.manual_folder_path, f"Fl_image_{add_on}.tif"))
            img = image.data
        else:
            sim = self.fl_data_simulation()
            image = sim()
            tifffile.imwrite(os.path.join(self.manual_folder_path, f"Fl_image_{add_on}.tif"), image)
            img = image
            print('[INFO] Collecting FL image.')
        return img

    def run_serial_acquisition_fl_images(self, update_callback, stop_event, start_timestamp, row=None, fl_settings=None,
                                         timestamp=None):
        self.image_queue = queue.Queue(maxsize=2000)
        self.coin_stop_event.clear()
        if self.oa.manufacturer != 'Demo':
            if self.mode == 'auto':
                folder_path = os.path.join(self.oa.folder_path, f"{row}-Dataset")
                self.oa.thermo_microscope.imaging.set_active_view(3)
                self.oa.thermo_microscope.imaging.set_active_device(8)
                self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = fl_settings['exposure_time']
                self.oa.thermo_microscope.detector.brightness.value = fl_settings['brightness']
                self.oa.thermo_microscope.detector.camera_settings.binning.value = fl_settings['binning']
                self.oa.thermo_microscope.detector.camera_settings.filter.type.value = fl_settings['filter_setting']
                self.oa.thermo_microscope.imaging.start_acquisition()
            if self.mode == 'manual':
                folder_path = self.manual_folder_path
                self.oa.thermo_microscope.imaging.set_active_view(3)
                self.oa.thermo_microscope.imaging.set_active_device(8)
                self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = self.manual_exposure_time
                self.oa.thermo_microscope.detector.brightness.value = self.manual_brightness
                self.oa.thermo_microscope.detector.camera_settings.binning.value = self.manual_binning
                self.oa.thermo_microscope.detector.camera_settings.filter.type.value = self.manual_filter_settings
                self.oa.thermo_microscope.imaging.start_acquisition()
            else:
                folder_path = None
                raise RuntimeError("No valid mode selected!")

            writer_thread = threading.Thread(target=self.image_writer, args=(folder_path, start_timestamp), daemon=True)
            writer_thread.start()
            i = 0
            start_time = datetime.now()
            emission_color = self.oa.thermo_microscope.detector.camera_settings.emission.type.value
            self.oa.thermo_microscope.detector.camera_settings.emission.start(emission_type=
                                                                       emission_color)
            image = self.oa.thermo_microscope.imaging.get_image()
            try:
                if self.oa.thermo_microscope.imaging.state == ImagingState.ACQUIRING:
                    while not stop_event.is_set():
                        now = datetime.now()
                        timestamp = (now - start_time).total_seconds() + start_timestamp
                        new_image = self.oa.thermo_microscope.imaging.get_image()
                        if np.array_equal(new_image.data[:5], image.data[:5]) is False:
                            self.image_queue.put((new_image.data.copy(), i))
                            if update_callback:
                                update_callback(new_image.data, timestamp)
                            image = new_image
                            i += 1
                        else:
                            continue
            except Exception as e:
                print(f"[ERROR] Exception occurred: {e}")
            finally:
                self.oa.thermo_microscope.detector.camera_settings.emission.stop()
                self.oa.thermo_microscope.imaging.stop_acquisition()
                self.image_queue.put(None)
                writer_thread.join()
        else:
            if self.mode == 'manual':
                folder_path = self.manual_folder_path
            else:
                folder_path = os.path.join(self.oa.folder_path, f"{row}-Dataset")
            start_time = datetime.now()
            writer_thread = threading.Thread(target=self.image_writer, args=(folder_path, start_timestamp), daemon=True)
            writer_thread.start()
            i = 0
            sim = self.fl_data_simulation()
            image = sim()
            try:
                while not stop_event.is_set():
                    now = datetime.now()
                    timestamp = (now - start_time).total_seconds() + start_timestamp
                    new_image = sim()
                    if np.array_equal(new_image[:5], image[:5]) is False:
                        self.image_queue.put((new_image.copy(), i))
                        if update_callback:
                            update_callback(new_image, timestamp)
                        image = new_image
                        i += 1
                    else:
                        continue
            except Exception as e:
                print(f"[ERROR] Exception occurred: {e}")
            finally:
                self.image_queue.put(None)
                writer_thread.join()

    def run_move_to_stored_location(self, row):
        if self.oa.manufacturer != 'Demo':
            try:
                self.oa.thermo_microscope.imaging.set_active_device(8)
                self.oa.thermo_microscope.detector.retract()
                self.oa.autoloader_control(self.position_data[row]["grid"])
                stored_stage_position = self.oa.fib_microscope.move_stage_absolute(self.position_data[row]['stage_position'])
                if self.oa.stage_position_within_limits(limit=10,
                                                        target_position=stored_stage_position) is True:
                    image = tifffile.imread(os.path.join(self.oa.folder_path, f"{row}-image_ib.tif"))
                    shift_x, shift_y = self.stage_position_correction(image)
                    self.oa.fib_microscope.move_stage_relative(FibsemStagePosition(x=shift_x, y=-shift_y))
                    self.oa.thermo_microscope.detector.camera_settings.focus.value \
                        = self.position_data[row]['fl_settings']['objective_focus']
                    return True
                else:
                    return False
            except Exception as e:
                print(f"Moving back to the stored position failed because of: {e}")
                return False
        else:
            print('Moved stage.')
            return True

    def stage_position_correction(self, reference_image):
        current_image = self.imaging.acquire_image(hfw=self.hfw, save=False, autofocus=True)
        shift, response = cv2.phaseCorrelate(np.float64(current_image.data), np.float64(reference_image))
        pixelsize = self.hfw / np.shape(current_image.data)[1]
        return shift[0]*pixelsize, shift[1]*pixelsize



#######################################################################################################################
#####       Functions required for the sample setup for the automatic coincidence experiment
#######################################################################################################################

    def set_data_callback(self, callback):
        self.data_callback = callback

    def save_position_data(self):
        """
        This functions writes the data to a variable which is then saved to disk and can be used as backup for the
        setup.
        """
        self.position_data_stored = copy.deepcopy(self.position_data)
        self.position_data_stored[-1]['stage_position'] = {'x': self.position_data[-1]['stage_position'].x,
                                                    'y': self.position_data[-1]['stage_position'].y,
                                                    'z': self.position_data[-1]['stage_position'].z,
                                                    'r': self.position_data[-1]['stage_position'].r,
                                                    't': self.position_data[-1]['stage_position'].t}
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.position_data_stored, f, indent=4)
        except Exception as e:
            print(f"Error saving position data: {e}")

    def run_add_position(self, row, fl_roi):
        """
        This function is called once the user adds an item to the table. It will record the current stage position,
        objective focus, emission settings, open an image to draw the ROI for the FL targeting.
        """
        if self.oa.manufacturer != 'Demo':
            self.oa.autoloader_control()
            current_grid_number = self.oa.loaded_grid.id
            current_stage_position = self.oa.fib_microscope.get_stage_position()
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            fl_settings = {'emission_color': self.oa.thermo_microscope.detector.camera_settings.emission.type.value,
                           'filter_setting': self.oa.thermo_microscope.detector.camera_settings.filter.type.value,
                           'exposure_time': self.oa.thermo_microscope.detector.camera_settings.exposure_time.value,
                           'binning': self.oa.thermo_microscope.detector.camera_settings.binning.value,
                           'objective_focus': self.oa.thermo_microscope.detector.camera_settings.focus.value,
                           'brightness': self.oa.thermo_microscope.detector.brightness.value,
                           'roi': fl_roi}
        else:
            StagePosition = namedtuple('StagePosition', ['x', 'y', 'z', 'r', 't'])
            current_stage_position = StagePosition(100, 100, 500, 0, 0)
            current_grid_number = 1
            fl_settings = {
                'emission_color': 'red',
                'filter_setting': 'reflection',
                'exposure_time': 200,
                'binning': 4,
                'objective_focus': 7700,
                'brightness': 0.4,
                'roi': fl_roi}
        self.position_data.append({
            "grid": current_grid_number,
            "stage_position": current_stage_position,
            "fl_settings": fl_settings})
        self.save_position_data()
        self.imaging.acquire_image(hfw=self.hfw, beam_type='ion', autofocus=True, filename=f"{row}-image")
        return current_grid_number, current_stage_position, fl_settings

    def run_edit_position(self, row, fl_roi):
        """
        This functions allows to edit a previous position.
        """
        current_grid_number, current_stage_position, fl_settings = self.run_add_position(row, fl_roi)
        self.imaging.acquire_image(hfw=self.hfw, beam_type='ion', filename=f"{row}-image", autofocus=True)
        self.position_data[row]["grid"] = current_grid_number
        self.position_data[row]["stage_position"] = current_stage_position
        self.position_data[row]["fl_settings"] = fl_settings
        self.save_position_data()
        return current_grid_number, current_stage_position, fl_settings

    def run_delete_position(self, row):
        """
        This function deletes an entry from the table and the backup file.
        """
        del self.position_data[row]
        self.save_position_data()
        file_path = os.path.join(self.oa.folder_path, f"{row}-image_ib.tif")

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print("File does not exist.")

    def run_display_position(self, row):
        """
        This function displays the FIB image of the setup position.
        """
        file_path = os.path.join(self.oa.folder_path, f"{row}-image_ib.tif")
        img = mpimg.imread(file_path)

        plt.figure("Image Viewer")
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)

#######################################################################################################################
#####       Functions required for the execution of the experiment
#######################################################################################################################

    def start_coincidence_milling(self, beam_current, stop_event, resume=False):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(2)
            self.oa.thermo_microscope.beams.ion_beam.beam_current.value = beam_current
            if resume is False:
                self.oa.thermo_microscope.patterning.start()
                if self.data_callback:
                    self.data_callback('go', True)
            else:
                self.oa.thermo_microscope.patterning.resume()
            while not stop_event.is_set():
                time.sleep(0.1)

    def stop_coincidence_milling(self, pause=False):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(2)
            if self.pause_not_stop is False:
                self.oa.thermo_microscope.patterning.stop()
                self.oa.thermo_microscope.patterning.clear_patterns()
            else:
                self.oa.thermo_microscope.patterning.pause()
        else:
            print('Milling stopped!')

    def run_coincidence_experiment(self, callback, stop_event, start_timestamp=0.0,
                                   test=False, row=None, beam_current=None):

        def wait_and_finalize_imaging_thread():
            self.imaging_thread.join()
            print("[INFO] Fluorescence experiment stopped.")
            self.stop_coincidence_milling()
            print("[INFO] Milling stopped")

        if self.mode == 'auto' and test is False:
            for row in range(len(self.position_data)):
                fl_settings = self.position_data[row]['fl_settings']
                roi = fl_settings['roi']
                print("[INFO] Moving to stored sample position and insert iFLM objective to stored focus ...")
                status_update = self.run_move_to_stored_location(row)
                if status_update is True:
                    now = datetime.now()
                    self.working_distance = self.oa.fib_microscope.get_available_values(key='working_distance',
                                                                                        beam_type=BeamType.ION)
                    if self.oa.manufacturer != 'Demo':
                        self.oa.thermo_microscope.imaging.set_active_view(2)
                        self.imaging.acquire_image(hfw=self.hfw, folder_path=self.manual_folder_path,
                                                   beam_type='ion', working_distance=self.working_distance,
                                                   autofocus=True, filename=f"FIB-before-image")
                        "HERE I NEED TO CREATE THE PATTERN!"
                        self.milling_thread = threading.Thread(target=self.start_coincidence_milling,
                                                                   args=(beam_current, stop_event))
                        self.milling_thread.start()
                        time.sleep(0.5)
                        print("[INFO] Starting the milling ...")
                        self.imaging_thread = threading.Thread(target=self.run_serial_acquisition_fl_images,
                                                               args=(callback, stop_event, start_timestamp,
                                                                     row, fl_settings))
                        print("[INFO] Performing the fluorescence experiment ...")
                        self.imaging_thread.start()

                        wait_and_finalize_imaging_thread()
                        if self.pause_not_stop is False:
                            self.stop_coincidence_experiment()

        elif self.mode == 'auto' and test is True:
            fl_settings = self.position_data[row]['fl_settings']
            print("[INFO] Moving to stored sample position and insert iFLM objective to stored focus ...")
            status_update = self.run_move_to_stored_location(row)
            if status_update is True:
                self.grab_fluorescence_image(row, add_on='before')
                self.start_coincidence_milling(beam_current)
                time.sleep(0.5)
                print("[INFO] Starting the milling ...")
                self.imaging_thread = threading.Thread(target=self.run_serial_acquisition_fl_images,
                                                       args=(callback, stop_event, start_timestamp, row, fl_settings))
                print("[INFO] Performing the fluorescence experiment ...")
                self.imaging_thread.start()
                threading.Thread(target=wait_and_finalize_imaging_thread).start()

        elif self.mode == 'manual':
            now = datetime.now()
            if not hasattr(self, "manual_folder_path"):
                self.manual_folder_path = Path(
                    os.path.join(self.oa.folder_path, str(date.today()), now.strftime("%H-%M")))
                self.manual_folder_path.mkdir(parents=True, exist_ok=True)
                if self.data_callback:
                    self.data_callback('path', self.manual_folder_path)
            if self.selected_options['FIB Milling'] is True:
                beam_current = self.oa.fib_microscope.get("current", beam_type=BeamType.ION)
                self.working_distance = self.oa.fib_microscope.get("working_distance", beam_type=BeamType.ION)

                if self.oa.manufacturer != 'Demo':
                    self.oa.thermo_microscope.imaging.set_active_view(2)
                    patterns = self.oa.thermo_microscope.patterning.get_patterns()
                    if len(patterns) == 0:
                        print("[ERROR] Please create a milling pattern!")
                        return

                if start_timestamp == 0.0:
                    if self.selected_options['Before/After FIB Image'] is True:
                        self.imaging.acquire_image(hfw=self.hfw, folder_path=self.manual_folder_path,
                                                beam_type='ion', working_distance=self.working_distance,
                                                   autofocus=True, filename=f"FIB-before-image")

                    self.milling_thread = threading.Thread(target=self.start_coincidence_milling, args=(beam_current, stop_event))
                else:
                    resume = True
                    self.milling_thread = threading.Thread(target=self.start_coincidence_milling,
                                                           args=(beam_current, stop_event, resume))
                self.milling_thread.start()
                time.sleep(0.5)
                print("[INFO] Starting the milling ...")
                self.imaging_thread = threading.Thread(target=self.run_serial_acquisition_fl_images,
                                                        args=(callback, stop_event, start_timestamp))
                print("[INFO] Performing the fluorescence experiment ...")
                self.imaging_thread.start()


                wait_and_finalize_imaging_thread()
                if self.pause_not_stop is False:
                    self.stop_coincidence_experiment()
            elif self.selected_options['SEM Imaging'] is True:
                print("[INFO] Not yet implemented.")

    def stop_coincidence_experiment(self):
        if self.selected_options['After Z-Stack'] is True:
            self.acquire_fl_z_stack()
        if self.selected_options['After Reflection Image'] is True:
            self.grab_reflection_image()
        if self.selected_options['Before/After FIB Image'] is True:
            self.imaging.acquire_image(hfw=self.hfw, working_distance=self.working_distance,
                                       folder_path=self.manual_folder_path, beam_type='ion',
                                       autofocus=True, filename=f"FIB-after-image")
        if self.data_callback:
            self.data_callback('done', True)

    def acquire_fl_z_stack(self, row=None):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.camera_settings.filter.type.value = CameraFilterType.FLUORESCENCE
            self.oa.thermo_microscope.detector.camera_settings.binning.value = self.manual_binning
            self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = self.manual_exposure_time
            self.oa.thermo_microscope.detector.brightness.value = self.manual_brightness
            mid_focus = self.oa.thermo_microscope.detector.camera_settings.focus.value
            z_stack = []
            for i in range(-5, 5):
                z_stack.append(mid_focus + (i * 500e-9))
            # for i in range(-5, 5):
            #     self.oa.thermo_microscope.detector.camera_settings.focus.value = mid_focus + (i * 500e-6)
            #     image = self.grab_fl_live_image(save=False)
            #     if isinstance(image, np.ndarray):
            #         z_stack.append(image)
            #     else:
            #         z_stack.append(image.data)
            #     if self.mode == 'manual':
            #         tifffile.imwrite(os.path.join(self.manual_folder_path, "Z_stack_after.tif"), z_stack)
            #     else:
            #         tifffile.imwrite(os.path.join(self.oa.folder_path, f"{row}-Dataset", "Z_stack_after.tif"), z_stack)
            #self.oa.thermo_microscope.detector.camera_settings.focus.value = mid_focus
        else:
            z_stack = []
            for i in range(-5, 5):
                image = (np.random.rand(512, 512) * 255).astype(np.uint8)
                if isinstance(image, np.ndarray):
                    z_stack.append(image)
                else:
                    z_stack.append(image.data)
                if self.mode == 'manual':
                    tifffile.imwrite(os.path.join(self.manual_folder_path, "Z_stack_after.tif"), z_stack)
                else:
                    tifffile.imwrite(os.path.join(self.oa.folder_path, f"{row}-Dataset", "Z_stack_after.tif"), z_stack)

    #######################################################################################################################
    #####       Functions required for insitu automatic data processing
    #######################################################################################################################

    def auto_image_processing(self, image, timestamp, roi):
        print(image)
        print(intensity)
        print(roi)
        self.timestamp.append(timestamp)
        self.intensity.append(intensity)




    def id_intensity_drop(self, timestamp, intensity):
        time.sleep(10)
        self.coin_stop_event.set()

    def parameter_test_function_id_drop(self, thresholds=[-1.5, -2.0, -2.5, -3.0, -3.5],
                                        debounce_values=[1, 2, 3, 4], window_size=10):
        print("Started the parameter function.")
        results = {}
        z_scores = []
        self.rolling_means = []
        rolling_window = deque(maxlen=window_size)
        for val in self.y_data:
            rolling_window.append(val)
            if len(rolling_window) < window_size:
                self.rolling_means.append(None)
                z_scores.append(None)
                continue
            mean = statistics.mean(rolling_window)
            std = statistics.stdev(rolling_window)
            z = (val - mean) / std if std != 0 else 0
            self.rolling_means.append(mean)
            z_scores.append(z)


        for z_thresh in thresholds:
            for debounce in debounce_values:
                drop_counter = 0
                result = None
                for i, z in enumerate(z_scores):
                    if z is None:
                        continue
                    if z < z_thresh:
                        drop_counter += 1
                    else:
                        drop_counter = 0
                    if drop_counter >= debounce:
                        result = i
                        break
                results[(z_thresh, debounce)] = result if result is not None else "No trigger"

        print(f"{'Z_TH':>7} | {'DBNC':>5} | {'Trigger at Z-slice':>20}")
        print("-" * 38)
        for (z, d), trig in sorted(results.items()):
            print(f"{z:>7} | {d:>5} | {trig!s:>20}")
        x = list(range(len(self.y_data)))
        return self.y_data, rolling_means, results

    def image_writer(self, folder_path, timestamp):
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                item = self.image_queue.get()
                if item is None:
                    break
                img, idx = item
                executor.submit(self.save_image, img, idx, folder_path, timestamp)
                self.image_queue.task_done()

    def save_image(self, img, idx, folder_path, timestamp):

        def get_highest_image_index(folder_path):
            max_index = -1
            pattern = re.compile(r"image_(\d+(?:\.\d+)?)\.tif")  # matches float-like numbers

            for filename in os.listdir(folder_path):
                match = pattern.match(filename)
                if match:
                    try:
                        index = float(match.group(1))
                        max_index = max(max_index, index)
                    except ValueError:
                        continue

            return max_index if max_index >= 0 else None

        if timestamp == 0.0:
            path = os.path.join(folder_path, f"image_{idx:.1f}.tif")
        else:
            max_index = get_highest_image_index(folder_path)
            path = os.path.join(folder_path, f"image_{max_index+1+idx:.1f}.tif")
        tifffile.imwrite(path, img)

    def fl_data_simulation(self):
        image_size = (4000, 4000)
        spot_radius = 50
        drop_start_frame = np.random.uniform(500, 1000)
        drop_duration = 20
        frame_count = [0]
        spot_center = (int(image_size[0] // 2.5), int(image_size[1] // 1.8))

        def _create_circular_mask():
            Y, X = np.ogrid[:image_size[0], :image_size[1]]
            dist_from_center = np.sqrt((X - spot_center[1]) ** 2 + (Y - spot_center[0]) ** 2)
            return dist_from_center <= spot_radius

        spot_mask = _create_circular_mask()

        def next_frame():
            frame_count[0] += 1
            bg_mean = np.random.uniform(0, 5)
            bg_std = np.random.uniform(0, 3)
            img = np.random.normal(loc=bg_mean, scale=bg_std, size=image_size)
            if frame_count[0] < drop_start_frame:
                mean_intensity = 200
            elif frame_count[0] < drop_start_frame+drop_duration:
                progress = (frame_count[0] - drop_start_frame) / drop_duration
                # Sharper drop using quadratic curve; you can use progress**1.5 or **2 for sharper fall
                mean_intensity = 200 - 180 * progress ** 2
            else:
                mean_intensity = 20

            spot_intensity = np.random.normal(loc=mean_intensity, scale=5)
            spot_noise = np.random.normal(0, 5, size=image_size)
            img[spot_mask] = spot_intensity + spot_noise[spot_mask]
            return np.clip(img, 0, 255).astype(np.uint8)

        return next_frame

#######################################################################################################################
#####       Functions required for auto GIS/Sputter routines
#######################################################################################################################

    def run_auto_gis_sputter(self, setup_settings):
        if len(setup_settings["selected_grids"]) > 0:
            print(setup_settings)
            self.auto_gis.run_automated_process(setup_parameters=setup_settings)
        else:
            raise RuntimeError("No grids selected")






