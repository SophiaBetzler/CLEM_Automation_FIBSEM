import sys
sys.path.append('C:\Program Files\Thermo Scientific Autoscript')
sys.path.append('C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages')

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import *
from autoscript_sdb_microscope_client.structures import *
from autoscript_sdb_microscope_client.structures import GetImageSettings
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import numpy as np
import os
import queue
import cv2
import threading
from datetime import datetime
from pathlib import Path
from datetime import date

def connect_to_autoscript(tool='arctis'):
    try:
        microscope = SdbMicroscopeClient()
        if tool == 'arctis':
            microscope.connect("127.0.0.1")
        elif tool == 'hydra':
            microscope.connect("192.168.0.1", 7520)
        else:
            print('No tool selected.')
        print("Connection establishment to Autoscript server...")
        return microscope
    except Exception as e:
        print(f"Failed to establish a connection to the microscope: {e}")

class TriCoincidence:
    def __init__(self, color, mode, tool, path, exposure_time, intensity, binning):
        self.color = color
        self.mode = mode
        self.path = path
        self.microscope = connect_to_autoscript(tool=tool)
        self.image_queue = queue.Queue(maxsize=2000)
        self.exposure_time = exposure_time
        self.intensity = intensity
        self.binning = binning
        if tool == 'arctis':
            available_modes = ['Fluorescence', 'Reflection']
            available_colors = ['Blue', 'GreenYellow', 'Red', 'Violet']
            if mode not in available_modes:
                raise ValueError(f"Mode '{mode}' is not in available modes: {available_modes}")
            if color not in available_colors:
                raise ValueError(f"Color '{color}' is not in available colors: {available_colors}")

    def define_roi(self):
        """
        Define a ROI in the fluorescence image which will be used to calculate the average.
        This script will take the currently displayed image in the 3 view of XT as reference.
        """
        roi_coords = [None]  # Use a mutable object to capture updates

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            roi_coords[0] = (xmin, xmax, ymin, ymax)

        def on_ok_clicked(event):
            if roi_coords[0] is not None:
                print(
                    f"Final ROI confirmed: x={roi_coords[0][0]}:{roi_coords[0][1]}, y={roi_coords[0][2]}:{roi_coords[0][3]}")
                plt.close(fig)
            else:
                print("No ROI selected yet.")

        self.microscope.imaging.set_active_view(3)
        image = self.microscope.imaging.get_image()

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(image.data, cmap='gray')
        ax.set_title("Draw ROI, then click OK to confirm")

        selector = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True)

        ok_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        ok_button = Button(ok_ax, 'OK')
        ok_button.on_clicked(on_ok_clicked)
        plt.show()

        if roi_coords:
            return roi_coords[0]
        else:
            print("No ROI was selected.")

    def image_writer(self):
        while True:
            item = self.image_queue.get()
            if item is None:
                break
            img, idx = item
            cv2.imwrite(os.path.join(self.path, f"image_{idx:.3f}.png"), img)
            self.image_queue.task_done()

    def record_images(self, roi_coords=None, flashing=False):

        self.microscope.imaging.set_active_device(ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE)
        self.microscope.detector.camera_settings.exposure_time.value = self.exposure_time
        self.microscope.detector.camera_settings.emission.type.value = self.color
        self.microscope.detector.brightness.value = self.intensity
        self.microscope.detector.camera_settings.filter.type.value = 'Fluorescence'
        self.microscope.detector.camera_settings.binning.value = self.binning
        self.microscope.imaging.start_acquisition()
        writer_thread = threading.Thread(target=self.image_writer)
        writer_thread.start()

        running = True
        def on_key(event):
            nonlocal running
            if event.key == 'q':
                print("Quit key pressed.")
                running = False

        if roi_coords is not None:
            fig, ax = plt.subplots()
            line, = ax.plot([], [], 'b-')
            fig.canvas.mpl_connect('key_press_event', on_key)
            xdata = []
            ydata = []
            image_stack = []
            start_time = datetime.now()
            i = 0
            while running:
                if self.microscope.imaging.state == ImagingState.ACQUIRING:
                    if flashing is False:
                        now = datetime.now()
                        timestamp = (now - start_time).total_seconds()
                        image = self.microscope.imaging.get_image()
                        #image_stack.append(image.data)
                        av_intensity = np.nanmean(image.data[roi_coords[0]:roi_coords[1], roi_coords[2]: roi_coords[3]])
                        xdata.append(timestamp)
                        ydata.append(av_intensity)
                        i += 1
                    else:
                        now = datetime.now()
                        timestamp = (now - start_time).total_seconds()
                        self.microscope.detector.camera_settings.emission.start(emission_type=self.color)
                        image = self.microscope.imaging.get_image()
                        self.microscope.detector.camera_settings.emission.stop()
                        av_intensity = np.nanmean(image.data[roi_coords[0]:roi_coords[1], roi_coords[2]: roi_coords[3]])
                        xdata.append(timestamp)
                        ydata.append(av_intensity)
                        i += 1

                line.set_data(xdata, ydata)
                ax.relim()  # Recompute the data limits based on current xdata/ydata
                ax.autoscale_view()  # Update the view (zoom) to include new limits
                plt.pause(0.01)

            self.microscope.imaging.stop_acquisition()
            plt.close(fig)
            self.image_queue.put(None)
            writer_thread.join()

            return (xdata, ydata), np.array(image_stack)

        else:
            raise RuntimeError('No ROI selected!')

######################################
### These are the parameters we have to adjust during the acquisition, don't forget to change the path otherwise data
### might get overwritten!
#####################################

basic_path = 'C:\\Users\\User\\Desktop\\'
#basic_path = '/Users/sophia.betzler/Desktop'
now = datetime.now()
folder_path = Path(os.path.join(basic_path, str(date.today()), now.strftime("%H-%M")))
folder_path.mkdir(parents=True, exist_ok=True)


tri = TriCoincidence(emission_color='blue', excitation_color='blue', path=folder_path, tool='hydra',
                     exposure_time=100e-6, intensity=0.01, binning=1)
roi_coords = tri.define_roi()
if roi_coords:
    (x, y), image_stack = tri.record_images(roi_coords=roi_coords)
    with open(os.path.join(folder_path, 'intensity_list.txt'), 'w') as f:
        for xi, yi in zip(x, y):
            f.write(f"{xi}, {yi}\n")
    #np.save(os.path.join(folder_path, "avg_int.npy"), arr=image_stack, allow_pickle=False)


