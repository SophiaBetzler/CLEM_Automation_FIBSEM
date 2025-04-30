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
    def __init__(self, emission_color, excitation_color, tool, path):
        self.emission_color = emission_color
        self.excitation_color = excitation_color
        self.path = path
        self.microscope = connect_to_autoscript(tool=tool)

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
            interactive=True
        )

        ok_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        ok_button = Button(ok_ax, 'OK')
        ok_button.on_clicked(on_ok_clicked)

        plt.show()  # Blocking call, waits until GUI is closed

        # This executes only after the window is closed
        if roi_coords[0]:
            xmin, xmax, ymin, ymax = roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3]
            plt.imshow(image[ymin:ymax, xmin:xmax])  # Note: y = rows, x = cols
            plt.show()
            return roi_coords
        else:
            print("No ROI was selected.")

    def record_images(self, roi_coords=None, flashing=False):

        #self.microscope.imaging.set_active_device(ImagingDevice.FLUORESCENCE_MICROSCOPE)
        #self.microscope.imaging.start_acquisition()
        self.microscope.imaging.set_active_device(ImagingDevice.ELECTRON_BEAM)
        self.microscope.imaging.start_acquisition()

        running = True
        def on_key(event):
            global running
            if event.key == 'q':
                print("Quit key pressed.")
                running = False
                plt.savefig(os.path.join(self.path, 'Target_Position_FIB_Focus_Shifted.png'), dpi=300, bbox_inches='tight')

        if roi_coords is not None:
            fig, ax = plt.subplots()
            line, = ax.plot([], [], 'b-')
            fig.canvas.mpl_connect('key_press_event', on_key)
            xdata = []
            ydata = []
            image_stack = []
            i = 0
            while running:
                self.microscope.imaging.state == ImagingState.ACQUIRING
                if flashing is False:
                    image = self.microscope.imaging.get_image()
                    image_stack.append(image)
                    intensity = image[roi_coords[0]:roi_coords[1], roi_coords[2]: roi_coords[3]]
                    xdata.append(i)
                    ydata.append(intensity)
                else:
                    #Here I would like to turn on the flash!
                    print('Not yet implemented')
                line.set_data(xdata, ydata)
                ax.relim()  # Recompute the data limits based on current xdata/ydata
                ax.autoscale_view()  # Update the view (zoom) to include new limits
                plt.pause(0.05)

            self.microscope.imaging.stop_acquisition()

            return (xdata, ydata), np.array(image_stack)

        else:
            raise RuntimeError('No ROI selected!')

######################################
### These are the parameters we have to adjust during the acquisition, don't forget to change the path otherwise data
### might get overwritten!
#####################################

folder_path = 'set path here'
tri = TriCoincidence(emission_color='blue', excitation_color='blue', path=folder_path, tool='hydra')
roi_coords = tri.define_roi()
if roi_coords:
    intensity_values, image_stack = tri.record_images()
    with open(os.path.join(folder_path + 'intensity_list.txt'), 'w') as f:
        for item in intensity_values:
            f.write(f"{item}\n")
    np.save(path=folder_path, arr=image_stack, allow_pickle=False)


