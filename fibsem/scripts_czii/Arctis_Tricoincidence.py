"""
Signed by Thermo Fisher Scientific
EMTjGBbbF95caOFo8erdlg/7VYDuxy8g4cSZ8xjrbojTsPtv2dogxHHHd5Wlpr4W4/A67rC8nZ7Du5Ua6xSMqsJJaKiDxK81Z7EE8zymaKDuvQarKovYuqp6
rMwLMFx8/l7kdQwwnZFfd2CgObp+TdAdiDLaPT1hJB88na/ycwh6Q08kovDUKlgOlM4b0SiEdcoMHHqt6UGEjv09JsCN4l2nksQhReUDSpiYCuRfHDA=
"""
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
            microscope.connect()
        else:
            print('No tool selected.')
        microscope.authenticate()
        print("Connection establishment to Autoscript server...")
    except:
        print("Failed to establish a connection to the microscope")

class TriCoincidence:
    def __init__(self, emission_color, excitation_color, tool, path):
        self.emission_color = emission_color
        self.excitation_color = excitation_color
        self.path = path
        self.microscope = connect_to_autoscript(tool=tool)


    def define_roi(self):
        """
        Define a roi in the fluorescence image which will be used to calculate the average. This function grabs the image
        currently visible in the active view of the microscope.
        """
        def onselect(eclick, erelease):
            global roi_coords
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            roi_coords = (xmin, xmax, ymin, ymax)

        def on_ok_clicked(event):
            if roi_coords is not None:
                print(f"Final ROI confirmed: x={roi_coords[0]}:{roi_coords[1]}, y={roi_coords[2]}:{roi_coords[3]}")
                plt.close(fig)
            else:
                print("No ROI selected yet.")

        self.microscope.imaging.set_active_view(3)
        image = self.microscope.imaging.get_image()

        roi_coords = None

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(image, cmap='gray')
        ax.set_title("Draw ROI, then click OK to confirm")

        selector = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        ok_ax = plt.axes([0.4, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
        ok_button = Button(ok_ax, 'OK')
        ok_button.on_clicked(on_ok_clicked)
        plt.show()

        if roi_coords:
            plt.imshow(image[roi_coords[0]:roi_coords[1], roi_coords[2]:roi_coords[3]])
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


