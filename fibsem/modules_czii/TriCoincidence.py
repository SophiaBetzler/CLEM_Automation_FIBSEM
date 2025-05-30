from Basic_Functions import BasicFunctions, OverArch
from fibsem import utils, structures, microscope
import matplotlib
import platform
pc_type = platform.system()
if pc_type == 'Windows':
    matplotlib.use('TkAgg')
elif pc_type == 'Darwin':
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Button
#from autoscript_sdb_microscope_client import SdbMicroscopeClient
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox
)
from PyQt5.QtCore import Qt, QEventLoop
from collections import namedtuple
import json
import matplotlib.image as mpimg
import os
from Imaging import Imaging
import tifffile
import cv2

class AutomatedTriCoincidence():
    """
    Sole purpose of this class is to open the GUI controlling the process.
    """
    def __init__(self, oa):
        self.oa = OverArch()
        self.app = QApplication(sys.argv)
        self.window = GUIforTriCoincidence(self.oa)
        self.run()

    def run(self):
        self.window.show()
        self.app.exec()


class TriCoincidence:
    def __init__(self, oa):
        self.oa = oa
        if self.oa.tool != 'Arctis':
            raise RuntimeError("This is not the right tool to run the automated tricoincidence routine.")
        self.imaging = Imaging(self.oa)
        self.oa.autoloader_control()
        self.position_data = []
        self.data_file = os.path.join(self.oa.folder_path, 'position_data.json')

    def run_automated_process(self):
        print("Result from add position:")

    def save_position_data(self):
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.position_data, f, indent=4)
        except Exception as e:
            print(f"Error saving position data: {e}")

    def grab_fl_live_image(self):
        #self.oa.thermo_microscoope.imaging.set_active_view(3)
        #image = self.oa.thermo_microscope.imaging.get_image()
        #image.save(os.path.join(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
        image = np.random.rand(500, 500)
        tifffile.imwrite(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"), image)
        return image

    def run_add_position(self, row, fl_roi):
        """
        This function is called once the user adds an item to the table. It will record the current stage position,
        objective focus, emission settings, open an image to draw the ROI for the FL targeting.
        """
        if self.oa.manufacturer != 'Demo':
            for i in len(self.oa.available_grids):
                if self.oa.available_grids[i].state == 'Loaded':
                    current_grid_number = i
                    current_stage_position = self.oa.fib_microscope.get_stage_position()
                    self.oa.thermo_microscope.imaging.set_active_view(3)
                    self.oa.thermo_microscope.imaging.set_active_device(ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE)
                    fl_settings = {'emission_color': self.oa.thermo_microscope.detector.camera_settings.emission.type.value,
                                   'filter_setting': self.oa.thermo_microscope.detector.camera_settings.filter.value,
                                   'exposure_time': self.oa.thermo_microscope.detector.camera_settings.exposure_time.value,
                                   'binning': self.oa.thermo_microscope.detector.camera_settings.binning.value,
                                   'objective_focus': self.oa.thermo_microscope.detector.camera_settings.focus.value,
                                   'roi': fl_roi,
                                   'filter': self.oa.thermo_microscope.detector_camera_settings.filter.type.value,
                                   }
                else:
                    raise RuntimeWarning("Not valid grid loaded in selected position.")
        else:
            StagePosition = namedtuple('StagePosition', ['x', 'y', 'z'])
            current_stage_position = StagePosition(100, 100, 500)
            current_grid_number = 1
            fl_settings = {
                'emission_color': 'red',
                'filter_setting': 'reflection',
                'exposure_time': 200,
                'binning': 4,
                'objective_focus': 7700,
                'roi': fl_roi,
                'filter': 'fluorescence',
                    }
        self.position_data.append({
            "grid": current_grid_number,
            "stage_position": {
                "x": current_stage_position.x,
                "y": current_stage_position.y,
                "z": current_stage_position.z},
            "fl_settings": fl_settings})
        self.save_position_data()
        self.imaging.acquire_image(hfw=600.0e-6, beam_type='ion', autofocus=True, filename=f"{row}-image")
        return current_grid_number, current_stage_position, fl_settings

    def run_edit_position(self, row, fl_roi):
        if self.oa.manufacturer != 'Demo':
            for i in len(self.oa.available_grids):
                if self.oa.available_grids[i].state == 'Loaded':
                    current_grid_number = i
                    current_stage_position = self.oa.fib_microscope.get_stage_position()
                    self.oa.thermo_microscope.imaging.set_active_view(3)
                    self.oa.thermo_microscope.imaging.set_active_device(ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE)
                    fl_settings = {'emission_color': self.oa.thermo_microscope.detector.camera_settings.emission.type.value,
                                   'filter_setting': self.oa.thermo_microscope.detector.camera_settings.filter.value,
                                   'exposure_time': self.oa.thermo_microscope.detector.camera_settings.exposure_time.value,
                                   'binning': self.oa.thermo_microscope.detector.camera_settings.binning.value,
                                   'objective_focus': self.oa.thermo_microscope.detector.camera_settings.focus.value,
                                   'roi': fl_roi,
                                   'filter': self.oa.thermo_microscope.detector_camera_settings.filter.type.value,
                                   }
                else:
                    raise RuntimeWarning("Not valid grid loaded in selected position.")
        else:
            StagePosition = namedtuple('StagePosition', ['x', 'y', 'z'])
            current_stage_position = StagePosition(100, 100, 500)
            current_grid_number = 1
            fl_settings = {
                'emission_color': 'red',
                'filter_setting': 'reflection',
                'exposure_time': 200,
                'binning': 4,
                'objective_focus': 7700,
                'roi': fl_roi,
                'filter': 'fluorescence'
                    }
        self.imaging.acquire_image(hfw=600.0e-6, beam_type='ion', filename=f"{row}-image", autofocus=True)
        self.position_data[row]["grid"] = current_grid_number
        self.position_data[row]["stage_position"] = {
            "x": current_stage_position.x,
            "y": current_stage_position.y,
            "z": current_stage_position.z
        }
        self.position_data[row]["fl_settings"] = fl_settings
        self.save_position_data()
        return current_grid_number, current_stage_position, fl_settings

    def run_delete_position(self, row):
        del self.position_data[row]
        self.save_position_data()
        file_path = os.path.join(self.oa.folder_path, f"{row}-image_ib.tif")

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print("File does not exist.")

    def run_display_position(self, row):
        file_path = os.path.join(self.oa.folder_path, f"{row}-image_ib.tif")
        img = mpimg.imread(file_path)

        plt.figure("Image Viewer")  # Optional: window title
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)

    def stage_position_correction(self, reference_image):
        current_image = self.imaging.acquire_image(hfw=600.0e-6, save=False, autofocus=True)
        result = cv2.matchTemplate(current_image, reference_image, method=cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        dx, dy = max_loc
        w, _ = current_image.size()
        pixelsize = 600.0e-6 / w
        print(f"The required shift is {dx*pixelsize} and {dy*pixelsize}")
        return dx*pixelsize, dy*pixelsize

    def grab_fluorescence_image(self, row, add_on):
        self.oa.thermo_microscope.imaging.set_active_device(3)
        self.oa.thermo_microscope.imaging.set_active_device(ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE)
        self.oa.thermo_microscope.detector.camera_settings.binning.value = self.position_data[row]['fl_settings']
        ['binning']
        self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = self.position_data[row]['fl_settings']
        ['exposure_time']
        self.oa.thermo_microscope.detector.camera_settings.filter.value = self.position_data[row]['fl_settings']
        ['filter_setting']
        self.oa.thermo_microscope.detector.camera_settings.color.value = self.position_data[row]['fl_settings']
        ['emission_color']
        self.oa.thermo_microscope.detector.camera_settings.focus.value = self.position_data[row]['fl_settings']
        ['objective_focus']
        image = self.oa.thermo_microscope.imaging.get_image()
        image.save(os.path.join(self.oa.folder_path, f"{row}_fl_image_{add_on}.tif"))

    def run_move_to_stored_location(self, row):
        try:
            stored_stage_position = self.oa.thermo_microscope.specimen.autoloader.load(self.position_data['grid_number'])
            self.oa.fib_microscope.move_stage_absolute(self.position_data['stage_position'])
            if self.oa.stage_position_within_limits(limit=10,
                                                    target_position=stored_stage_position) is True:
                image = AdornedImage.load(os.path.join(self.oa.folder_path, f"{row}-fl_image.tif"))
                shift_x, shift_y = self.stage_position_correction(image)
                self.oa.fib_microscope.move_stage_relative(x=shift_x, y=shift_y)
                self.oa.thermo_microscope.detector.camera_settings.focus.value = self.position_data['fl_settings']
                ['objective_focus']
                return True
            else:
                return False
        except Exception as e:
            print(f"Moving back to the stored position failed because of: {e}")
            return False

    def tricoincidence_milling_setup(self):
        #### Beam-Coincidence between FL and Ion Beam Image!! Do I need to store the Ion Beam Shifts?
        self.oa.thermo_microscope.imaging.set_active_view(2)
        self.oa.thermo_microscope.imaging.set_active_device(ImagingDevice.ION_BEAM)
        self.oa.thermo_microscope.beams.ion_beam.beam_current = 0.1e-9
        self.oa.thermo_microscope.beams.ion_beam.high_voltage = 30000
        self.oa.thermo_microscope.beams.ion_beam.horizontal_field_width = 80.0e-6
        self.oa.thermo_microscope.beams.ion_beam.scanning.rotation.value = 0.0
        pattern = self.oa.thermo_microscope.patterning.create_cleaning_cross_section(0, 0, 10.0e-6, 50.0e-6, 1.0e-6)
        pattern.application_file = "Si-ccs"
        return pattern


    def run_automated_experiment(self, row):
        self.oa.thermo_microscope.imaging.set_active_device(ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE)
        self.oa.thermo_microscope.detector.retract()
        print("Retracting objective ...")
        status_update = self.run_move_to_stored_location(row)
        print("Moving to stored sample position ...")
        if status_update is True:
            self.grab_fluorescence_image(row, add_on='before')
            print("Performing tricoincidence experiment ...")
            pattern = self.tricoincidence_milling_setup()

class GUIforTriCoincidence(QWidget):
    def __init__(self, oa):
        super().__init__()
        self.oa = oa
        self.selector = None
        self.ok_button = None
        self.tricoincidence = TriCoincidence(self.oa)
        self.setWindowTitle("Setup of the TriCoincidence Routine")

        # === POSITION SETUP SECTION ===
        position_setup_title = QLabel("Setup of the Positions")
        position_setup_title.setStyleSheet("font-weight: bold; font-size: 14px")

        position_setup_description = QLabel("Please select all position of interest on all grids. Make sure to adjust"
                                            "the optical focus and click 'Add'. \nPositions can be edited or deleted"
                                            " by selection the respective row.")
        position_setup_description.setStyleSheet("color: gray; font-size: 11px")


        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Grid", "Stage Position", "Objective Focus", "ROI", "Status"])

        # Vertical side buttons (now: Add, Edit, Delete)
        side_buttons = QVBoxLayout()

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_position)
        side_buttons.addWidget(add_button)

        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self.edit_selected)
        side_buttons.addWidget(edit_button)

        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_selected)
        side_buttons.addWidget(delete_button)

        import_button = QPushButton("Import")
        import_button.clicked.connect(self.import_from_file)
        side_buttons.addWidget(import_button)

        display_button = QPushButton("Display")
        display_button.clicked.connect(self.display_selected)
        side_buttons.addWidget(display_button)

        side_buttons.addStretch()  # Push buttons to the top

        # Table layout only (no bottom row buttons anymore)
        table_layout = QVBoxLayout()
        table_layout.addWidget(self.table)

        # Combine table + side buttons
        position_setup_layout = QHBoxLayout()
        position_setup_layout.addLayout(table_layout, stretch=4)
        position_setup_layout.addLayout(side_buttons, stretch=1)

        # === DIVIDER ===
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)

        # === GIS / SPUTTER SECTION ===
        gis_sputter_title = QLabel("GIS / Sputter Setup")
        gis_sputter_title.setStyleSheet("font-weight: bold; font-size: 14px")
        gis_sputter_description = QLabel("Please select all grids which should be used and set the correct"
                                         "sputter and GIS times. The process step will be skipped if the"
                                         "time is set to 0.0 s.")
        gis_sputter_description.setStyleSheet("color: gray; font-size: 11px")

        gis_sputter_input_layout = QVBoxLayout()
        gis_sputter_input_layout.setAlignment(Qt.AlignLeft)  # Align left

        sputter1_container = QHBoxLayout()
        sputter1_label = QLabel("Sputter Step 1:")
        sputter1_label.setFixedWidth(100)
        self.sputter1 = QSpinBox()
        self.sputter1.setValue(0)
        self.sputter1.setMaximumWidth(100)
        sputter1_container.addWidget(sputter1_label)
        sputter1_container.addWidget(self.sputter1)

        gis_container = QHBoxLayout()
        gis_label = QLabel("GIS Step:")
        gis_label.setFixedWidth(100)
        self.gis = QSpinBox()
        self.gis.setValue(0)
        self.gis.setMaximumWidth(100)
        gis_container.addWidget(gis_label)
        gis_container.addWidget(self.gis)

        sputter2_container = QHBoxLayout()
        sputter2_label = QLabel("Sputter Step 2:")
        sputter2_label.setFixedWidth(100)
        self.sputter2 = QSpinBox()
        self.sputter2.setValue(0)
        self.sputter2.setMaximumWidth(100)
        sputter2_container.addWidget(sputter2_label)
        sputter2_container.addWidget(self.sputter2)

        for container in [sputter1_container, gis_container, sputter2_container]:
            widget = QWidget()
            widget.setLayout(container)
            gis_sputter_input_layout.addWidget(widget)

        gis_sputter_buttons = QHBoxLayout()
        for label in ["Start", "Abort"]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, l=label: self.gis_sputter_button_clicked(l))
            gis_sputter_buttons.addWidget(btn)

        gis_sputter_layout = QVBoxLayout()
        gis_sputter_layout.addWidget(gis_sputter_title)
        gis_sputter_layout.addWidget(gis_sputter_description)
        gis_sputter_layout.addLayout(gis_sputter_input_layout)
        gis_sputter_layout.addLayout(gis_sputter_buttons)

        gis_sputter_title = QLabel("GIS / Sputter Setup")
        gis_sputter_title.setStyleSheet("font-weight: bold; font-size: 14px")

        # === PROCESS PROGRESS SECTION ===
        progress_title = QLabel("Start the Automated Experiment")
        progress_title.setStyleSheet("font-weight: bold; font-size: 14px")

        progress_buttons = QHBoxLayout()
        for label in ["Start", "Abort"]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, l=label: self.progress_button_clicked(l))
            progress_buttons.addWidget(btn)

        progress_layout = QVBoxLayout()
        progress_layout.addWidget(progress_title)
        progress_layout.addLayout(progress_buttons)

        # === FINAL LAYOUT ===
        main_layout = QVBoxLayout()
        main_layout.addWidget(position_setup_title)
        main_layout.addWidget(position_setup_description)
        main_layout.addLayout(position_setup_layout)
        main_layout.addWidget(divider)
        main_layout.addLayout(gis_sputter_layout)
        main_layout.addWidget(divider)
        main_layout.addLayout(progress_layout)
        self.setLayout(main_layout)

    def add_position(self):
        row = self.table.rowCount()
        self.tricoincidence.grab_fl_live_image()
        fl_roi = self.define_roi()
        grid_number, stage_position, fl_settings = self.tricoincidence.run_add_position(row, fl_roi)
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(grid_number)))
        self.table.setItem(row, 1, QTableWidgetItem(f"x: {np.round(stage_position.x*1000, 1)}, "
                                                    f"y: {np.round(stage_position.y*1000, 1)}, "
                                                    f"z: {np.round(stage_position.z*1000, 1)}"))
        self.table.setItem(row, 2, QTableWidgetItem(str(fl_settings['objective_focus'])))
        self.table.setItem(row, 3, QTableWidgetItem(str(fl_settings['roi'])))
        self.table.resizeColumnsToContents()


    def delete_selected(self):
        selected = self.table.selectionModel().selectedRows()
        for index in sorted(selected, key=lambda x: x.row(), reverse=True):
            row = index.row()
            self.table.removeRow(row)
            self.tricoincidence.run_delete_position(row)

    def edit_selected(self):
        selected = self.table.selectionModel().selectedRows()
        if selected:
            row = selected[0].row()
            fl_roi = self.define_roi()
            grid_number, stage_position, fl_settings = self.tricoincidence.run_edit_position(row, fl_roi)
            self.table.setItem(row, 0, QTableWidgetItem(str(grid_number)))
            self.table.setItem(row, 1, QTableWidgetItem(f"x: {np.round(stage_position.x * 1000, 1)}, "
                                                        f"y: {np.round(stage_position.y * 1000, 1)}, "
                                                        f"z: {np.round(stage_position.z * 1000, 1)}"))
            self.table.setItem(row, 2, QTableWidgetItem(str(fl_settings['objective_focus'])))
            self.table.setItem(row, 3, QTableWidgetItem(str(fl_settings['roi'])))
            self.table.resizeColumnsToContents()

    def import_from_file(self):
        try:
            with open(self.data_file, 'r') as f:
                self.position_data = json.load(f)
                i = 0
            for data in self.position_data:
                row = i
                stage_pos = data["stage_position"]
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(str(data["grid"])))
                self.table.setItem(row, 1, QTableWidgetItem(f"x:{np.round(stage_pos['x'] * 1000, 1)}, "
                                                            f"y:{np.round(stage_pos['y'] * 1000, 1)}, "
                                                            f"z:{np.round(stage_pos['z'] * 1000, 1)}"))
                self.table.setItem(row, 2, QTableWidgetItem(str(str(data["objective_focus"]))))
                self.table.setItem(row, 3, QTableWidgetItem(str(data["roi"])))
                i = i+1

        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading position data: {e}")

    def display_selected(self):
        selected = self.table.selectionModel().selectedRows()
        if selected:
            row = selected[0].row()
            self.tricoincidence.run_display_position(row)

    def gis_sputter_button_clicked(self, label):
        print(f"{label} clicked — Input1: {self.sputter1.value()}, Input2: {self.gis.value()}, Input3: {self.sputter2.value()}")

    def progress_button_clicked(self, label):
        print(f"{label} clicked.")

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
                if self.event_loop:
                    self.event_loop.quit()
            else:
                print("No ROI selected yet.")

        #image = AdornedImage.load(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
        image = np.random.rand(500, 500)
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(image.data, cmap='gray')
        ax.set_title("Draw ROI, then click OK to confirm")

        self.selector = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True)

        ok_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.ok_button = Button(ok_ax, 'OK')
        self.ok_button.on_clicked(on_ok_clicked)
        plt.show(block=False)
        self.event_loop = QEventLoop()
        self.event_loop.exec_()

        file_path = os.path.join(self.oa.temp_folder_path, f"fl_image.tif")

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print("File does not exist.")

        if roi_coords:
            return roi_coords[0]
        else:
            print("No ROI was selected.")
            return None