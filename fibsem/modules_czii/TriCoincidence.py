from Basic_Functions import OverArch
from fibsem import utils, structures, microscope
from fibsem.structures import FibsemStagePosition
from GIS_Sputter_Setup import GisSputterAutomation
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import copy
import queue
import platform
import time
pc_type = platform.system()
if pc_type == 'Windows':
    matplotlib.use('Qt5Agg')
elif pc_type == 'Darwin':
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Button
#from autoscript_sdb_microscope_client import SdbMicroscopeClient
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QFormLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox, QCheckBox, QGridLayout, QFileDialog
)
from PyQt5.QtCore import Qt, QEventLoop, QObject, pyqtSignal, QThread
from collections import namedtuple
import json
import matplotlib.image as mpimg
import os
from Imaging import Imaging
import tifffile
import cv2
import threading
from datetime import datetime

class AutomatedTriCoincidenceGUI:
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
        self.hfw = 80.0e-6
        self.auto_gis = GisSputterAutomation(self.oa)
        self.lock  = threading.Lock()
        self.milling_stop_event = threading.Event()


#######################################################################################################################
#####       Functions controlling the Arctis
#######################################################################################################################

    def grab_fl_live_image(self):
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            image = self.oa.thermo_microscope.imaging.get_image()
            #image.save(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
            tifffile.imwrite(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"), image.data)
        else:
            image = np.random.rand(1536, 1024)
            tifffile.imwrite(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"), image)
        return image

    def grab_fluorescence_image(self, row, add_on):
        self.oa.thermo_microscope.imaging.set_active_device(3)
        self.oa.thermo_microscope.imaging.set_active_device(8)
        self.oa.thermo_microscope.detector.camera_settings.binning.value = self.position_data[row]['fl_settings']
        ['binning']
        self.oa.thermo_microscope.detector.brightness.value = self.position_data[row]['brightness']
        self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = self.position_data[row]['fl_settings']
        ['exposure_time']
        self.oa.thermo_microscope.detector.camera_settings.filter.value = self.position_data[row]['fl_settings']
        ['filter_setting']
        self.oa.thermo_microscope.detector.camera_settings.color.value = self.position_data[row]['fl_settings']
        ['emission_color']
        self.oa.thermo_microscope.detector.camera_settings.focus.value = self.position_data[row]['fl_settings']
        ['objective_focus']
        image = self.oa.thermo_microscope.imaging.grab_frame()
        image.save(os.path.join(self.oa.folder_path, f"{row}_fl_image_{add_on}.tif"))

    def run_serial_acquisition_fl_images(self, row, fl_settings, update_callback=None):
        self.image_queue = queue.Queue(maxsize=2000)
        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_view(3)
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.camera_settings.exposure_time.value = fl_settings['exposure_time']
            self.oa.thermo_microscope.detector.brightness.value = fl_settings['brightness']
            self.oa.thermo_microscope.detector.camera_settings.binning.value = fl_settings['binning']
            self.oa.thermo_microscope.detector.camera_settings.filter.type.value = fl_settings['filter_setting']
            self.oa.thermo_microscope.detector.camera_settings.color.value = fl_settings['emission_color']
            self.oa.thermo_microscope.imaging.start_acquisition()
            writer_thread = threading.Thread(target=self.image_writer, args=(row,), daemon=True)
            writer_thread.start()

            if fl_settings['roi'] is not None:
                x_data = []
                y_data = []
                start_time = datetime.now()
                run_tricoincidence_experiment = True
                i=0
                if self.oa.thermo_microscope.imaging.state == ImagingState.ACQUIRING:
                    while run_tricoincidence_experiment is True:
                        now = datetime.now()
                        timestamp = (now - start_time).total_seconds()
                        image = self.oa.thermo_microscope.imaging.get_image()
                        self.image_queue.put((image.data.copy(), i))
                        self.oa.thermo_microscope.detector.camera_settings.emission.start(emission_type=
                                                                                          fl_settings['emission_color'])
                        av_intensity = np.nanmean(image.data[fl_settings['roi'][0]:fl_settings['roi'][1],
                                                  fl_settings['roi'][2]: fl_settings['roi'][3]])
                        x_data.append(timestamp)
                        y_data.append(av_intensity)
                        if update_callback:
                            update_callback(timestamp, av_intensity)
                        ### DESIGN THE FITTING FUNCTION HERE
                        run_tricoincidence_experiment = self.intensity_profile_analysis(y_data, f)
                        i+=1
                self.oa.thermo_microscope.detector.camera_settings.emission.stop()
                self.oa.thermo_microscope.imaging.stop_acquisition()
                self.image_queue.put(None)
                writer_thread.join()
            ### POTENTIALLY JUST DIRECTLY WRITE THE DATA TO THE DISK HERE, CHECK WHERE IMAGE_WRITER STORES THE FILES
                self.tricoincidence_result_queue = (x_data, y_data)
            else:
                raise RuntimeError('No ROI selected!')
        else:
            x_data = []
            y_data = []
            start_time = datetime.now()
            writer_thread = threading.Thread(target=self.image_writer, args=(row,), daemon=True)
            writer_thread.start()
            run_tricoincidence_experiment = True
            i = 0
            while run_tricoincidence_experiment is True:
                now = datetime.now()
                timestamp = (now - start_time).total_seconds()
                image = np.random.rand(1536, 1024)
                self.image_queue.put((image.copy(), i))
                av_intensity = np.nanmean(image[100:200,
                                          140:200])
                x_data.append(timestamp)
                y_data.append(av_intensity)
                if update_callback:
                    update_callback(timestamp, av_intensity)
                i += 1
            self.image_queue.put(None)
            writer_thread.join()
            self.tricoincidence_result_queue = (x_data, y_data)

    def run_move_to_stored_location(self, row):
        try:
            self.oa.autoloader_control(self.position_data[row]["grid"])
            stored_stage_position = self.oa.fib_microscope.move_stage_absolute(self.position_data[row]['stage_position'])
            if self.oa.stage_position_within_limits(limit=10,
                                                    target_position=stored_stage_position) is True:
                image = tifffile.imread(os.path.join(self.oa.folder_path, f"{row}-image_ib.tif"))
                shift_x, shift_y = self.stage_position_correction(image)
                self.oa.fib_microscope.move_stage_relative(FibsemStagePosition(x=shift_x, y=-shift_y))
                self.oa.thermo_microscope.detector.camera_settings.focus.value = self.position_data[row]['fl_settings']['objective_focus']
                return True
            else:
                return False
        except Exception as e:
            print(f"Moving back to the stored position failed because of: {e}")
            return False

    def stage_position_correction(self, reference_image):
        current_image = self.imaging.acquire_image(hfw=self.hfw, beam_type='ion', save=False, autofocus=True)
        shift, response = cv2.phaseCorrelate(np.float64(current_image.data), np.float64(reference_image))
        pixelsize = self.hfw / np.shape(current_image.data)[1]
        return shift[0]*pixelsize, shift[1]*pixelsize

    def stop_milling_tricoincidence(self):
        self.milling_stop_event.set()

#######################################################################################################################
#####       Functions required for the manual sample setup
#######################################################################################################################

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

        plt.figure("Image Viewer")  # Optional: window title
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)

#######################################################################################################################
#####       Functions required for the automatic processing
#######################################################################################################################

    def run_tricoincidence_experiment(self, fl_settings, row, beam_current, fitting_parameters):
        serial_acquisition_thread = threading.Thread(target=self.run_serial_acquisition_fl_images,
                                                       args=(row, fl_settings))
        milling_thread = threading.Thread(target=self.milling_tricoincidence, args=(beam_current,))
        serial_acquisition_thread.start()
        milling_thread.start()

        self.tricoincidence_result_queue = queue.Queue()  # Maybe not needed


    def stop_tricoincidence_experiment(self):
        "stop milling and imaging functions in parallel and write data to disk."

    def run_automated_experiment(self):
        for i in range(len(self.position_data)):
            fl_settings = self.position_data[i]['fl_settings']
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.retract()
            print("Retracting objective ...")
            status_update = self.run_move_to_stored_location(i)
            print("Moving to stored sample position and insert iFLM objective to stored focus...")
            if status_update is True:
                self.grab_fluorescence_image(i, add_on='before')
                print("Performing tricoincidence experiment ...")
                self.run_tricoincidence_experiment(fl_settings)
                print("Finished tricoincidence milling experiment.")

    def run_test_experiment(self, row, update_callback):
        self.tricoincidence_result_queue = queue.Queue()

        if self.oa.manufacturer != 'Demo':
            self.oa.thermo_microscope.imaging.set_active_device(8)
            self.oa.thermo_microscope.detector.retract()
            print("Retracting objective ...")
            status_update = self.run_move_to_stored_location(row)
            print("Moving to stored sample position and insert iFLM objective to stored focus...")
            self._running = True
            #### HERE I SHOULD ALSO MAKE SURE TO INCLUDE THE IMAGE ACQUISITION
            self.run_serial_acquisition_fl_images(row=row,
                                                  fl_settings=self.position_data[row]['fl_settings'],
                                                  update_callback=update_callback)
        else:
            self._running = True
            self.run_serial_acquisition_fl_images(row=row,
                                                 fl_settings=self.position_data[row]['fl_settings'],
                                                 update_callback=update_callback)


    def stop_test_experiment(self):
        self._running = False
        self.milling_stop_event.set()

    def image_writer(self, row):
        while True:
            item = self.image_queue.get()
            if item is None:
                break
            img, idx = item
            path = os.path.join(self.oa.folder_path, f"{row}-Dataset", f"image_{idx:.3f}.png")
            cv2.imwrite(path, img)
            self.image_queue.task_done()

#######################################################################################################################
#####       Functions required for auto GIS/Sputter routines
#######################################################################################################################

    def run_auto_gis_sputter(self, setup_settings):
        if len(setup_settings["selected_grids"]) > 0:
            print(setup_settings)
            self.auto_gis.run_automated_process(setup_parameters=setup_settings)
        else:
            raise RuntimeError("No grids selected")


class GUIforTriCoincidence(QWidget):
    def __init__(self, oa):
        super().__init__()
        self.oa = oa
        self.selector = None
        self.ok_button = None
        self.tricoincidence = TriCoincidence(self.oa)
        self.setWindowTitle("Setup of the TriCoincidence Routine")

        # === GIS / SPUTTER SECTION ===
        gis_sputter_full_layout = QVBoxLayout()
        gis_sputter_title = QLabel("GIS / Sputter Setup")
        gis_sputter_title.setStyleSheet("font-weight: bold; font-size: 14px")
        gis_sputter_description = QLabel("Please select all grids which should be used and set the correct "
                                         "sputter and GIS times. The process step will be skipped if the "
                                         "time is set to 0.0 s.\n"
                                         "The default conditions are Xenon, 30kV, 0.15 nA.")
        gis_sputter_description.setStyleSheet("color: gray; font-size: 11px")

        gis_sputter_full_layout.addWidget(gis_sputter_title)
        gis_sputter_full_layout.addWidget(gis_sputter_description)

        gis_sputter_setup_layout = QHBoxLayout()
        gis_sputter_setup_layout.setSpacing(10)

        gis_sputter_input_boxes_layout = QVBoxLayout()
        gis_sputter_input_boxes_layout.setAlignment(Qt.AlignLeft)
        row_layout = QFormLayout()
        row_layout.setContentsMargins(50, 0, 0, 0)  # Optional: tight layout
        row_layout.setLabelAlignment(Qt.AlignLeft)
        row_layout.setFormAlignment(Qt.AlignLeft)
        self.setup_times = {}
        labels = ["Sputter_Step1", "GIS_Step1", "Sputter_Step2"]
        self.setup_times = {}
        for label_text in labels:
            line_edit = QLineEdit("0")
            self.setup_times[label_text] = line_edit
            row_layout.addRow(label_text + ":", line_edit)
        form_widget = QWidget()
        form_widget.setLayout(row_layout)
        gis_sputter_input_boxes_layout.addWidget(form_widget)
        gis_sputter_setup_layout.addLayout(gis_sputter_input_boxes_layout)

        grid_checkbox_layout = QVBoxLayout()
        self.load_from_file_checkbox = QCheckBox("Load Settings from File")
        self.load_from_file_checkbox.stateChanged.connect(self.load_settings_file_if_checked)
        grid_checkbox_layout.addWidget(self.load_from_file_checkbox)

        self.unselect_all_checkbox = QCheckBox("Unselect All")
        self.unselect_all_checkbox.stateChanged.connect(self.unselect_all_grids)
        grid_checkbox_layout.addWidget(self.unselect_all_checkbox)

        self.grid_checkboxes = {}
        checkbox_grid = QGridLayout()
        grids = self.oa.available_grids
        columns = 4
        for i, grid in enumerate(grids):
            checkbox = QCheckBox(f"Grid {grid.id}")
            checkbox.setChecked(True)
            self.grid_checkboxes[grid.id] = checkbox
            row = i // columns
            col = i % columns
            checkbox_grid.addWidget(checkbox, row, col)

        grid_checkbox_layout.addLayout(checkbox_grid)
        gis_sputter_setup_layout.addLayout(grid_checkbox_layout)

        gis_sputter_full_layout.addLayout(gis_sputter_setup_layout)

        gis_sputter_buttons = QHBoxLayout()
        self.start_gis_button = QPushButton("Start")
        self.start_gis_button.setFixedWidth(120)
        self.start_gis_button.clicked.connect(self.start_gis_sputter)
        gis_sputter_buttons.addWidget(self.start_gis_button)
        self.abort_gis_button = QPushButton("Abort")
        self.abort_gis_button.setFixedWidth(120)
        self.abort_gis_button.clicked.connect(self.start_gis_sputter)
        gis_sputter_buttons.addWidget(self.abort_gis_button)
        gis_sputter_buttons.setAlignment(Qt.AlignLeft)
        gis_sputter_buttons.setContentsMargins(50, 0, 0, 0)

        gis_sputter_full_layout.addLayout(gis_sputter_buttons)


        # === POSITION SETUP SECTION ===
        position_setup_title = QLabel("Setup of the Positions")
        position_setup_title.setStyleSheet("font-weight: bold; font-size: 14px")

        position_setup_description = QLabel("Please select all position of interest on all grids. Make sure to adjust"
                                            "the optical focus and click 'Add'. \nPositions can be edited or deleted"
                                            " by selection the respective row.")
        position_setup_description.setStyleSheet("color: gray; font-size: 11px")


        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Grid", "Stage Position", "Objective Focus", "ROI", "Status"])

        side_buttons = QVBoxLayout()
        add_button = QPushButton("Add")
        add_button.setFixedWidth(120)
        add_button.clicked.connect(self.add_position)
        side_buttons.addWidget(add_button)

        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self.edit_selected)
        edit_button.setFixedWidth(120)
        side_buttons.addWidget(edit_button)

        delete_button = QPushButton("Delete")
        delete_button.setFixedWidth(120)
        delete_button.clicked.connect(self.delete_selected)
        side_buttons.addWidget(delete_button)

        import_button = QPushButton("Import")
        import_button.setFixedWidth(120)
        import_button.clicked.connect(self.import_from_file)
        side_buttons.addWidget(import_button)

        display_button = QPushButton("Display")
        display_button.setFixedWidth(120)
        display_button.clicked.connect(self.display_selected)
        side_buttons.addWidget(display_button)

        side_buttons.addStretch()

        table_layout = QVBoxLayout()
        table_layout.addWidget(self.table)

        position_setup_layout = QHBoxLayout()
        position_setup_layout.addLayout(table_layout, stretch=4)
        position_setup_layout.addLayout(side_buttons, stretch=1)

        # === DIVIDER ===
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)

        # === PROCESS PROGRESS SECTION ===
        progress_title = QLabel("Start the Automated Experiment")
        progress_title.setStyleSheet("font-weight: bold; font-size: 14px")

        progress_buttons = QHBoxLayout()
        progress_buttons.setAlignment(Qt.AlignLeft)
        progress_buttons.setContentsMargins(50, 0, 0, 0)
        testexperiment_button = QPushButton("Test-Experiment")
        testexperiment_button.clicked.connect(self.test_experiment)
        progress_buttons.addWidget(testexperiment_button)
        testexperiment_button.setFixedWidth(150)
        start_progress_button = QPushButton("Start")
        start_progress_button.clicked.connect(self.progress_start_button_clicked)
        progress_buttons.addWidget(start_progress_button)
        start_progress_button.setFixedWidth(120)
        abort_progress_button = QPushButton("Abort")
        abort_progress_button.clicked.connect(self.progress_abort_button_clicked)
        abort_progress_button.setFixedWidth(120)
        progress_buttons.addWidget(abort_progress_button)

        progress_layout = QVBoxLayout()
        progress_layout.addWidget(progress_title)
        progress_layout.addLayout(progress_buttons)

        # === FINAL LAYOUT ===
        main_layout = QVBoxLayout()
        main_layout.addLayout(gis_sputter_full_layout)
        main_layout.addWidget(divider)
        main_layout.addWidget(position_setup_title)
        main_layout.addWidget(position_setup_description)
        main_layout.addLayout(position_setup_layout)
        main_layout.addWidget(divider)
        main_layout.addLayout(progress_layout)
        self.setLayout(main_layout)

    def error_messagebox(self, text):
        box = QMessageBox()
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Warning")
        box.setText(text)
        box.setStandardButtons(QMessageBox.Abort)
        choice = box.exec_()
        if choice == QMessageBox.Abort:
            return

#######################################################################################################################
#####       Functions controlling the manual setup of the sample position
#######################################################################################################################P

    def add_position(self):
        row = self.table.rowCount()
        self.tricoincidence.grab_fl_live_image()
        fl_roi = self.define_roi()
        grid_number, stage_position, fl_settings = self.tricoincidence.run_add_position(row, fl_roi)
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(grid_number)))
        self.table.setItem(row, 1, QTableWidgetItem(f"x: {np.round(stage_position.x*1e6, 1)}, "
                                                    f"y: {np.round(stage_position.y*1e6, 1)}, "
                                                    f"z: {np.round(stage_position.z*1e6, 1)}"))
        self.table.setItem(row, 2, QTableWidgetItem(str(fl_settings['objective_focus']*1000)))
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
            self.table.setItem(row, 1, QTableWidgetItem(f"x: {np.round(stage_position.x * 1e6, 1)}, "
                                                        f"y: {np.round(stage_position.y * 1e6, 1)}, "
                                                        f"z: {np.round(stage_position.z * 1e6, 1)}"))
            self.table.setItem(row, 2, QTableWidgetItem(str(fl_settings['objective_focus']*1000)))
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
                self.table.setItem(row, 1, QTableWidgetItem(f"x:{np.round(stage_pos['x'] * 1e6, 1)}, "
                                                            f"y:{np.round(stage_pos['y'] * 1e6, 1)}, "
                                                            f"z:{np.round(stage_pos['z'] * 1e6, 1)}"))
                self.table.setItem(row, 2, QTableWidgetItem(str(str(data["objective_focus"]*1000))))
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

#######################################################################################################################
#####       Functions controlling the GIS/Sputter setup
#######################################################################################################################

    def unselect_all_grids(self):
        if self.unselect_all_checkbox.isChecked():
            for cb in self.grid_checkboxes.values():
                cb.setChecked(False)
            self.unselect_all_checkbox.setChecked(False)

    def load_settings_file_if_checked(self, state):
        if state == Qt.Checked:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Grid List File", "", "YAML Files (*.yaml);;All Files (*)")
            self.settings_file_path = None
            if file_path:
                with open(file_path, 'r') as f:
                    self.settings_file_path = file_path
            else:
                self.load_from_file_checkbox.setChecked(False)
                self.settings_file_path = None
        elif state == Qt.Unchecked:
            self.settings_file_path = None

    def readin_input_values(self):
        try:
            sputter1_time = float(self.setup_times["Sputter_Step1"].text())
            gis_time = float(self.setup_times["GIS_Step1"].text())
            sputter2_time = float(self.setup_times["Sputter_Step2"].text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter numeric times.")
            return

        grid_number = [grid_id for grid_id, checkbox in self.grid_checkboxes.items()
                             if checkbox.isChecked()]

        setup_params = {
            "selected_grids": grid_number,
            "sputter1": sputter1_time,
            "gis": gis_time,
            "sputter2": sputter2_time,
            "settings_file_path": getattr(self, "settings_file_path", None)}

        return setup_params

    def start_gis_sputter(self):
        setup_params = self.readin_input_values()
        self.tricoincidence.run_auto_gis_sputter(setup_params)

#######################################################################################################################
#####       Functions which control the automatic tricoincidence experiment
#######################################################################################################################

    def experiment_test(self):
        selected = self.table.selectionModel().selectedRows()
        if selected:
            row = selected[0].row()
            self.tricoincidence.run_move_to_stored_location(row)
            self.tricoincidence.run_test_experiment()

    def progress_start_button_clicked(self):
        self.tricoincidence.run_automated_experiment()

    def progress_abort_button_clicked(self):
       print("Abort button clicked.")

#######################################################################################################################
#####       Functions which open separate plotting windows
#######################################################################################################################

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

        image = tifffile.imread(os.path.join(self.oa.temp_folder_path, f"fl_image.tif"))
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
            # plt.imshow(image[roi_coords[0][2]:roi_coords[0][3], roi_coords[0][0]:roi_coords[0][1]])
            # plt.show()
            return roi_coords[0]
        else:
            print("No ROI was selected.")
            return None

    def test_experiment(self):
        """
        Runs a test experiment which allows the user to manually stop the milling and data-recording. It will automatically
        terminate after 30 minutes.
        """
        def update_plot(timestamp, intensity):
            if not self.test_experiment_running:
                return
            x_data.append(timestamp)
            y_data.append(intensity)
            line.set_data(x_data, y_data)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        def stop_test_experiment(event=None):
            print("Stop button pressed.")
            self.test_experiment_running = False
            self.tricoincidence.stop_test_experiment()

        def start_test_experiment():
            self.test_thread = threading.Thread(
                            target=self.tricoincidence.run_test_experiment,
                            args=(row, update_plot))
            self.test_thread.start()

            try:
                while self.test_thread.is_alive():
                    if not self.test_experiment_running:
                        break
                    plt.pause(0.1)
            except KeyboardInterrupt:
                stop_test_experiment()
                self.test_thread.join()
                print("Stopped.")

        def stop_test_experiment_after_delay(obj, delay_seconds):
            time.sleep(delay_seconds)
            print("[WARNING] Maximal experiment time exceeded (30 minutes).")
            obj.stop_test_experiment()

        threading.Thread(target=stop_test_experiment_after_delay, args=(self.tricoincidence, 1800)).start()

        selected = self.table.selectionModel().selectedRows()
        if selected:
            row = selected[0].row()
        else:
            self.error_messagebox("Please select a sample position to run the test experiment.")
            return

        self.test_experiment_running = True
        x_data = []
        y_data = []

        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'b-')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity")
        plt.subplots_adjust(bottom=0.3)
        ax_stop = plt.axes([0.4, 0.1, 0.2, 0.075])
        stop_button = Button(ax_stop, 'STOP')
        stop_button.on_clicked(stop_test_experiment)
        plt.ion()
        plt.show()
        start_test_experiment()








