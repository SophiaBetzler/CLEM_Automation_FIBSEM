from PyQt5.QtWidgets import QFormLayout

from fibsem import acquire, utils, microscope, structures, milling, calibration
from fibsem.structures import BeamType, FibsemStagePosition, FibsemDetectorSettings
import yaml
import os
from pathlib import Path
import numpy as np
import time
import cv2
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QFormLayout,
                             QVBoxLayout, QHBoxLayout, QMessageBox,
                             QRadioButton, QButtonGroup, QGridLayout, QCheckBox, QFileDialog)
from PyQt5.QtCore import Qt
from Basic_Functions import OverArch


class GisSputterAutomationGUI:
    """
    Sole purpose of this class is to open the GUI controlling the process.
    """
    def __init__(self, oa):
        self.oa = OverArch()
        self.app = QApplication(sys.argv)
        self.auto_gis = GisSputterAutomation(self.oa)
        self.input_window = InputWindow(self.auto_gis.run_automated_process, self.oa)
        self.run()

    def run(self):
        self.input_window.show()
        self.app.exec()

class GisSputterAutomation:

    def __init__(self, overarch: OverArch):
        self.project_root = Path(__file__).resolve().parent.parent
        self.oa = overarch
        self.before_stage_position = self.oa.fib_microscope.get_stage_position()

    def setup_sputtering(self, process_time, grid_number=None):
        if process_time == 0.0:
            print(f"Sputter step is skipped.")
        else:
            self.oa.thermo_microscope.patterning.clear_patterns()
            if self.oa.tool == 'Arctis':
                try:
                    self.oa.thermo_microscope.specimen.sputter_coater.current = (
                        self.oa.thermo_microscope.beams.ion_beam.beam_current.value)
                    self.oa.thermo_microscope.specimen.sputter_coater.prepare()
                    self.oa.thermo_microscope.specimen.sputter_coater.run(int(process_time))
                    self.oa.thermo_microscope.specimen.sputter_coater.recover()
                except Exception as e:
                    print(f"The sputtering process failed because of {e}.")

            elif self.oa.tool == 'Hydra':
                if grid_number is None:
                    raise RuntimeError("A grid number must be selected on the Hydra.")
                target_stage_position = self.oa.retrieve_stage_position(grid_number=grid_number, position_name='sputter')
                self.oa.fib_microscope.move_stage_absolute(target_stage_position)
                if self.oa.stage_position_within_limits(limit=5,
                                                target_position=target_stage_position) is True:
                    try:
                        pt_needle = self.oa.thermo_microscope.gas.get_gis_port('Pt dep')
                        pt_needle.insert()
                        pt_needle_pattern =[[1.45651561426744E-04, 6.64882312424326E-04],
                                               [-5.22964004557272E-04, 6.49570505569732E-04],
                                              [-7.27121429285215E-04, 5.27076050732976E-04],
                                               [-8.8862682243944E-04, 1.94128991749241E-04],
                                               [-9.31558247406688E-04, -5.13542946156679E-05],
                                               [-8.72306303805652E-04, -2.80703364890847E-04],
                                               [-7.64395328406466E-04, -4.54857094726658E-04],
                                               [-6.11405614772345E-04, -5.85789981686577E-04],
                                               [-2.96417971236887E-04, -6.63272670403714E-04],
                                               [-8.32068785097803E-06, -6.70840422732026E-04],
                                               [4.55219705511043E-04, -6.62604481026923E-04],
                                               [7.96403352747065E-04, -4.63087459197467E-04],
                                               [9.49574359688812E-04, -1.63231241628326E-04],
                                               [9.41403205733267E-04, 1.79649552180094E-04],
                                               [8.55704192307081E-04, 3.75876937069012E-04],
                                               [5.53009422954219E-04, 6.13842956242344E-04]]
                        self.oa.thermo_microscope.patterning.create_polygon(pt_needle_pattern, 5e-4)
                        self.oa.thermo_microscope.patterning.start()
                        time.sleep(process_time)
                        self.oa.thermo_microscope.patterning.stop()
                        self.oa.thermo_microscope.patterning.clear_patterns()
                        pt_needle.retract()
                    except Exception as e:
                        print(f"The sputter process failed because of {e}.")
                        pt_needle = self.oa.thermo_microscope.gas.get_gis_port('Pt dep')
                        pt_needle.retract()
                else:
                    raise RuntimeError("Stage position not correct for sputtering.")
            else:
                raise RuntimeError("Automatic Sputter/GIS Setup not available for this tool.")

    def setup_gis(self, process_time, grid_number=None):
        if process_time == 0:
            print(f"GIS step is skipped.")
        else:
            if grid_number is None:
                raise RuntimeError("A grid number must be selected on the Hydra.")
            target_stage_position = self.oa.retrieve_stage_position(grid_number=grid_number, position_name='gis')
            self.oa.fib_microscope.move_stage_absolute(target_stage_position)
            if self.oa.stage_position_within_limits(limit=5, target_position=target_stage_position) is True:
                try:
                    if self.oa.tool == 'Arctis':
                        print("[ERROR] Setup not yet tested for the Arctis. Please don't use.")
                        #gis_needle = self.oa.thermo_microscope.gas.get_gis_port('CRYO Pt ')
                        #gis_needle.turn_heater_on()
                        #gis_needle.insert()
                        #gis_needle.open()
                        #time.sleep(process_time)
                        #gis_needle.close()
                        #gis_needle.retract()
                    elif self.oa.tool == 'Hydra':
                        try:
                            multichem_needle = self.oa.thermo_microscope.gas.get_multichem()
                            multichem_needle.turn_heater_on('CRYO Pt ')
                            multichem_needle.insert()
                            self.oa.thermo_microscope.patterning.clear_patterns()
                            pattern = self.oa.thermo_microscope.patterning. \
                                        create_rectangle(center_x=0.0, center_y=0.0, width=2e-6, height=2e-6, depth=50e-6)
                            pattern.application_file = "W_M 12kV"
                            pattern.gas_type = 'CRYO Pt '
                            multichem_needle.open()
                            time.sleep(process_time + 2)
                            multichem_needle.close()
                            multichem_needle.retract()
                        except Exception as e:
                            print(f"The GIS setup failed because of {e}.")
                            multichem_needle = self.oa.thermo_microscope.gas.get_multichem()
                            multichem_needle.retract()
                        else:
                            raise RuntimeError(f"GIS setup not established for this tool.")
                except Exception as e:
                     print(f"The GIS layer failed because of {e}.")

    def run_automated_process(self, setup_parameters):
        try:
            if setup_parameters['settings_file_path'] is not None:
                ion_beam_settings = self.oa.read_from_yaml(setup_parameters['settings_file_path'],
                                                           imaging_settings_yaml=False)
            else:
                ion_beam_settings = {'Hydra': {'plasma_source': 'Xenon',
                                     'voltage': 12000,
                                     'beam_current': 0.12e-6},
                                     'Arctis': {'plasma_source': 'Xenon',
                                     'voltage': 12000,
                                     'beam_current': 70.0e-9}}

            self.oa.thermo_microscope.imaging.set_active_view(2)
            self.oa.thermo_microscope.imaging.set_active_device(2)
            self.oa.thermo_microscope.beams.ion_beam.turn_on()
            self.oa.thermo_microscope.beams.ion_beam.source.plasma_gas.value = ion_beam_settings[self.oa.tool]['plasma_source']
            self.oa.thermo_microscope.beams.ion_beam.high_voltage.value = ion_beam_settings[self.oa.tool]['voltage']
            self.oa.thermo_microscope.beams.ion_beam.beam_current.value = ion_beam_settings[self.oa.tool]['beam_current']
            self.oa.thermo_microscope.beams.ion_beam.scanning.rotation.value = 0.0
            self.oa.thermo_microscope.beams.ion_beam.horizontal_field_width.value = (
                self.oa.thermo_microscope.beams.ion_beam.horizontal_field_width.limits.max)

            if self.oa.tool == 'Arctis':
                print(self.oa.tool)
                print("[ERROR] Setup not yet tested for Arctis, will be skipped.")
            #    for grid in setup_parameters['grid_number']:
            #        self.oa.autoloader_control(grid)
            #        self.setup_sputtering(setup_parameters['sputter1'])
            #        self.setup_gis(setup_parameters['gis'])
            #        self.setup_sputtering(setup_parameters['sputter2'])
            elif self.oa.tool == 'Hydra':
                self.oa.thermo_microscope.specimen.stage.link()
                self.setup_sputtering(setup_parameters['sputter1'], setup_parameters['selected_grids'])
                if setup_parameters['sputter1'] != 0:
                    print('[INFO] Sputter step 1 DONE.')
                self.setup_gis(setup_parameters['gis'], setup_parameters['selected_grids'])
                self.setup_sputtering(setup_parameters['sputter2'], setup_parameters['selected_grids'])
                if setup_parameters['sputter2'] != 0:
                    print('[INFO] Sputter step 2 DONE.')
                self.oa.thermo_microscope.specimen.stage.unlink()
        except Exception as e:
            print(f"The automated sputter/GIS process failed because of: {e}")

    def run(self):
        self.input_window.show()
        sys.exit(self.app.exec_())

class InputWindow(QWidget):
    def __init__(self, on_submit_callback, oa):
        super().__init__()
        self.oa = oa
        self.setWindowTitle("Setup of the Automated Sputter/GIS Routine.")
        self.init_ui()
        self.on_submit_callback = on_submit_callback

    def init_ui(self):
        self.setup_times = {}
        labels = ["Sputter_Step1", "GIS_Step1", "Sputter_Step2"]

        form_layout = QFormLayout()
        for label in labels:
            line_edit = QLineEdit("0")
            self.setup_times[label] = line_edit
            form_layout.addRow(label + ":", line_edit)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.load_from_file_checkbox = QCheckBox("Load Settings from File")
        self.load_from_file_checkbox.stateChanged.connect(self.load_settings_file_if_checked)
        layout.addWidget(self.load_from_file_checkbox)

        # === Separate grid selection row for Hydra ===
        if self.oa.tool == 'Hydra':
            self.grid1_radio = QRadioButton("Grid 1")
            self.grid2_radio = QRadioButton("Grid 2")
            self.grid_group = QButtonGroup()
            self.grid_group.addButton(self.grid1_radio)
            self.grid_group.addButton(self.grid2_radio)

            hydra_grid_selection_layout = QHBoxLayout()
            hydra_grid_selection_layout.addWidget(self.grid1_radio)
            hydra_grid_selection_layout.addWidget(self.grid2_radio)
            layout.addLayout(hydra_grid_selection_layout)

        # === Arctis: vertical layout with checkboxes ===
        elif self.oa.tool == 'Arctis':
            grid_checkbox_layout = QVBoxLayout()

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
            layout.addLayout(grid_checkbox_layout)

        else:
            raise RuntimeError(f"GIS/Sputter Setup Automation not available for this tool {self.oa.tool}.")

        # === Submit button ===
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.readin_input_values)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def readin_input_values(self):
        try:
            sputter1_time = float(self.setup_times["Sputter_Step1"].text())
            gis_time = float(self.setup_times["GIS_Step1"].text())
            sputter2_time = float(self.setup_times["Sputter_Step2"].text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter numeric times.")
            return

        if self.oa.tool == 'Hydra':
            if self.grid1_radio.isChecked():
                grid_number = 1
                self.oa.grid_number = 1
            elif self.grid2_radio.isChecked():
                grid_number = 2
                self.oa.grid_number = 2
            else:
                grid_number = None
                self.oa.grid_number = None
        elif self.oa.tool == 'Arctis':
            grid_number = [grid_id for grid_id, checkbox in self.grid_checkboxes.items()
                             if checkbox.isChecked()]
        else:
            raise RuntimeError("Automatic GIS/Sputter setup not working for this tool.")

        setup_params = {
            "selected_grids": grid_number,
            "sputter1": sputter1_time,
            "gis": gis_time,
            "sputter2": sputter2_time,
            "settings_file_path": getattr(self, "settings_file_path", None)}

        self.on_submit_callback(setup_params)
        self.close()

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

        # # HERE I SWITCH TO AUTOSCRIPT
        #     pt_needle = self.thermo_microscope.gas.get_gis_port('µSputter')
        #     pt_needle.insert()
        #     try:
        #         self.thermo_microscope.set_active_view(2)
        #         self.thermo_microscope.patterns.clear_patterns()
        #         self.thermo_microscope.beams.ion_beam.source.plasma_gas.value = self.fib_microscope.PlasmaGasType.ARGON
        #         self.thermo_microscope.beams.ion_beam.beam_current.value = 15.0e-9
        #         self.thermo_microscope.beams.ion_beam.scanning.rotation = 0.0
        #         self.thermo_microscope.beams.ion_beam.high_voltage.value = 12000.0
        #         Pt_needle = np.array([[1.45651561426744E-04, 6.64882312424326E-04],
        #                                    [-5.22964004557272E-04, 6.49570505569732E-04],
        #                                    [-7.27121429285215E-04, 5.27076050732976E-04],
        #                                    [-8.8862682243944E-04, 1.94128991749241E-04],
        #                                    [-9.31558247406688E-04, -5.13542946156679E-05],
        #                                    [-8.72306303805652E-04, -2.80703364890847E-04],
        #                                    [-7.64395328406466E-04, -4.54857094726658E-04],
        #                                    [-6.11405614772345E-04, -5.85789981686577E-04],
        #                                    [-2.96417971236887E-04, -6.63272670403714E-04],
        #                                    [-8.32068785097803E-06, -6.70840422732026E-04],
        #                                    [4.55219705511043E-04, -6.62604481026923E-04],
        #                                    [7.96403352747065E-04, -4.63087459197467E-04],
        #                                    [9.49574359688812E-04, -1.63231241628326E-04],
        #                                    [9.41403205733267E-04, 1.79649552180094E-04],
        #                                    [8.55704192307081E-04, 3.75876937069012E-04],
        #                                    [5.53009422954219E-04, 6.13842956242344E-04]], dtype=float)
        #         needle_width = np.max(Pt_needle[:, 0])-np.min(Pt_needle[:, 0])
        #         needle_height = np.max(Pt_needle[:, 1])-np.min(Pt_needle[:, 1])
        #         mask = np.zeros((1024, 1536), dtype=np.uint8)
        #         needle_pattern_scaled = (Pt_needle * 0.95 * np.array([1536, 1024]) /
        #                          (2 * np.max(np.abs(Pt_needle), axis=0)))
        #         needle_pattern_int32 = np.round(needle_pattern_scaled + np.array([1536 / 2, 1024 / 2])).astype(np.int32)
        #         mask_pattern = np.array(cv2.fillPoly(mask, [needle_pattern_int32], color=1))
        #         needle_milling_bitmap = np.zeros(mask_pattern.shape + (2,), dtype=int)
        #         needle_milling_bitmap[mask_pattern == 1, 1] = 1
        #         needle_bitmap_pattern = self.thermo_microscope.BitmapPatternDefinition()
        #         needle_bitmap_pattern.points = np.array(needle_milling_bitmap)
        #         pattern = self.thermo_microscope.microscope.patterning.create_bitmap(0, 0, needle_width,
        #                                                                              needle_height, 10e-6,
        #                                                                              needle_bitmap_pattern)
        #         pattern.application_file = "Si" # if you don't set this there might
        #         pattern.start()
        #         print("Milling is running.")
        #         time.sleep(time)
        #         pattern.stop()
        #         print("Milling stopped.")
        #         self.thermo_microscope.patterns.clear_patterns()
        #
        #         pt_needle.retract()
        #     except Exception as e:
        #         print(f"The sputter process failed because of {e}.")
        #         pt_needle.retract()





