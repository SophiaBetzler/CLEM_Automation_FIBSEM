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
                             QRadioButton, QButtonGroup)
from Basic_Functions import BasicFunctions

class GisSputterAutomation(BasicFunctions):

    def __init__(self):
        super().__init__()
        self.project_root = Path(__file__).resolve().parent.parent
        self.before_stage_position = self.fib_microscope.get_stage_position()
        self.app = QApplication(sys.argv)
        self.input_window = InputWindow(self.run_automated_process)
        self.run()

    def setup_sputtering(self, time):
        if time == 0:
            print(f"Sputter step is skipped.")
        else:
            self.thermo_microscope.beams.ion_beam.turn_on()
            self.thermo_microscope.beams.ion_beam.source.plasma_gas.value = 'ARGON'
            self.thermo_microscope.beams.ion_beam.high_voltage = 12000
            self.thermo_microscope.beams.ion_beam.beam_current = 60e-9
            self.thermo_microscope.beams.ion_beam.scanning.rotation = 0.0
            self.thermo_microscope.beams.ion_beam.horizontal_field_width = (
                self.thermo_microscope.beams.ion_beam.horizontal_field_width.limits.min)
            self.thermo_microscope.patterning.clear_patterns()
            if self.tool == 'Arctis':
                try:
                    self.thermo_microscope.specimen.sputter_coater.current = 60e-9
                    self.thermo_microscope.specimen.sputter_coater.run(int(time))
                except Exception as e:
                    print(f"The sputtering process failed because of {e}.")
            elif self.tool == 'Hydra':
                target_stage_position = self.retrieve_stage_position(self.grid_number, 'sputter')
                self.fib_microscope.move_stage_absolute(target_stage_position)
                if self.stage_position_within_limits(limit=5,
                                                 target_position=target_stage_position) is True:
                    pt_needle = self.thermo_microscope.gas.get_gis_port('µSputter')
                    pt_needle.insert()
                    pt_needle_pattern = np.array([[1.45651561426744E-04, 6.64882312424326E-04],
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
                                          [5.53009422954219E-04, 6.13842956242344E-04]], dtype=float)
                    self.thermo_microscope.patterning.create_polygon(pt_needle_pattern, 5e-4)
                    self.thermo_microscope.patterning.start()
                    time.sleep(time)
                    self.thermo_microscope.patterning.stop()
                    self.thermo_microscope.patterning.clear_patterns()
            else:
                RuntimeError("Automatic Sputter/GIS Setup not available for this tool.")

    def setup_gis(self, time):
        if time == 0:
            print(f"GIS step is skipped.")
        else:
            target_position = self.retrieve_stage_position(self.grid_number, 'gis')
            self.fib_microscope.move_stage_absolute(target_position)
            if self.stage_position_within_limits(limit=5, target_position=target_position) is True:
                try:
                    if self.tool == 'Arctis':
                        gis_needle = self.thermo_microscope.gas.get_gis_port('CRYO Pt ')
                        gis_needle.turn_heater_on()
                        gis_needle.insert()
                        gis_needle.open()
                        time.sleep(time)
                        gis_needle.close()
                        gis_needle.retract()
                    elif self.tool == 'Hydra':
                        multichem_needle = self.thermo_microscope.gas.get_multichem()
                        multichem_needle.turn_heater_on('CRYO Pt ')
                        if multichem_needle.state == 'Retracted':
                            multichem_needle.insert()
                            self.thermo_microscope.beams.ion_beam.turn_on()
                            self.thermo_microscope.beams.ion_beam.source.plasma_gas.value = 'ARGON'
                            self.thermo_microscope.beams.ion_beam.high_voltage = 12000
                            self.thermo_microscope.beams.ion_beam.beam_current = 60e-9
                            self.thermo_microscope.beams.ion_beam.scanning.rotation = 0.0
                            self.thermo_microscope.beams.ion_beam.horizontal_field_width = (
                                self.thermo_microscope.beams.ion_beam.horizontal_field_width.limits.min)
                            self.thermo_microscope.patterning.clear_patterns()
                            pattern = self.thermo_microscope.microscope.patterning. \
                                create_rectangle(center_x=0.0, center_y=0.0, width=2e-6, height=2e-6, depth=50e-6)
                            pattern.application_file = "W_M e"
                            pattern.gas_type = 'CRYO Pt '
                            pattern.gas_flow = [80.0]
                            # self.thermo_microscope.microscope.patterning.start()    # ?? DO I NEED THIS?
                            multichem_needle.open()
                            time.sleep(time + 2)
                            multichem_needle.close()
                            multichem_needle.retract()
                    else:
                        RuntimeError(f"The GIS setup process failed, correct stage position not reached.")
                except Exception as e:
                    print(f"The GIS layer failed because of {e}.")

    def run_automated_process(self, input_values):
        try:
            if input_values['Grid'] is not None:
                self.grid_number = int(input_values['Grid'])
            else:
                self.grid_number = None
            self.setup_sputtering(input_values['Sputter_Step1'])
            self.setup_gis(input_values['GIS_Step1'])
            self.setup_sputtering(input_values['Sputter_Step2'])
        except Exception as e:
            print(f"The automated sputter/GIS process failed because of: {e}")

    def run(self):
        self.input_window.show()
        sys.exit(self.app.exec_())


class InputWindow(QWidget):
    def __init__(self, on_submit_callback):
        super().__init__()
        self.setWindowTitle("Setup of the Automated Sputter/GIS Routine.")
        self.init_ui()
        self.on_submit_callback = on_submit_callback

    def init_ui(self):
        self.inputs = {}
        labels = ["Sputter_Step1", "GIS_Step1", "Sputter_Step2"]

        form_layout = QFormLayout()
        for label in labels:
            line_edit = QLineEdit("0")
            self.inputs[label] = line_edit
            form_layout.addRow(label + ":", line_edit)

        self.grid1_radio = QRadioButton("Grid 1")
        self.grid2_radio = QRadioButton("Grid 2")
        self.grid_group = QButtonGroup()
        self.grid_group.addButton(self.grid1_radio)
        self.grid_group.addButton(self.grid2_radio)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.grid1_radio)
        radio_layout.addWidget(self.grid2_radio)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.read_values)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(radio_layout)
        layout.addWidget(self.submit_button)
        self.setLayout(layout)

    def read_values(self):
        try:
            values = {label: float(field.text()) for label, field in self.inputs.items()}
            if self.grid1_radio.isChecked():
                values["Grid"] = "1"
            elif self.grid2_radio.isChecked():
                values["Grid"] = "2"
            else:
                values["Grid"] = None
            self.on_submit_callback(values)
            self.close()
            return values
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter numeric values only.")



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





