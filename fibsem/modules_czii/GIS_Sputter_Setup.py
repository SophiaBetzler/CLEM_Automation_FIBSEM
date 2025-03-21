from fibsem import acquire, utils, microscope, structures, milling, calibration
from fibsem.structures import BeamType, FibsemStagePosition, FibsemDetectorSettings
import yaml
import os
from pathlib import Path
import numpy as np
import time
import cv2

class GisSputterAutomation:

    def __init__(self, bf, fib_microscope, grid_number):
        self.fib_microscope = fib_microscope
        self.grid_number = grid_number
        self.bf = bf
        self.project_root = Path(__file__).resolve().parent.parent
        with open(os.path.join(self.project_root, 'config', 'czii-stored-stage-positions.yaml'), "r") as file:
            self.saved_stage_positions = yaml.safe_load(file)
        self.thermo_microscope = self.fib_microscope.connection


    def retrieve_stage_position(self, position_name):
        """
        Imports the stage position from the config file.
        """
        dict_position_names = {'sputter': '_sputter_position',
                               'gis': '_GIS_position',
                                'mapping': '_mapping_position'
                                }

        stage_position = next(
            (d for d in self.saved_stage_positions if d.get("name") == f"grid{self.grid_number}{dict_position_names[position_name]}"), None)
        if stage_position:
            fibsem_stage_position = FibsemStagePosition(x=stage_position['x']/1000,
                                                 y=stage_position['y']/1000,
                                                 z=stage_position['z']/1000,
                                                 r=np.deg2rad(stage_position['r']),
                                                 t=np.deg2rad(stage_position['t']))
            return fibsem_stage_position
        else:
            print('Stage position retrieval not valid.')

    def setup_sputtering(self, time):
        #self.fib_microscope.move_stage_absolute(self.retrieve_stage_position(self.grid_number, 'sputter'))

        #target_stage_position = self.retrieve_stage_position('sputter')
        #if self.bf.stage_position_within_limits(limit=5, current_position=current_stage_position,
        #                                  target_position=target_stage_position) is True:
        #self.fib_microscope.connection.gas.list_all_gis_ports()
            pt_needle = self.thermo_microscope.gas.get_gis_port('µSputter')
            pt_needle.insert()
            try:
                self.fib_microscope.set("plasma_gas", 'Argon',
                                    beam_type=BeamType.ION)
                self.fib_microscope.set("plasma", True,
                                    beam_type=BeamType.ION)
                Pt_needle = np.array([[1.45651561426744E-04, 6.64882312424326E-04],
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
                needle_width = np.max(Pt_needle[:, 0])-np.min(Pt_needle[:, 0])
                needle_height = np.max(Pt_needle[:, 1])-np.min(Pt_needle[:, 1])
                mask = np.zeros((1024, 1536), dtype=np.uint8)
                needle_pattern_scaled = (Pt_needle * 0.95 * np.array([1536, 1024]) /
                                 (2 * np.max(np.abs(Pt_needle), axis=0)))
                needle_pattern_int32 = np.round(needle_pattern_scaled + np.array([1536 / 2, 1024 / 2])).astype(np.int32)
                mask_pattern = np.array(cv2.fillPoly(mask, [needle_pattern_int32], color=1))
                bitmap_pattern_definition_needle = self.thermo_microscope.BitmapPatternDefinition()
                bitmap_pattern_definition_needle.points = (
                    np.stack([mask_pattern, np.zeros_like(mask_pattern)], axis=-1).astype(float))
                needle_pattern = (self.thermo_microscope.patterning.
                                  create_bitmap(-needle_width/2, needle_height/2, needle_width, needle_height, 1e-6,
                                                bitmap_pattern_definition_needle))
                self.thermo_microscope.patterns.clear_patterns()
                needle_pattern.application_file = "Si"
                needle_pattern.time = time
                sputter_milling_settings = structures.FibsemMillingSettings(
                    milling_current=10e-9,
                    milling_voltage=12e3,
                    hfw=1200e-6,
                    application_file="Si",
                    patterning_mode="Serial")
                milling_alignment = structures.MillingAlignment(enabled=False)
                milling_stage = structures.FibsemMillingStage(
                    name="Milling Stage",
                    milling=sputter_milling_settings,
                    pattern=needle_pattern,
                    alignment=milling_alignment)
                milling.setup_milling(self.fib_microscope, milling_stage)
                milling.draw_patterns(self.fib_microscope, milling_stage.pattern.define())
                milling.run_milling(self.fib_microscope, milling_stage.milling.milling_current,
                                    milling_stage.milling.milling_voltage)
                print("Milling is running.")
                milling.finish_milling(self.fib_microscope)
                pt_needle.retract()
            except Exception as e:
                print(f"The sputter process failed because of {e}.")
                pt_needle.retract()

    def setup_gis(self, time):
        #self.fib_microscope.move_stage_absolute(self.retrieve_stage_position(self.grid_number, 'gis'))
        multichem_needle = self.thermo_microscope.gas.get_multichem()
        multichem_needle.turn_heater_on('CRYO Pt ')
        try:
            if multichem_needle.state == 'Retracted':
                multichem_needle.insert()
                print('Needle retracted')
                self.thermo_microscope.microscope.patterning.set_default_beam_type(BeamType.ION)
                self.thermo_microscope.microscope.patterning.set_default_application_file("W_M e")
                self.thermo_microscope.clear_patterns()
                pattern = self.thermo_microscope.microscope.patterning.\
                    create_rectangle(center_x=0.0, center_y=0.0, width=2e-6, height=2e-6, depth=1e-6)
                pattern.application_file = "W_M e"
                pattern.gas_type = 'CRYO Pt '
                pattern.gas_flow = [80.0]
                pattern.time = time
                self.thermo_microscope.microscope.patterning.start()    # ?? DO I NEED THIS?
                multichem_needle.open()
                time.sleep(time+2)
                multichem_needle.retract()
        except Exception as e:
            print(f"The GIS layer failed because of {e}.")
            if multichem_needle.state == 'Inserted':
                multichem_needle.retract()


