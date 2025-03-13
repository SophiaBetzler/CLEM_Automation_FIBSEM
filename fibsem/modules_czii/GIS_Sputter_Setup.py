from fibsem import acquire, utils, microscope, structures, milling, calibration
from fibsem.structures import BeamType, FibsemStagePosition, FibsemDetectorSettings
import yaml
import os
from pathlib import Path
import numpy as np
import time

class GisSputterAutomation:

    def __init__(self, fib_microscope, grid_number):
        self.fib_microscope = fib_microscope
        self.grid_number = grid_number
        self.project_root = Path(__file__).resolve().parent.parent
        with open(os.path.join(self.project_root, 'config', 'czii-stored-stage-positions.yaml'), "r") as file:
            self.saved_stage_positions = yaml.safe_load(file)


    def retrieve_stage_position(self, position_name):
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

    def setup_sputtering(self):
        #self.fib_microscope.move_stage_absolute(self.retrieve_stage_position(self.grid_number, 'sputter'))
        #self.fib_microscope.connection.gas.list_all_gis_ports()
        pt_needle = self.fib_microscope.connection.gas.get_gis_port('µSputter')
        try:
            pt_needle.insert()
            time.sleep(5)
            pt_needle.retract()
        except Exception as e:
            print(f"The sputter process failed because of {e}.")
            pt_needle.retract()

    def setup_gis(self, time):
        #self.fib_microscope.move_stage_absolute(self.retrieve_stage_position(self.grid_number, 'gis'))
        multichem_needle = self.fib_microscope.connection.gas.get_multichem()
        print(dir(multichem_needle))
        #print(multichem_needle.list_all_gases())
        #print(multichem_needle.turn_heater_on('CRYO Pt'))
        print(multichem_needle.get_temperature('CRYO Pt '))
        try:
            if multichem_needle.state == 'Retracted':
                #multichem_needle.insert()
                print('Needle retracted')
            #multichem_needle.turn_heater_on('CRYO Pt ')
            #multichem_needle.open()
            #multichem_needle.open()
            #time.sleep(time)
            #multichem_needle.retract()
        except Exception as e:
            print(f"The GIS layer failed because of {e}.")
            if multichem_needle.state == 'Inserted':
                multichem_needle.retract()


