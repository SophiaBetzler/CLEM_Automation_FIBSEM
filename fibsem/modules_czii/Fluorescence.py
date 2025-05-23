from sympy import continued_fraction

from Basic_Functions import BasicFunctions
from fibsem import structures, acquire, calibration, microscope
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import math
import napari
import glob
from skimage import io
from magicgui import magicgui

class Fluorescence:
    def __init__(self, bf, grid_number, fl_microscope=None, excitation=0.0, emission=0.0, fib_microscope=None):

        self.fl_microscope = fl_microscope
        self.excitation = excitation
        self.emission = emission
        self.fib_microscope = None
        self.bf = bf
        self.grid_number = grid_number
        self.stored_stage_positions = self.bf.read_from_yaml(filename='czii-stored-stage-positions',
                                                       imaging_settings_yaml=False)
        self.fl_stage_position = next((d for d in self.stored_stage_positions if d["name"] ==
                                                           f"grid_{self.grid_number}_fl_position"), None)

    def insert_objective(self, objective_position):

        if self.fl_microscope == 'Thermo':
            #verify stage position
            print('Setup not yet established for Thermo.')
        elif self.fl_microscope == 'Meteor':
            microscope.move_stage_absolute(microscope.FibsemStagePosition(x=self.fl_stage_position['x'],
                                           y=self.fl_stage_position['y'],
                                           z=self.fl_stage_position['z'],
                                           r=self.fl_stage_position['r'],
                                           t=self.fl_stage_position['t']))
            current_stage_position = microscope.get_stage_position()
            if self.bf.stage_position_within_limits(5, current_stage_position, self.fl_stage_position) is True:
                self.bf.socket_cumminication(target_pc='Meteor_PC',
                                             function='insert_objective',
                                             args=objective_position)
            else:
                RuntimeError("Cannot insert objective because the stage position is not within the allowed range.")


            #insert objective to stored value, if no stored value found through error and ask user to insert objective manually,
            #after manual insertion update objective position for future reference.
        else:
            ValueError('No fluorescence microscope selected.')

    def retract_objective(self):
        if self.fl_microscope == 'Thermo':
            print('Not yet done.')
        elif self.fl_microscope == 'Meteor':
            print('Not yet done.')
        else:
            ValueError('No fluorescence objective selected.')

    def acquire_fl_image(self, fl_signal=True, reflection=True):
        print('Not yet done.')
        #verify that objective is inserted
        #autofocus
        #acquire Zstack using the same exitation for both refraction and emission imaging
