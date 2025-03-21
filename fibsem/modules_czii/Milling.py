import yaml
from fibsem import milling, structures
from fibsem.structures import Point
from Imaging import Imaging
from fibsem.milling.patterning.patterns2 import RectanglePattern, FiducialPattern
from fibsem.milling.patterning.plotting import draw_milling_patterns

class MillingSetup():
    def __init__(self, bf, fib_microscope, fiducial_id):
        self.bf = bf
        self.fiducial_id = fiducial_id
        self.fib_microscope = fib_microscope
        self.dict_milling_parameters = bf.read_from_yaml(filename=f"milling_stages_lamella")

    def read_in_pattern(self, pattern_type):
        """
        Reads in the pattern dimension pre-set in the milling_pattern.yaml file.
        pattern_type: fiducial, trench, rough_milling, polishing
        """
        all_patterns = self.bf.read_from_yaml('milling_pattern', imaging_settings_yaml=False)
        for i, d in enumerate(all_patterns):
            for key in list(d.keys()):
                return all_patterns[i][key]

    def perform_milling(self, pattern_type, fiducial_pattern_dimensions):
        """
        Function that performs the milling, based on the position of fiducials in the sample.
        It will always verify that the fiducials were identified based on the amount the stage has moved.
        If there was no stage movement, the function will return an error.
        """
        parameters = self.read_in_pattern(pattern_type=pattern_type)

        milling_settings = structures.FibsemMillingSettings(milling_current=parameters['milling_current'],
                                                            milling_voltage=parameters['milling_voltage'],
                                                            hfw=parameters['hfw'],
                                                            application_file=parameters['application_file'],
                                                            patterning_mode=parameters['patterning_mode'])

        milling_alignment = milling.MillingAlignment(enabled=False)

        milling_stage = milling.FibsemMillingStage(
            name="Milling Stage",
            milling=milling_settings,
            pattern=fiducial_pattern_dimensions,
            alignment=milling_alignment)

        milling.draw_patterns(self.fib_microscope, milling_stage.pattern.define())
        milling.run_milling(self.fib_microscope, milling_stage.milling.milling_current,
                            milling_stage.milling.milling_voltage)

    def create_fiducial(self):
        """
        Creates a fiducial based on the settings stored in the milling_pattern.txt file
        """
        parameters = self.read_in_pattern(pattern_type='fiducial')

        fiducial_pattern = FiducialPattern(width=parameters['width'], height=parameters['height'],
                                           depth=parameters['depth'], rotation=parameters['rotation'],
                                           point=Point(parameters['centerX'], parameters['centerY']))

        self.perform_milling(pattern_type='fiducial', fiducial_pattern_dimensions=fiducial_pattern)

    def create_trench(self):
        """
        Creates a trench in the middle of two fiducials, which should exist before starting.
        """
        current_stage_position = self.fib_microscope.get_stage_position()
        self.fiducial_id.fiducial_identification(number_fiducials=2)
        new_stage_position = self.fib_microscope.get_stage_position()
        if self.bf.stage_position_within_limits(0.5, current_stage_position, new_stage_position) is False:
            parameters = self.read_in_pattern(pattern_type='fiducial')
            

##### I WAS HERE I WAS HERE I WAS HERE ##########
            # rectangle_pattern = RectanglePattern(width=parameters['width'], height=parameters['height'],
            #                                    depth=parameters['depth'], rotation=parameters['rotation'],
            #                                    point=Point(parameters['centerX'], parameters['centerY']))
        else:
            raise ValueError("It looks like the fiducials were not identified correctly because the stage was not moved."
                             "Interrupting the milling process.")

    def create_stage_setup(self):
        fiducial = (milling.patterning.patterns2.RectanglePattern.
                    from_dict(self.dict_milling_parameters['patterns']['fiducial']))
        trench = (milling.patterning.patterns2.RectanglePattern.
                  from_dict(self.dict_milling_parameters['patterns']['rectangle']))
        print(trench)
    # milling_settings = structures.FibsemMillingSettings(
    #     milling_current=15e-9,
    #     milling_voltage=12e3,
    #     hfw=80e-6,
    #     application_file="Si",
    #     patterning_mode="Serial",
    # )
    #
    # milling_alignment = milling.MillingAlignment(
    #     enabled=False,
    # )
    #
    # milling_stage = milling.FibsemMillingStage(
    #     name="Milling Stage",
    #     milling=milling_settings,
    #     pattern=rectangle_pattern,
    #     alignment=milling_alignment,
    # )
    #
    # milling.draw_patterns(self.microscope, milling_stage.pattern.define())
    #
    # # 3. run milling
    # milling.run_milling(self.microscope, milling_stage.milling.milling_current, milling_stage.milling.milling_voltage)
    #
    # # 4. finish milling (restore imaging beam settings, clear shapes, ...)
    # milling.finish_milling(self.microscope)