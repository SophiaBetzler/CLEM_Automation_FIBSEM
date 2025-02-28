import yaml
from fibsem import milling, structures
from Basic_Functions import BasicFunctions
from Imaging import Imaging


class MillingSetup():
    def __init__(self):
        bf = BasicFunctions()
        self.dict_milling_parameters = bf.read_from_yaml(filename=f"milling_stages_lamella")
        print(self.dict_milling_parameters)


    def create_stage_setup(self):
        fiducial = milling.patterning.patterns2.RectanglePattern.from_dict(self.dict_milling_parameters['patterns']['fiducial'])
        trench = milling.patterning.patterns2.RectanglePattern.from_dict(self.dict_milling_parameters['patterns']['rectangle'])
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