from Basic_Functions import BasicFunctions
from Fluorescence import Fluorescence
from Imaging import Imaging
from GIS_Sputter_Setup import GisSputterAutomation
from Fiducial_Identification import FiducialID
from Automatic_CLEM import *
from TriCoincidence import AutomatedTriCoincidence
from Milling import MillingSetup
import numpy as np
from EucentricHeight import EucentricHeight


# bf = BasicFunctions(manufacturer='Thermo',
#                 ip='192.168.0.1',
#                 tool='Hydra',
#                 pc_type='windows')

### I removed the path, important for the CLEM pipeline

#### I NEED TO FIND AN EFFICIENT WAY TO SET THE GRID NUMBER. I CAN SET IT WHEN I CALL THE GIS/SPUTTER AUTOMATION BUT THERE
### SHOULD ALSO BE ANOTHER WAY. MAYBE WHEN IT IS NONE I SHOULD CREATE A FUNCTION WHICH LETs THE USER SELECT?
# I Need it for the fluorescence and the GIS. Do I need it for anything else?
# You can sent the grid number to the OverArch function and then it is accessible for all other functions. How and when
# Should this be done?
# The function to do this is self.overarch.set_variable("name", value)
# AutomatedTriCoincidence()

gis = GisSputterAutomation()

#fl = Fluorescence(fl_microscope='Meteor', grid_number=1, bf=bf, fib_microscope=fib_microscope)

#fl.insert_objective()
#fl_stack, fl_scale = bf.import_images(bf.folder_path + 'FL_Z_stack.tiff')
#fib_image, fib_scale = bf.import_images(bf.folder_path + 'FIB_image.tiff')






#clem = AutomaticCLEM(bf=bf, target_position=[22, 632, 458], lamella_top_y=412.0)
#clem.run_full_3dct_pipeline()

#imaging = Imaging(fib_microscope=fib_microscope, bf=bf)


#user_selected_rois = imaging.acquire_tileset(method='overview')



#fiducial_id = FiducialID(bf=bf, imaging=imaging, fib_settings=fib_settings)
#milling = MillingSetup(bf=bf, fib_microscope=fib_microscope, fiducial_id=fiducial_id)


#gis = GisSputterAutomation(fib_microscope=fib_microscope, bf=bf, grid_number=1)
#gis.setup_sputtering(60)
#gis.setup_gis(wait_time=1)
#eucentric = EucentricHeight(fib_microscope=fib_microscope, bf=bf)

#eucentric.eucentric_height_tilt_series(z_shifts=[-500, -250, 0.0, +250, +500],
#                                        tilts=[-35.0, -25.0, -20.0, -15.0, -10.0, -5.0,
#                                                0.0, +5.0, +10.0])

# measurements = np.array([[1.0, 3.0, 2.0], [4.0, 3.0, 2.0], [10.0, 4.0, 3.0], [5.0, 3.0, 1.0]])
# eucentric = EucentricHeight(bf=bf, fib_microscope=fib_microscope)
# eucentric.determine_eucentric_plane(measurements)
#_ = imaging.acquire_image()
#imaging.fast_acquire(200, line_acquisition=True)
# parameters_dict = {
#                      'stage_bias': [300.0, 400.0, 500.0, 800.0],
#                      'current': [1.5, 1.0, 3.0]
# }
# imaging.acquire_multiple(parameters_dict, beam_type='electron')
#imaging.acquire_tileset()
# fiducials = FiducialID(bf = bf,
#                        imaging = imaging,
#                       number_fiducials = 2,
#                       fib_settings = fib_settings)
# fiducials.fiducial_identification()


#ms = MillingSetup()
#ms.create_stage_setup()