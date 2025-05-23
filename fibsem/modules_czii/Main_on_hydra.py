from Basic_Functions import BasicFunctions
from Imaging import Imaging
from GIS_Sputter_Setup import GisSputterAutomation
from Fiducial_Identification import FiducialID
from Milling import MillingSetup
import numpy as np
from EucentricHeight import EucentricHeight

# bf = BasicFunctions(manufacturer='Thermo',
#                 ip='192.168.0.1',
#                 tool='Hydra',
#                 pc_type='windows')

fib_microscope, fib_settings = bf.connect_to_microscope()

fib_microscope.connect

# bf = BasicFunctions(
#                 pc_type='mac',
#                 manufacturer='Demo',
#                  ip='localhost',
#                  tool='Hydra')



#imaging = Imaging(fib_microscope=fib_microscope, bf=bf)
#imaging.acquire_image(save=False, beam_type='electron', autofocus=True, fib_settings=fib_settings)
#imaging.acquire_tileset(method='overview')



#fiducial_id = FiducialID(bf=bf, imaging=imaging, fib_settings=fib_settings)
#milling = MillingSetup(bf=bf, fib_microscope=fib_microscope, fiducial_id=fiducial_id)


#gis = #GisSputterAutomation(fib_microscope=fib_microscope, bf=bf, grid_number=1)
#gis.#setup_sputtering(1)
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