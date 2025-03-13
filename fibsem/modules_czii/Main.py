from Basic_Functions import BasicFunctions
from Imaging import Imaging
from GIS_Sputter_Setup import GisSputterAutomation
from Fiducial_Identification import FiducialID
from Milling import MillingSetup
from EucentricHeight import EucentricHeight

bf = BasicFunctions(manufacturer='Thermo',
                ip='192.168.0.1',
                tool='Hydra',
                pc_type='windows')

# bf = BasicFunctions(
#                 pc_type='mac',
#                 manufacturer='Demo',
#                  ip='localhost',
#                  tool='Hydra')

fib_microscope, fib_settings = bf.connect_to_microscope()
#imaging = Imaging(fib_microscope=fib_microscope, beam='electron', bf=bf)

gis = GisSputterAutomation(fib_microscope=fib_microscope, grid_number=1)
gis.setup_sputtering()
#gis.setup_gis(time=1)
#eucentric = EucentricHeight(fib_microscope=fib_microscope, bf=bf)

#eucentric.eucentric_height_tilt_series(z_shifts=[-1.0e-5, -0.5e-5, 0.0e-5, +0.5e-5],
#                                        tilts=[-35.0, -25.0, -20.0, -15.0, -10.0, -5.0,
#                                                0.0, +5.0, +10.0])


#_ = imaging.acquire_image()
#imaging.fast_acquire(200, line_acquisition=True)
# parameters_dict = {
#                      'hfw': [300.0e-6, 400.0e-6, 500.0e-6, 600.0e-6],
#                      'current': [1e-12, 1e-12, 1e-11]
# }
# imaging.acquire_multiple(parameters_dict)
# fiducials = FiducialID(bf = bf,
#                        imaging = imaging,
#                       number_fiducials = 2,
#                       fib_settings = fib_settings)
# fiducials.fiducial_identification()


#ms = MillingSetup()
#ms.create_stage_setup()