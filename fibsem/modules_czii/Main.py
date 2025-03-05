from Basic_Functions import BasicFunctions
from Imaging import Imaging
from Fiducial_Identification import FiducialID
from Milling import MillingSetup

bf = BasicFunctions(manufacturer='Demo',
                ip='localhost',
                tool='Hydra')

fib_microscope, fib_settings = bf.connect_to_microscope()

imaging = Imaging(fib_microscope=fib_microscope, beam='electron')
#imaging.acquire_image()
imaging.fast_acquire(10)
# parameters_dict = {
#                      'hfw': [300.0e-6, 400.0e-6, 500.0e-6, 600.0e-6],
#                      'current': [1e-12, 1e-12, 1e-11]
# }
# imaging.acquire_multiple(parameters_dict)
# fiducials = FiducialID(number_fiducials=2,
#                        pc_type='mac',
#                         fib_microscope=fib_microscope,
#                         beam='ion',
#                        fib_settings=fib_settings)
# fiducials.fiducial_identification()


#ms = MillingSetup()
#ms.create_stage_setup()