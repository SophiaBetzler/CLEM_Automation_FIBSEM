from Basic_Functions import BasicFunctions
from Imaging import Imaging
from Fiducial_Identification import FiducialID
from Milling import MillingSetup

bf = BasicFunctions(manufacturer='Demo',
                ip='localhost',
                tool='Hydra')

fib_microscope, fib_settings = bf.connect_to_microscope()

# imaging = Imaging(fib_microscope=fib_microscope, beam='ion')
# imaging.acquire_image()
# parameters_dict = {
#                     'hfw': [300, 400, 500, 600],
#                     'current': [1e-9, 1e-10, 1e-11]
# }
fiducials = FiducialID(number_fiducials=2,
                       pc_type='mac',
                        fib_microscope=fib_microscope,
                        beam='ion',
                       fib_settings=fib_settings)
fiducials.fiducial_identification()


ms = MillingSetup()
ms.create_stage_setup()