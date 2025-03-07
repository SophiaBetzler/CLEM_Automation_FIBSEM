import fibsem
from fibsem import microscope, utils
import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent

try:
    # for hydra microscope use:
    config_path = os.path.join(project_root, 'config', 'czii-tfs-hydra-configuration.yaml')
    # for arctis microscope use:
    # config_path = os.path.join(self.project_root, 'config', 'tfs-arctis-configuration.yaml')
    # self.microscope, self.settings = utils.setup_session(manufacturer='Thermo', ip_address='192.168.0.1',
    #                                                      config_path=config_path)
    #
    fib_microscope, fib_settings = utils.setup_session(manufacturer='Demo', ip_address='localhost',
                                                         config_path=config_path)
except Exception as e:
    print(f"Connection to microscope failed: {e}")

Thermo = microscope.ThermoMicroscope()
print(Thermo)
current = Thermo.connection.beams.electron_beam.beam_current.value
print(f"The current of the electron beam is {current}.")