from abc import ABC, abstractmethod
import fibsem.utils as utils
from pathlib import Path
import os
import logging
from autoscript_sdb_microscope_client import SdbMicroscopeClient

class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""    
    @abstractmethod
    def connect(self, host: str, port: int):
        pass

    @abstractmethod
    def disconnect(self):
        pass


class ThermoMicroscope(FibsemMicroscope):
    """ThermoFisher Microscope class, uses FibsemMicroscope as blueprint 

    Args:
        FibsemMicroscope (ABC): abstract implementation
    """
    def __init__(self):
        self.connection = SdbMicroscopeClient()

    def disconnect(self):
        self.connection.disconnect()
        pass
    
    # @classmethod
    def connect_to_microscope(self, ip_address: str, port: int = 7520) -> None:
        """Connect to a Thermo Fisher microscope at a specified I.P. Address and Port
        
        Args:
            ip_address (str): I.P. Address of microscope 
            port (int): port of microscope (default: 7520)
            """
        try:
            # TODO: get the port
            logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
            self.connection.connect(host=ip_address, port=port)
            logging.info(f"Microscope client connected to [{ip_address}:{port}]")
        except Exception as e:
            logging.error(f"Unable to connect to the microscope: {e}")
            

            
        
        
        