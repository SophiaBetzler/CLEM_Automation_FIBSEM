import datetime
import glob
import json
import logging
import os
import time
from pathlib import Path

import yaml
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import AdornedImage
from PIL import Image
import fibsem
from fibsem.structures import (
    BeamType,
    MicroscopeSettings,
    ImageSettings,
    StageSettings,
    SystemSettings,
    DefaultSettings,
)


def connect_to_microscope(ip_address="10.0.0.1"):
    """Connect to the FIBSEM microscope."""
    try:
        # TODO: get the port
        logging.info(f"Microscope client connecting to [{ip_address}]")
        microscope = SdbMicroscopeClient()
        microscope.connect(ip_address)
        logging.info(f"Microscope client connected to [{ip_address}]")
    except Exception as e:
        logging.error(f"Unable to connect to the microscope: {e}")
        microscope = None

    return microscope

def sputter_platinum_v2(
    microscope: SdbMicroscopeClient,
    protocol: dict,
    whole_grid: bool = False,
    default_application_file: str = "autolamella",
):
    """Sputter platinum over the sample.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        settings (dict): platinum protcol dictionary
        whole_grid (bool, optional): sputtering protocol. Defaults to False.

    Raises:
        RuntimeError: Error Sputtering
    """

    # protcol = settings["protocol"]["platinum"] in old system
    # protocol = settings.protocol["platinum"] in new
    if whole_grid:

        sputter_time = protocol["whole_grid"]["time"]  # 20
        hfw = protocol["whole_grid"]["hfw"]  # 30e-6
        line_pattern_length = protocol["platinum"]["whole_grid"]["length"]  # 7e-6
        logging.info("sputtering platinum over the whole grid.")
    else:
        sputter_time = protocol["weld"]["time"]  # 20
        hfw = protocol["weld"]["hfw"]  # 100e-6
        line_pattern_length = protocol["weld"]["length"]  # 15e-6
        logging.info("sputtering platinum to weld.")

    # Setup
    original_active_view = microscope.imaging.get_active_view()
    microscope.imaging.set_active_view(BeamType.ELECTRON.value)
    microscope.patterning.clear_patterns()
    microscope.patterning.set_default_application_file(protocol["application_file"])
    microscope.patterning.set_default_beam_type(BeamType.ELECTRON.value)
    multichem = microscope.gas.get_multichem()
    multichem.insert(protocol["position"])
    multichem.turn_heater_on(protocol["gas"])  # "Pt cryo")
    time.sleep(3)

    # Create sputtering pattern
    microscope.beams.electron_beam.horizontal_field_width.value = hfw
    pattern = microscope.patterning.create_line(
        -line_pattern_length / 2,  # x_start
        +line_pattern_length,  # y_start
        +line_pattern_length / 2,  # x_end
        +line_pattern_length,  # y_end
        2e-6,
    )  # milling depth
    pattern.time = sputter_time + 0.1

    # Run sputtering
    microscope.beams.electron_beam.blank()
    # TODO: synchronous sputtering version, fix the safety on this...
    if microscope.patterning.state == "Idle":
        logging.info("Sputtering with platinum for {} seconds...".format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
        time.sleep(sputter_time + 5)
    else:
        raise RuntimeError("Can't sputter platinum, patterning state is not ready.")
    if microscope.patterning.state == "Running":
        microscope.patterning.stop()
    else:
        logging.warning("Patterning state is {}".format(microscope.patterning.state))
        logging.warning("Consider adjusting the patterning line depth.")

    # Cleanup
    microscope.patterning.clear_patterns()
    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(BeamType.ION.value)  # set ion beam
    multichem.retract()
    logging.info("sputtering platinum finished.")


def save_image(image, save_path, label=""):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{label}.tif")
    image.save(path)


def current_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d.%I-%M-%S%p")


def _format_time_seconds(seconds: float) -> str:
    """Format a time delta in seconds to proper string format."""
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


# TODO: better logs: https://www.toptal.com/python/in-depth-python-logging
def configure_logging(path: Path = "", log_filename="logfile", log_level=logging.INFO):
    """Log to the terminal and to file simultaneously."""
    logfile = os.path.join(path, f"{log_filename}.log")

    logging.basicConfig(
        format="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(),
        ],
    )

    return logfile


def load_yaml(fname: Path) -> dict:

    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_metadata(settings: dict, path: Path):
    fname = os.path.join(path, "metadata.json")
    with open(fname, "w") as fp:
        json.dump(settings, fp, sort_keys=True, indent=4)


def create_gif(path: Path, search: str, gif_fname: str, loop: int = 0) -> None:
    filenames = glob.glob(os.path.join(path, search))

    imgs = [Image.fromarray(AdornedImage.load(fname).data) for fname in filenames]

    print(f"{len(filenames)} images added to gif.")
    imgs[0].save(
        os.path.join(path, f"{gif_fname}.gif"),
        save_all=True,
        append_images=imgs[1:],
        loop=loop,
    )


def setup_session(
    config_path: Path = None, protocol_path: Path = None
) -> tuple[SdbMicroscopeClient, MicroscopeSettings]:
    """Setup microscope session

    Args:
        config_path (Path): path to config directory
        protocol_path (Path): path to protocol file

    Returns:
        tuple: microscope, settings, image_settings
    """

    # load settings
    settings = load_settings_from_config(config_path, protocol_path)

    # create session directories
    session = f'{settings.protocol["name"]}_{current_timestamp()}'
    session_path = os.path.join(os.path.dirname(protocol_path), session)
    os.makedirs(session_path, exist_ok=True)

    # configure logging
    configure_logging(session_path)

    # connect to microscope
    microscope = connect_to_microscope(ip_address=settings.system.ip_address)

    # image_setttings
    settings.image_settings.save_path = session_path

    logging.info(f"Finished setup for session: {session}")

    return microscope, settings


def load_settings_from_config(
    config_path: Path = None, protocol_path: Path = None
) -> MicroscopeSettings:
    """Load microscope settings from configuration files

    Args:
        config_path (Path, optional): path to config directory. Defaults to None.
        protocol_path (Path, optional): path to protocol file. Defaults to None.

    Returns:
        MicroscopeSettings: microscope settings
    """

    if config_path is None:
        config_path = os.path.join(os.path.dirname(fibsem.__file__), "config")

    # system settings
    settings = load_yaml(os.path.join(config_path, "system.yaml"))
    system_settings = SystemSettings.__from_dict__(settings)

    # user settings
    config = load_yaml(os.path.join(config_path, "config.yaml"))
    default_settings = DefaultSettings.__from_dict__(config)
    image_settings = ImageSettings.__from_dict__(config)

    # protocol settings
    protocol = load_protocol(protocol_path)

    settings = MicroscopeSettings(
        system=system_settings,
        default=default_settings,
        image_settings=image_settings,
        protocol=protocol,
    )

    return settings


def load_protocol(protocol_path: Path = None) -> dict:
    """Load the protocol file from yaml

    Args:
        protocol_path (Path, optional): path to protocol file. Defaults to None.

    Returns:
        dict: protocol dictionary
    """
    if protocol_path is not None:
        protocol = load_yaml(protocol_path)
    else:
        protocol = {"name": "demo"}

    protocol = _format_dictionary(protocol)

    return protocol


def _format_dictionary(dictionary: dict) -> dict:
    """Recursively traverse dictionary and covert all numeric values to flaot.

    Parameters
    ----------
    dictionary : dict
        Any arbitrarily structured python dictionary.

    Returns
    -------
    dictionary
        The input dictionary, with all numeric values converted to float type.
    """
    for key, item in dictionary.items():
        if isinstance(item, dict):
            _format_dictionary(item)
        elif isinstance(item, list):
            dictionary[key] = [
                _format_dictionary(i)
                for i in item
                if isinstance(i, list) or isinstance(i, dict)
            ]
        else:
            if item is not None:  # TODO: change to isinstance(int/float)
                try:
                    dictionary[key] = float(dictionary[key])
                except ValueError:
                    pass
    return dictionary
