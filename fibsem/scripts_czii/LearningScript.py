from fibsem import structures, milling, utils, acquire, microscope
import os
from datetime import datetime
from pathlib import Path

microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")

current_date = datetime.now().strftime("%Y%m%d")
desktop_path = Path.home() / "Desktop/"
folder_path = os.path.join(desktop_path, 'TestImages/', current_date)
acquisition_time = datetime.now().strftime("%H-%M")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
else:
    print('Directory already exists')
settings.image.path = folder_path

rectangle_pattern_1 = structures.FibsemRectangleSettings(
        rotation=30,
        width = 10.0e-6,
        height = 50.0e-6,
        centre_x = 0,
        centre_y = 0,
        depth = 3e-6,
)

rectangle_pattern_2 = structures.FibsemRectangleSettings(
        rotation = -30,
        width = 10.0e-6,
        height = 100.0e-6,
        centre_x = 0,
        centre_y = 0,
        cleaning_cross_section=True,
        depth = 3e-6,
)


milling_settings = milling.FibsemMillingSettings(
        milling_voltage = 30000,
        milling_current = 15e-9,
        patterning_mode = 'Serial',
)

ionbeam_settings = milling.FibsemMillingSettings(
        milling_voltage = 30000,
        milling_current = 60e-12,
)

ionbeam_imaging_settings = structures.ImageSettings(
        autocontrast = True,
        autogamma = False,
        resolution = [1536, 1024],
        dwell_time = 3e-6,
        filename = 'before',
        save = True,
        hfw = 150.0e-6,
        path = folder_path,
)


milling.draw_patterns(microscope, [rectangle_pattern_1, rectangle_pattern_2])
print(milling.estimate_milling_time(microscope, [rectangle_pattern_1, rectangle_pattern_2]))
milling.run_milling(microscope, milling_settings.milling_voltage, milling_settings.milling_current)
acquire.new_image(microscope, ionbeam_imaging_settings)
milling.finish_milling(microscope, imaging_current=ionbeam_settings.milling_current,
                   imaging_voltage=ionbeam_settings.milling_voltage)
acquire.new_image(microscope, ionbeam_imaging_settings)