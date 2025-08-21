import os
import sys
from fibsem.microscopes import odemis_microscope

odemis_path = f"{home_path}/PythonProjects/CLEM_Automation_FIBSEM/fibsem/config/odemis_czii.conf"
config = parse_config(odemis_path)





# with open(odemis_path, "r") as f:
#     for line in f:
#         if "=" in line and not line.strip().startswith("#"):
#             key, value = line.strip().split('=', 1)
#             config[key] = value.replace('"', '')
#
# print(config)
# # Extract paths
# devpath = config.get("DEVPATH", "")
# pythonpath = config.get("PYTHONPATH", "")
# python_interpreter = config.get("PYTHON_INTERPRETER", "")
#
# print(f"DEVPATH: {devpath}")
# print(f"PYTHONPATH: {pythonpath}")
# print(f"PYTHON_INTERPRETER: {python_interpreter}")
#
# sys.path.append("/Users/sophia.betzler/PythonProjects/Odemis/src")
# for path in sys.path:
#     print(path)
#
#
# sys.path.append("/Users/sophia.betzler/PythonProjects/Odemis/Pyro4/src")
#
# try:
#     import odemis
#     print("✅ Odemis module successfully imported!")
# except ModuleNotFoundError:
#     print("❌ ERROR: Could not import Odemis!")