from fibsem.microscopes import odemis_microscope
odemis_microscope.add_odemis_path()
from odemis import model, acq
from odemis.model import getComponent
import inspect

from odemis.acq.stream import SEMCCDMDStream
from odemis.util.dataio import open_acquisition
from odemis.model._dataflow import DataFlow, DataArray
import odemis.acq.stream as stream
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

test_image = DataArray(np.random.randint(0, 255,size=(512, 512), dtype=np.uint8),
                       metadata={"simulated": True})
df = DataFlow()

streams = []
for name in dir(stream):
    obj = getattr(stream, name)
    if inspect.isclass(obj) and issubclass(obj, stream.Stream):
        streams.append(obj)
camera = model.getComponent(role='ccd')
camera.binning.value = (2, 2)
light = model.getComponent(role='light')
dict_excitation = {
    '375': 0,
    '460': 1,
    '454': 2,
    '615': 3
} # position in the list for the power intensity
dict_emission = {
    'pass-through': 0.08,
    '500': 0.865498,
    '580': 1.650796,
    '663': 2.4361944
} # wheel position in radians
light.power.value[dict_excitation['375']] = 1.0
light.power.value = [0.0, 0.0, 0.0, 0.0] # limit is [1.4, 0.78, 1.45, 0.85]

print(light.power.value) # in W
filter_wheel = model.getComponent(role='filter')

print(filter_wheel.position.value)
#obs = acq.acqmng.SettingsObserver(microscope=root, components=all_components)
fl_stream = stream.FluoStream(name='FL',
                              detector=camera,
                              emitter=light,
                              em_filter=filter_wheel,
                              dataflow=camera.data
                              )



future = acq.acqmng.acquire([fl_stream])
print(future)
data, err = future.result()
plt.imshow(data[0])
plt.show()
if err:
    print("Error during acquisition:", err)
else:
    print("Image acquired. Data shape:", data[0].shape)

#stream = SEMCCDMDStream("CCD ZStack", getComponent(role="ccd"))

