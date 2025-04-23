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

light = model.getComponent(role='light')
filter_wheel = model.getComponent(role='filter')
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

