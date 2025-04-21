import tifffile
import inspect
from fibsem.microscopes import odemis_microscope
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class OdemisControl:

    def __init__(self):
        odemis_microscope.add_odemis_path()
        from odemis import model, acq
        from odemis.acq.stream import FIBStream, SEMStream
        from odemis.util.dataio import open_acquisition
        import odemis.acq.stream as stream
        self.stream = stream
        self.acq = acq
        self.model = model
        self.FIBStream = FIBStream
        self.SEMStream = SEMStream
        self.open_acquisition = open_acquisition
        self.focuser = self.model.getComponent(role='focus')
        self.light = self.model.getComponent(role='light')
        self.lens = self.model.getComponent(role='lens')
        self.filter_wheel = self.model.getComponent(role='filter')
        self.camera = self.model.getComponent(role='ccd')

    def insert_objective(self):
        print('I made to here.')
        print(self.focuser.position.value)
        print(self.focuser.__dict__)
        for comp in self.model.getComponents():
            print(f"{comp.name} (role: {comp.role}).")
        print(self.focuser.position.value)

    def acquire_fl_image(self):
        self.camera.exposureTime.value = 0.1
        self.camera.binning.value = (4, 4)
        fl_stream = self.stream.FluoStream(name='FL',
                                      detector=self.camera,
                                      emitter=self.light,
                                      em_filter=self.filter_wheel,
                                      dataflow=self.camera.data
                                      )
        future = self.acq.acqmng.acquire([fl_stream])
        data, err = future.result()
        plt.imshow(data[0])
        plt.show()
        meta_data = {'fl_binning': self.camera.binning.value,
                     'fl_exposure_time': self.camera.exposureTime.value}
        print(meta_data)
        if err:
            print("Error during acquisition:", err)
        else:
            print("Image acquired. Data shape:", data[0].shape)

    def acquire_fl_tileset(self):
        from self.acq.stitching import WEAVER_MEAN, REGISTER_IDENTITY
        self.camera.exposureTime.value = 0.1
        self.camera.binning.value = (4, 4)
        fl_stream = self.stream.FluoStream(name='FL',
                                           detector=self.camera,
                                           emitter=self.light,
                                           em_filter=self.filter_wheel,
                                           dataflow=self.camera.data
                                           )
        zleverls = [0.0, 0.2e-6, 0.4e-6, 0.6e-6]
        future = self.acq.stitching.acquireOverview(streams=[fl_stream],
                                                    stage=stage_component,
                                                    areas=[(x0, y0, x1, y1)], #tile area in meters
                                                    focus=z_actuator,
                                                    detector=detector_component,
                                                    overlap=0.1,
                                                    settings_obs=settings_observer,
                                                    log_path='/temp/zstack_test.json',
                                                    weaver=WEAVER_MEAN,
                                                    registrat=REGISTER_IDENTITY,
                                                    zlevels=zlevels,
                                                    focusing_metod=FocusingMethod.MAX_INTENSITY_PROJECTION,
                                                    use_autfocus=False,
                                                    focus_points_dist=50-6)

    def acquire_fl_stack(self):
        z_positions = [0.0, 0.2e-6, 0.4e-6, 0.6e-6]
        self.camera.exposureTime.value = 0.1
        self.camera.binning.value = (4, 4)
        fl_stream = self.stream.FluoStream(name='FL',
                                           detector=self.camera,
                                           emitter=self.light,
                                           em_filter=self.filter_wheel,
                                           dataflow=self.camera.data
                                           )
        future = self.acq.acqmng.acquireZStack([fl_stream],
                                               zlevels=z_positions)

        data, err = future.result()
        plt.imshow(data[0])
        plt.show()
        if err:
            print("Error during acquisition:", err)
        else:
            print("Image acquired. Data shape:", data[0].shape)


    def load_tif_as_array(self, path):
        image_tif = tifffile.TiffFile(path)
        array = image_tif.asarray()
        print(image_tif.pages[0].tags)
        metadata = image_tif.pages[0].tags
        image_dict = {tag.name: tag.value for tag in metadata.values()}
        return {"image": array,
                "metadata": image_dict}