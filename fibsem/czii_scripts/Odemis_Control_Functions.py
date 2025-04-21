import tifffile
from fibsem.microscopes import odemis_microscope

class OdemisControl:

    def __init__(self):
        odemis_microscope.add_odemis_path()
        from odemis import model, acq
        from odemis.acq.stream import FIBStream, SEMStream
        from odemis.util.dataio import open_acquisition
        self.model = model
        self.acq = acq
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

    def acquire_FL_image(self):

        stream = self.acq.stream.OpticalStream()

        stream.detector.value = self.camera  # Link the hardware
        stream.tint.value = 0.1  # Integration time in seconds
        stream.excitation.value = "DAPI"  # Or appropriate enum/string
        stream.emission.value = "blue"

        root = self.model.getRootComponent()
        all_components = set(self.model.getHwComponents(root))
        obs = self.acq.acqmng.SettingsObserver(microscope=root, components=all_components)

        future = self.acq.acqmng.acquire([stream], settings_obs=obs)

        data, err = future.result()
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