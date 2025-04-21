import tifffile
from fibsem.microscopes import odemis_microscope

class OdemisControl:

    def __init__(self):
        odemis_microscope.add_odemis_path()
        from odemis import model
        from odemis.acq.stream import FIBStream, SEMStream
        from odemis.util.dataio import open_acquisition

        self.model = model
        self.FIBStream = FIBStream
        self.SEMStream = SEMStream
        self.open_acquisition = open_acquisition

    def insert_objective(self):
        print('I made to here.')
        Focuser = self.model.getComponent(role='focus')
        print(Focuser.position.value)
        Focuser.position.value[-0.010]
        print(Focuser.position.value)

    def load_tif_as_array(self, path):
        image_tif = tifffile.TiffFile(path)
        array = image_tif.asarray()
        print(image_tif.pages[0].tags)
        metadata = image_tif.pages[0].tags
        image_dict = {tag.name: tag.value for tag in metadata.values()}
        return {"image": array,
                "metadata": image_dict}