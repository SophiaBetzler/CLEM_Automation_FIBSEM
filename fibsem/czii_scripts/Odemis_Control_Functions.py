import tifffile
from fibsem.microscopes import odemis_microscope

class OdemisControl:

    def __init__(self):
        self.conn = self.connect_to_odemis()

    def connect_to_odemis(self):
        odemis_microscope.add_odemis_path()
        from odemis import model
        from odemis.acy.stream import FIBStream, SEMStream
        from odemis.util.dataio import open_acquisition

    def insert_objective(self):
        Focuser = model.getComponent(role='filter')
        print(Focuser.__dict__)


    def load_tif_as_array(path):
        image_tif = tifffile.TiffFile(path)
        array = image_tif.asarray()
        print(image_tif.pages[0].tags)
        metadata = image_tif.pages[0].tags
        image_dict = {tag.name: tag.value for tag in metadata.values()}
        return {"image": array,
                "metadata": image_dict}