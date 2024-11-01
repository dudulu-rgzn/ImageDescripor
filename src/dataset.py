import os
import glob
from typing import Iterator

from PIL import Image

class Dataset:
    def __init__(self,config) -> None:
        load_funtion_map = {
            "singlefolder":self._load_from_single_folder
        }
        self.images_path = []
        self.type = config["type"]
        load_funtion = load_funtion_map[self.type]
        load_funtion(config)
        
    def _load_from_single_folder(self, config:dict):
        path = glob.escape(config["path"]) # we need "escape" to escape ('?', '*' and '[')
        image_extensions = [".bmp", ".gif", ".png", ".tiff", ".tif", ".webp", ".jpeg", ".jpg"]
        glob_path = os.path.join(path,"**")
        images_path = glob.iglob(glob_path,recursive=True)
        for i in images_path:
            if "."+i.split(".")[-1] in image_extensions:
                self.images_path.append(i)
    
    def _read_image(self, path) -> Image.Image:
        return Image.open(path)
    
    def __len__(self):
        return len(self.images_path)
    
    def get_images(self) -> Iterator:
        for image_path in self.images_path:
            yield image_path ,self._read_image(image_path)
            
    def output(self,image,path):
        return 