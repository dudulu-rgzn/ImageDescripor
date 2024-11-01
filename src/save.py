import os
import time
from PIL import Image


class DateSaver:
    def __init__(self,config) -> None:
        self.config = config
        self.saved_image_amount = 0
        self.start_time =  time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime()) # used to get default_save_path 
        self.save_path = None # cache
        self.trigger_words = config["dataset"]["Trigger Words"]["text"]
        self.is_path_exist = False # cache
    
    def save(self,image:Image.Image,caption,path=None):
        if self.save_path:
            path = os.path.join(self.save_path,str(self.saved_image_amount)+".png")
        if path is None:
            if not hasattr(image,"filename"):
                path = os.path.join(self._get_default_save_path(),str(self.saved_image_amount)+".png")
            else:    
                path = image.filename.replace("/raw/","/output/") # type: ignore
                path = image.filename.replace("\\raw\\","\\output\\") # type: ignore
        if not (self.is_path_exist or os.path.exists(os.path.split(path)[0])):
            os.makedirs(os.path.abspath(os.path.split(path)[0]))
            self.is_path_exist = True
        self._save_image(image=image,file_path=path)
        txt_path = os.path.splitext(path)[0]+".txt"
        self._save_txt(self._add_trigger_words(caption,self.trigger_words),file_path=txt_path)
        self.saved_image_amount += 1

        
    def _get_default_save_path(self) -> str:
        count = 1
        while os.path.exists(os.path.join("dataset","output",self.start_time+"-"+str(count))):
            count += 1
        self.save_path = os.path.join("dataset","output",self.start_time+"-"+str(count))
        return self.save_path
        
        
    
    def _save_image(self,image:Image.Image,file_path:str):
        image.save(file_path)
            
    def _add_trigger_words(self,caption:str,trigger_words:str):
        return trigger_words+ caption
    
    def _save_txt(self,context:str,file_path:str):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(context)
            
    