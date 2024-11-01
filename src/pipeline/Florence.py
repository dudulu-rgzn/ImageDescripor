
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

from pipeline.basepipeline import BaseCropPipeline, BaseCaptionPipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class FlorencePipeline(BaseCaptionPipeline):
    def __init__(self,path:str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.cropPipeline = FlorenceCropPipeline(model=self.model,processor=self.processor)
        self.captionPipeline = FlorenceCaptionPipeline(model=self.model,processor=self.processor)
        
    def __call__(self, image : Image.Image) -> tuple[Image.Image,str]:
        crop_image = self.cropPipeline(image)
        caption = self.captionPipeline(crop_image)
        return crop_image,caption

    
class FlorenceCropPipeline(BaseCaptionPipeline):
    
    def __init__(self,model,processor,cropping_method=None):
        self.cropping_method = cropping_method if cropping_method else self.default_cropping_method
        self.model = model
        self.processor = processor
        
    def florenceCropDetection(self,image,text_input):
        """
        text_input:string of detection object 
        example : "car,face,person"
        """
        
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>" # detection by text prompt
        prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer
        
    def default_cropping_method(self,image : Image.Image,florence_answer):
        max_length = max(image.height,image.width)
        center = [[(i[0]+i[2])/2,(i[1]+i[3])/2] for i in florence_answer['<CAPTION_TO_PHRASE_GROUNDING>']["bboxes"]] # first get the box center
        center = [sum(i)/len(i) for i in zip(*center)] 
        """s
        Coord
      (0,0)---------> Don't need add one
        |(.5,.5) |  LinK for pil crood: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
        |(.5,1.5)|
        |...
        ↓
        """# get mean for every box
        if image.height>image.width:
            upper = max(min(center[1]-image.width/2,image.height-image.width),0) # limit 0 to (height - width) - 1
            lower = min(max(center[1]+image.width/2,image.height-image.width),image.height) # width - 1 to height -1
            cropped_image = image.crop((0, round(upper), image.width, round(lower)))
        else: # image.height<=image.width
            left = max(min(center[0]-image.height/2,image.width-image.height),0)
            right = min(max(center[0]+image.height/2,image.width-image.height),image.width)
            cropped_image = image.crop((round(left), 0, round(right), image.height)) # (left, upper, right, lower)
        resize_image = cropped_image.resize((1024,1024))
   
        return resize_image
        
    def __call__(self, image : Image.Image) -> Image.Image:
        image = self.cropping_method(image=image,florence_answer=self.florenceCropDetection(image=image,text_input="person,face"))
        return image
    
class FlorenceCaptionPipeline(BaseCaptionPipeline):
    def __init__(self,model,processor):
        self.model = model
        self.processor = processor
        
    def florenceCaptionGenerate(self,image) -> str:
        """
        text_input:string of detection object 
        example : "car,face,person"
        """
        task_prompt = "<MORE_DETAILED_CAPTION>" # detection by text prompt
        prompt = task_prompt
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        return parsed_answer['<MORE_DETAILED_CAPTION>']
    
    def __call__(self, image : Image.Image) -> str:
        return self.florenceCaptionGenerate(image=image)