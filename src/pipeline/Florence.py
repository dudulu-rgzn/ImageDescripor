
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

from pipeline.basepipeline import BaseCropPipeline, BaseCaptionPipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class FlorencePipeline(BaseCaptionPipeline):
    def __init__(self,path:str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True,).to(device)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.cropPipeline = FlorenceCropPipeline(model=self.model,processor=self.processor)
        self.captionPipeline = FlorenceCaptionPipeline(model=self.model,processor=self.processor)
        
    def __call__(self, image : Image.Image) -> tuple[Image.Image,str]:
        crop_image = self.cropPipeline(image)
        caption = self.captionPipeline(crop_image)
        return crop_image,caption
    
    
class FlorenceCropPipeline(BaseCaptionPipeline):
    
    def __init__(self,model,processor):
        self.model = model
        self.processor = processor
        
    def florenceCropDetection(self,image,text_input):
        """
        text_input:string of detection object 
        example : "car,face,person"
        """
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>" # detection by text prompt
        prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
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
        
        
    def __call__(self, image : Image.Image) -> Image.Image:
        return image
    
class FlorenceCaptionPipeline(BaseCaptionPipeline):
    def __init__(self,model,processor):
        self.model = model
        self.processor = processor
        
    def florenceCaptionGenerate(self,image):
        """
        text_input:string of detection object 
        example : "car,face,person"
        """
        task_prompt = "<MORE_DETAILED_CAPTION>" # detection by text prompt
        prompt = task_prompt
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
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
    
    def __call__(self, image : Image.Image) -> str:
        return self.florenceCaptionGenerate(image=image)