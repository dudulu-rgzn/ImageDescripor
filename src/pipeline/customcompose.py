from typing import Any
from pipeline.basepipeline import BaseComposePipeline, BaseCropPipeline, BaseCaptionPipeline

class CustomComposePipeline(BaseComposePipeline):
    def __init__(self,cropPipeline:BaseCropPipeline,captionPipeline:BaseCaptionPipeline) -> None:
        self.cropPipeline = cropPipeline
        self.captionPipeline = captionPipeline
    
    def __call__(self, image) -> tuple[Any,str]:
        return self.cropPipeline(image),self.captionPipeline(image)