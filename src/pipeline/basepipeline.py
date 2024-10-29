from typing import Any,Tuple

class BaseComposePipeline:
    def __init__(self,config) -> None:
        pass
    
    def __call__(self, image) -> Tuple[Any,str]:
        text = "image context"
        return image, text

class BaseCaptionPipeline:
    """
    Base Class of all caption pipeline
    """
    def __init__(self,model) -> None:
        self.model = model
        
    def __call__(self, image) -> str :
        return self.model(image)
    
class BaseCropPipeline:
    """
    Base Class of all caption pipeline
    """
    def __init__(self,model) -> None:
        self.model = model
        
    def __call__(self, image) -> Any :
        """
        input:
        image
        output:
        image
        """
        return self.model(image)