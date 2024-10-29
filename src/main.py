from dataset import Dataset
from pipeline import FlorencePipeline
from config import config
                
dataset = Dataset(config=config["dataset"])

florencePipeline = FlorencePipeline(r"models\AI-ModelScope\Florence-2-large-ft")
# cropimage , text = florencePipeline()
# print(text)

for image in dataset.get_images():
    cropimage , text = florencePipeline(image)
    print(text)
