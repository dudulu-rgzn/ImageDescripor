from tqdm import tqdm

from dataset import Dataset
from pipeline import FlorencePipeline
from config import config
from save import DateSaver

dataset = Dataset(config=config["dataset"])

florencePipeline = FlorencePipeline(r"models\AI-ModelScope\Florence-2-large-ft")
# cropimage , text = florencePipeline()
# print(text)

saver = DateSaver(config=config)
for image_path,image in tqdm(dataset.get_images(),total=len(dataset)):
    cropimage , text = florencePipeline(image)
    saver.save(image=cropimage,caption=text)
    
