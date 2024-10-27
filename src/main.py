from PIL import Image

from pipeline import FlorencePipeline

image = Image.open(r"dataset\raw\dev-dataset\car.jpg")

florencePipeline = FlorencePipeline(r"models\AI-ModelScope\Florence-2-large-ft")
cropimage , text = florencePipeline(image=image)
print(text)