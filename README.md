# ImageDescripor

A annotate tools geared text-to-image Fine-tuning datasets

## Description

A annotate tools geared text-to-image Fine-tuning datasets,

Auto crop image and add caption

### example 

We have a folder of pictures.

One of them look like this.

![A example photo by https://www.facebook.com/hieuthaooo/](/assets/images/example.jpg)

#### step 1

We will use pre-training object recogntion model to crop the main body from source image.

There are several principles for cropping pictures.  

- Keep image as large as possible.
- Keep the subject in the center or in line with the composition.

#### step 2

add a caption by Vision model(as gpt4o BLIP 3 ...)

## Getting Started

### Dependencies

python >= 3.8

### Installing

`git clone https://github.com/dudulu-rgzn/ImageDescripor.git`
`python -m venv .venv`

#### Activate environment

##### Linux

`source .venv/bin/activate`

##### Windows

`.venv/Scripts/activate.bat`

#### Installing Python Dependencies

`pip install -r requirements.txt`

### Executing program

`cd src`
edit config.py
`python main.py`

## Help

To do

## Authors

To do

## Version History


## License

This project is licensed under the [Apache License Version 2.0 License](https://www.apache.org/licenses/LICENSE-2.0#apache-license-version-20)

## Acknowledgments

To do

## Useful Links

[Developer Discussions](https://github.com/dudulu-rgzn/ImageDescripor/discussions)

[Fine-tuning FLUX.1[dev]](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md)

[dataset config.json](https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md)

[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling/tree/main)
