# Dias Autocrop
This python script enables automatic cropping of black borders from dia-images which are fotographed with a digital camera. Some samples can be found in the Sample_images folder.

## Usage
In the `main.py` specify the input directory where the files to be cropped are located and the output directory, where the cropped files will be saved. To start the batch just run the main file in the same folder with the `utils.py` module. 

## Dependencies
This script depends on the `Pillow` and `opencv-python` package, to install them use following commands:

`pip install Pillow`

`pip install opencv-python`

## Remark
If this script does not work for your files or you need the ability to autimatically rotate and/or transform your images please check out [this](https://github.com/z80z80z80/autocrop) python script from z80z80z80 which does exactly that.
