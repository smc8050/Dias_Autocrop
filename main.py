from utils import *
import os
from multiprocessing import Pool
import datetime

# This project was inspired by z80z80z80 on https://github.com/z80z80z80/autocrop
# Thanks for that!

def main(input_dir, output_dir):
    thread_count = get_thread_count()
    params = []
    create_file_structure(output_dir)
    print("Setting up file parameters for threads...")
    picture_counter = 0
    tstart = datetime.datetime.now()  # start timer
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            picture_counter += 1
            params.append({"input_path": os.path.join(input_dir, filename),  # path of the image file
                           "output_dir": output_dir,  # output directory where all data will be saved
                           "name": os.path.splitext(filename)[0],  # get filename without extension
                           "border": 20})  # additional # of pixels cropped from border
    print("Start cropping...")
    with Pool(thread_count) as p:
        p.map(crop_dias, params)
    tend = datetime.datetime.now()  # stop timer
    tdelta = tend - tstart
    print(f"Batch finished, {picture_counter} Pictures in {tdelta.seconds} Seconds")


if __name__ == "__main__":
    main("Original", "Boxed")
