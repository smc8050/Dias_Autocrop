from PIL import ImageChops, Image, ImageDraw, ImageStat, ImageFilter
import cv2 as cv
import numpy as np
import os
from shutil import copy2
import multiprocessing


def crop_dias(params):
    """
    This is the main function for cropping images with a black border.
    The main cropping idea is inspired by Matt Howell,
    from https://mail.python.org/pipermail/image-sig/2008-July/005092.html
    :param params: list of parameters which have to be passed:
        input_path: Path of the file which has to be cropped
        name: name of the cropped file without extension
        border: additional crop # number of pixels to cut border effects from dia
    :return: none
    """
    img_in_path = params['input_path']
    output_dir = params['output_dir']
    img_name = params['name']
    cropp_addition = params['border']

    debug = False

    imcv2 = cv.imread(img_in_path)

    # TODO: Rotation before cropping has to be optimised (more accuracy)
    # theta = get_rotation(imcv2)
    # imcv2 = rotate_image(imcv2, 180 * theta / np.pi - 90)

    img = Image.fromarray(cv.cvtColor(imcv2, cv.COLOR_BGR2RGB))  # convert from cv2 image file to pil image file
    original_img = img
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=4))  # to remove outlier pixels
    binary_img = convert_to_binary(blurred_img,40,255)
    bg = Image.new(binary_img.mode, binary_img.size)
    diff = ImageChops.difference(binary_img, bg)
    bbox = diff.getbbox()
    if bbox:
        color_control = get_control_value(original_img, bbox)
        if color_control > 1.5:
            # if the color_control value is higher than the threshold, the file will be listed in the txt file and
            # saved in a seperate folder (need_review) for manual review together with a compressed version of the
            # suggested crop boundaries
            with open(f"{output_dir}/need_review.txt", "a+") as f:
                f.write(f'{img_name};{color_control}\n')
            output_dir = f"{output_dir}/need_review"
            review_image = draw_cropline(original_img, bbox)
            review_image.save(f'{output_dir}/{img_name}_review.jpg', quality=20, optimize=True)
        if debug:
            # for debugging purpose the pictures will not be cropped
            # but the crop boundaries are drawn instead for reference
            im = draw_cropline(original_img, bbox)
        else:
            im = original_img.crop(bbox)
            im = subtract_border(im, cropp_addition)
        im.save(f'{output_dir}/{img_name}_cropped.jpg', quality=100, subsampling=0)
        print(f'cropped \"{img_name}\", (Control value: {color_control})')
    else:
        # found no bounding box to crop,
        # will be copied to the failed_files folder
        print(f'cannot trim {img_name}')
        with open(f'{output_dir}/failed.txt', "a+") as f:
            f.write(f'{img_name}\n')
        copy2(img_in_path, f'{output_dir}/failed_files')


def get_rotation(imcv2):
    """
    TODO:get more accurate angle
    This function gets the rotation angle of the image via a Hough transform
    :param imcv2: cv2 image
    :return: the tilt angle theta
    """
    global theta
    gray = cv.cvtColor(imcv2, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    # theta = np.mean(lines, axis=0)[0][1]
    # rho = np.mean(lines, axis=0)[0][0]

    debug = True
    if debug:
        # save image with angled line for reference
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 60000 * (-b))
            y1 = int(y0 + 60000 * a)
            x2 = int(x0 - 60000 * (-b))
            y2 = int(y0 - 60000 * a)
            cv.line(imcv2, (x1, y1), (x2, y2), (0, 0, 255), 5)
        cv.imwrite('DEBUG_houghlines.jpg', imcv2)
    return theta


def rotate_image(imcv2, angle):
    """
    This function rotates the image about the center with the angle theta
    :param imcv2: OpenCV image array
    :param angle: angle if desired rotation in degree
    :return: returns the rotated image
    """
    image_center = tuple(np.array(imcv2.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(imcv2, rot_mat, imcv2.shape[1::-1], flags=cv.INTER_LINEAR)
    # cv.imwrite('rotated.jpg', result)
    return result


def convert_to_binary(image,lower_threshold,upper_threshold):
    """
    This function converts a image to a binary image
    :param image: OpenCV image array
    :return: Binary image as OpenCV image array
    """
    original_imgcv2 = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
    grayImage_imgcv2 = cv.cvtColor(original_imgcv2, cv.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv.threshold(grayImage_imgcv2, lower_threshold, upper_threshold, cv.THRESH_BINARY)
    # cv.imwrite("binary.jpg", blackAndWhiteImage)
    image = Image.fromarray(
        cv.cvtColor(blackAndWhiteImage, cv.COLOR_BGR2RGB))  # convert from cv2 image file to pil image file
    return image


def subtract_border(im, border):
    """
    This function substracts a equal sized border from all sides of the image
    :param im: PIL Image file
    :param border: width of border in pixels
    :return: cropped PIL Image file
    """
    width, height = im.size  # Get dimensions
    left = border
    top = border
    right = width - border
    bottom = height - border
    im = im.crop((left, top, right, bottom))  # crop borders
    return im


def draw_cropline(im, bbox):
    """
    This function draws the bounding box in the picture
    :param im: PIL image
    :param bbox: PIL bbox array
    :return: PIL image with drawn bbox
    """
    draw = ImageDraw.Draw(im)
    draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline="red", width=5)
    return im


def get_control_value(im, bbox):
    """
    This function gets the mean color value outside of the bounding box (bbox).
    The closer to 0 the darker is the section that will be removed. If the value is
    higher than a certain threshold this could indicate an wrong autocrop boundingbox
    and non black pixels are cropped.
    :param im: PIL Image
    :param bbox: PIL bbox array
    :return: mean color value (float)
    """
    control_img = im
    mask_layer = Image.new("L", control_img.size, 255)
    draw = ImageDraw.Draw(mask_layer)
    draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=0)
    avg_list = ImageStat.Stat(control_img, mask=mask_layer).mean
    avg = round(sum(avg_list) / 3, 2)
    return avg


def create_file_structure(base_directory):
    """
    This function creates the directory structure which this script needs.
    The created structure looks as follows:
        base_directory/
        ├── failed.txt
        ├── need_review.txt
        ├── failed_files/
        └── need_review/

    :param base_directory: String of the path to the base directory
    :return: none
    """
    if not os.path.isdir(base_directory):
        os.mkdir(base_directory)
    if len(os.listdir(base_directory)) == 0:
        os.mkdir(f'{base_directory}/failed_files')
        os.mkdir(f'{base_directory}/need_review')
        with open(f"{base_directory}/need_review.txt", "a+") as f:
            f.write("The following pictures need reviewing as the control value is to high, and are thus"
                    "saved in the subfolder failed_files together with a reference image\n"
                    "Please consider comparing the cropped with the original image.\n")
        with open(f"{base_directory}/failed.txt", "a+") as f:
            f.write("Following files failed:\n")
    else:
        print('Output directory is not empty...\n'
              'Please choose a empty directory as output.')
        quit()


def get_thread_count():
    """
    This function counts the CPU cores and multiplies by 2 to get the # of threads
    :return: num_threads: number of threads
    """
    try:
        num_threads = multiprocessing.cpu_count()
        print(f"Using {num_threads} threads.")
    except:
        print("Automatic thread detection didn't work. Defaulting to 1 thread only.")
        num_threads = 1
    return num_threads
