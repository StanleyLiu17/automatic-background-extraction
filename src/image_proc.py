import numpy as np
import cv2
from PIL import Image, ImageChops
import image_slicer, os, shutil, sklearn

def pad_n_slice(image):
    arr = np.asarray(image)


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

