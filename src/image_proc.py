import numpy as np
import cv2
from PIL import Image, ImageChops
import image_slicer, os, shutil, sklearn
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from skimage import io

def pad_n_slice(image):
    arr = np.asarray(image)

def slice_img(img):
    image = io.imread(img)
    patches = extract_patches_2d(image, (256, 256), max_patches=20)
    np.savetxt('patches.txt', patches)
    for i in range(len(patches)):
        im = Image.fromarray(patches[i])
        im.save(f"./Patches/{i}.png")

def stitch_patches(image, dir):
    patches, patch_array = os.listdir(dir), []
    
    for patch in patches:
        patch_array.append(np.asarray(patch))
    
    #image = reconstruct_from_patches_2d(patch_array)

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

if __name__ == "__main__":
    slice_img('Tests/top_mosaic_09cm_area5.png')