import numpy as np
from PIL import Image, ImageChops
import image_slicer, os, shutil, sklearn
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from skimage import io
from empatches import EMPatches
from ast import literal_eval

def pad_n_slice(image):
    arr = np.asarray(image)

def slice_img(img):
    
    image = io.imread(img)
    emp = EMPatches()
    patches, indices = emp.extract_patches(image, patchsize=256, overlap=0.1)
    
    with open('patches.txt', 'w') as f:
        f.write(str(patches))

    with open('indices.txt', 'w') as f:
        f.write(str(indices))
    
    for i in range(len(patches)):
        im = Image.fromarray(patches[i])
        im.save(f"./Patches/{i}_{os.path.basename(img)}")

def stitch_patches(patch_dir, out_dir):
    
    emp = EMPatches()
    patches = [io.imread(os.path.join(patch_dir, patch)) for patch in os.listdir(patch_dir)]
   
    with open('patches_array.txt', 'w') as f:
        f.write(str(patches))
    
    with open('indices.txt') as f:
        indices = literal_eval(f.readlines()[0])
    
    im = emp.merge_patches(patches, indices) 
    image = Image.fromarray(im)
    image.save(f"{out_dir}/result.png")

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

if __name__ == "__main__":
    slice_img('Tests/top_mosaic_09cm_area5.png')
    stitch_patches('./Patches', '.')