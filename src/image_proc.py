from PIL import Image, ImageChops
import os
from skimage import io
from empatches import EMPatches
from ast import literal_eval
from natsort import natsorted

def slice_img(img):
    
    image = io.imread(img)
    emp = EMPatches()
    patches, indices = emp.extract_patches(image, patchsize=256, overlap=0.1)

    with open('indices.txt', 'a+') as f:
        f.write(str(indices))
    
    for i in range(len(patches)):
        im = Image.fromarray(patches[i])
        im.save(f"./Patches/{i}_{os.path.basename(img)}")

def stitch_patches(patch_dir, out_dir, i):
    
    emp = EMPatches()
    patches = [io.imread(patch) for patch in natsorted([os.path.join(patch_dir, patch) for patch in os.listdir(patch_dir)])]
    
    with open('indices.txt') as f:
        indices = [literal_eval(index) for index in f.readlines()]
    
    im = emp.merge_patches(patches, indices) 
    image = Image.fromarray(im)
    image.save(f"{out_dir}/result_{i}.png")

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)