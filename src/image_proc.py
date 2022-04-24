from PIL import Image, ImageChops
import os, re
from requests import patch
from skimage import io
from empatches import EMPatches
from ast import literal_eval
from natsort import natsorted
from copy import copy
import pickle

emp = EMPatches()

def slice_img(img, i):
    
    image = io.imread(img)
    patches, indices = emp.extract_patches(image, patchsize=256, overlap=0.2)
    
    with open('indices.txt', 'a+') as f:
        f.write(str(indices))
    
    for j, patch in enumerate(patches):
        im = Image.fromarray(patch)
        im.save(f"./Patches/{i}_{j}_{os.path.basename(img)}")
            
def stitch_patches(patch_dir, out_dir, i):
    
    patches = [io.imread(patch) for patch in natsorted([os.path.join(patch_dir, patch) for patch in os.listdir(patch_dir)])]
    
    with open('indices.txt') as f:
        indices = [literal_eval(f.readlines[i])]
    
    image = Image.fromarray(emp.merge_patches(patches, indices))
    image.save(f"{out_dir}/result_{i}.png")

def slice_img_dir(image_paths):
    """This function takes in a directory of images, slices them all into 256 x 256 patches, and places them
       into ./Patches
       Each patchs' filename will take the form {i}_{j}_filename where i represents the i'th image
       in the input directory and j represents the j'th patch comprising the total i'th image
       It stores a list of partition arrays for each image in indices.txt, so the order that patches
       are postprocessed is important.
    Args:
        image_paths (List): List of strings representing file paths for each image

    Returns:
        Str: String representing relative path of directory where patches are saved
    """

    with open('indices.txt', 'a+') as f:
        f.truncate(0)
        for i, image_path in enumerate(natsorted(image_paths)):
            
            image = io.imread(image_path)
            
            patches, indices = emp.extract_patches(image, patchsize=256, overlap=0.2)
            f.write(str(indices)+'\n')
            
            for j, patch in enumerate(patches):
                im = Image.fromarray(patch)
                im.save(f"./Patches/{i}_{j}_{os.path.basename(image_path)}")
    
    return [os.path.join('./Patches', im) for im in os.listdir('./Patches')]

def stitch_patches_dir(out_dir):
    """This function takes an output directory and stitches patches produced by inference. It 
       'naturally sorts' file patches by their filename (e.g. 1_1.png, 1_2.png, 2_1.png, etc.)
       and uses indices.txt to get the according list of partition arrays. 

    Args:
        out_dir (Str): String representation of directory to save output to. This will be provided by argparse.
    """
    all_patch_paths = natsorted(os.path.join('./Patches/results', patch) for patch in os.listdir('./Patches/results'))
    
    # Get number of images to recover from patches
    with open('indices.txt') as f:
        image_num = len([line for line in f.readlines() if line.strip()])
    
    with open('indices.txt') as f:
        for i in range(0, image_num):
            patch_counter, curr_patches = 0, []
            
            for j, patch_path in enumerate(all_patch_paths): # Read through list of all patches
                if int(re.search(r'\d+', os.path.basename(patch_path)).group()) == i:
                    patch_counter = copy(j) # If x = i where 'x_y_*.png', then it belongs to image being built in current iteration, record y num
                    continue
                break
            
            curr_patches = all_patch_paths[:patch_counter + 1] # Get patch names using patch_counter
            all_patch_paths = all_patch_paths[patch_counter + 1:] # Remove patch names from list of patches
            patches = [io.imread(patch) for patch in curr_patches]
            
            indices = literal_eval(f.readline())
            image = Image.fromarray(emp.merge_patches(patches, indices))
            image.save(f"{out_dir}/result_{i}.png")
        
def cleanup():
    """Cleans up folders and files used to pre/post-process images
    """
    if os.path.exists('indices.txt'):
        os.remove('indices.txt')
    if os.path.exists('./Patches'):
        os.rmdir('./Patches')

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

if __name__ == "__main__":
    stitch_patches_dir('./results')