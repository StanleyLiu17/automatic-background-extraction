from PIL import Image, ImageChops
import os, re
from skimage import io
from empatches import EMPatches
from ast import literal_eval
from natsort import natsorted

emp = EMPatches()

def slice_img(img, i):
    
    image = io.imread(img)
    patches, indices = emp.extract_patches(image, patchsize=256, overlap=0.1)

    with open('indices.txt', 'a+') as f:
        f.write(str(indices))
    
    for j in range(len(patches)):
        im = Image.fromarray(patches[i])
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
        for i, image_path in enumerate(len(image_paths)):
            
            image = io.imread(image_path)    
            patches, indices = emp.extract_patches(image, patchsize=256, overlap=0.1)
            f.write(str(indices))
            
            for j in range(len(patches)):
                im = Image.fromarray(patches[i])
                im.save(f"./Patches/{i}_{j}_{os.path.basename(image_path)}")

    return './Patches'

def stitch_patches_dir(out_dir):
    """This function takes an output directory and stitches patches produced by inference. It 
       'naturally sorts' file patches by their filename (e.g. 1_1.png, 1_2.png, 2_1.png, etc.)
       and uses indices.txt to get the according list of partition arrays. 

    Args:
        out_dir (Str): String representation of directory to save output to. This will be provided by argparse.
    """
    all_patch_paths = natsorted(os.path.join('./Patches/results', patch) for patch in os.listdir('./Patches/results'))
    
    for i, patch_path in enumerate(len(all_patch_paths)):
        patch_counter = 0
        
        while int(re.search(r'\d+', os.path.basename(patch_path)).group()) == i:
            patch_counter += 1
        
        patch_paths = all_patch_paths[0:patch_counter + 1] # Get patch paths belonging to one image
        all_patch_paths = all_patch_paths[patch_counter + 2:] # Remove patch paths from list

        patches = [io.imread(patch) for patch in patch_paths]
        
        with open('indices.txt') as f:
            indices = [literal_eval(f.readlines[i])]
        
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