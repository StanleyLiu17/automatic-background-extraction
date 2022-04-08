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

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)