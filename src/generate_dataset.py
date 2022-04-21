from collections import namedtuple
import numpy as np
from natsort import natsorted
import os
from skimage import io
from PIL import Image
from empatches import EMPatches
import glob
from random import randrange
import PIL.ImageOps
import splitfolders

Label = namedtuple('Label' , ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color', 'm_color',])
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color          multiplied color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0      ),
    Label(  'ship'                 ,  1 ,        0 , 'transport'       , 1       , True         , False        , (  0,  0, 63) , 4128768),
    Label(  'storage_tank'         ,  2 ,        1 , 'transport'       , 1       , True         , False        , (  0, 63, 63) , 4144896),
    Label(  'baseball_diamond'     ,  3 ,        2 , 'land'            , 2       , True         , False        , (  0, 63,  0) , 16128  ),
    Label(  'tennis_court'         ,  4 ,        3 , 'land'            , 2       , True         , False        , (  0, 63,127) , 8339200),
    Label(  'basketball_court'     ,  5 ,        4 , 'land'            , 2       , True         , False        , (  0, 63,191) , 12533504),
    Label(  'Ground_Track_Field'   ,  6 ,        5 , 'land'            , 2       , True         , False        , (  0, 63,255) , 16727808),
    Label(  'Bridge'               ,  7 ,        6 , 'land'            , 2       , True         , False        , (  0,127, 63) , 4161280),
    Label(  'Large_Vehicle'        ,  8 ,        7 , 'transport'       , 1       , True         , False        , (  0,127,127) , 8355584),
    Label(  'Small_Vehicle'        ,  9 ,        8 , 'transport'       , 1       , True         , False        , (  0,  0,127) , 8323072),
    Label(  'Helicopter'           , 10 ,        9 , 'transport'       , 1       , True         , False        , (  0,  0,191) , 12517376),
    Label(  'Swimming_pool'        , 11 ,       10 , 'land'            , 2       , True         , False        , (  0,  0,255) , 16711680),
    Label(  'Roundabout'           , 12 ,       11 , 'land'            , 2       , True         , False        , (  0,191,127) , 8371968),
    Label(  'Soccer_ball_field'    , 13 ,       12 , 'land'            , 2       , True         , False        , (  0,127,191) , 12549888),
    Label(  'plane'                , 14 ,       13 , 'transport'       , 1       , True         , False        , (  0,127,255) , 16744192),
    Label(  'Harbor'               , 15 ,       14 , 'transport'       , 1       , True         , False        , (  0,100,155) , 10183680),
]
palette = {label.id : label.color for label in labels}
invert_palette = {v: k for k, v in palette.items()}
emp = EMPatches()
RAND_RANGE = 6
SEGMAP_RAND_MAP = {0: 1, 1: 2, 3: 8, 4: 9, 5: 10, 6: 14}

OG_DATASET = 'A:/Downloads/Datasets/iSAID/train/RGB Images'
INPAINTING_DATASET = 'A:/Downloads/Datasets/iSAID/train/Inpainting Dataset'
SEMANTIC_MASKS = 'A:/Downloads/Datasets/iSAID/train/Semantic_masks/images'
INPAINTING_OUT = 'A:/Downloads/Datasets/iSAID/train/Augmented Inpainting Dataset'
MASKS_OUT = 'A:/Downloads/Datasets/iSAID/train/Masks'
IRREGULAR_MASKS_TRAIN = 'A:/Downloads/Datasets/qd_imd/train'
IRREGULAR_MASKS_VAL = 'A:/Downloads/Datasets/qd_imd/val'
IRREGULAR_MASKS_TEST = 'A:/Downloads/Datasets/qd_imd/test'

def decode_segmap(image, random=0):
    """Generates numpy array where 1's corresponding to objects of interest as shown in the Label object
       and 0's are everything else

    Args:
        image (array): Array representation of image

    Returns:
        np.array: binary array 
    """
    r = np.zeros_like(image).astype(np.uint8)
    for c, i in palette.items():
        if random:
            num_mask_objects = randrange(2, RAND_RANGE) # Get random number of objects to mask
            objects = []
        else: # Non random masks, remove vehicles
            idx = image == 8
            r[idx] = 255
            idx = image == 9
            r[idx] = 255
    return np.array(r)

def remove_objects(real_img,mask):
    """Remove parts of image based on mask

    Args:
        real_img (np.array): Array representation of image to be altered
        mask (np.array): Binary array representation of mask where pixels set to 1 indicate area to be removed

    Returns:
        np.array: Array representation of altered image
    """
    new_img = real_img.copy()
    idx = mask == 255
    new_img[idx] = 255
    return new_img

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def augment_dataset(dataset_path, out_dir, overlap_per=0.4):
    """Augments dataset by splitting it into 256 x 256 images with overlap percent
       set by percentage

    Args:
        dataset_path (str): file path to dataset to be augmented
        out_dir (str): file path to folder where results will be saved
        overlap_per (float, optional): Overlap percent for mask patches. Defaults to 0.4.
    """
    images = glob.glob(f"{dataset_path}/*.png")
    
    for i, image_path in enumerate(images):        
        image = io.imread(image_path)
        patches, _ = emp.extract_patches(image, patchsize=256, overlap=overlap_per)
            
        for j, patch in enumerate(patches):
            im = Image.fromarray(patch)
            im.save(f"{out_dir}/{i}_{j}_{os.path.basename(image_path)}")

def rename(image_dir, str_remove):
    """Rename default file names produced by inpainting tool Cleanup.Pictures to original pic name

    Args:
        image_dir (str): path to image directory
        str_remove (str): string to remove from file names
    """
    im_paths = glob.glob(f"{image_dir}/*.png")
    for im in im_paths:
        os.rename(im, im.replace(str_remove,''))

def generate_masks(inpainting_path, mask_dir, out_dir, overlap_per=0.4):
    """Generates 256 x 256 masks for inpainting training

    Args:
        inpainting_path (str): path to folder containing inpainted dataset
        mask_dir (str): path to folder containing semantic masks
        out_dir (str): path to folder where masks should be saved
        overlap_per (float, optional): Overlap percent for mask patches. Defaults to 0.4.
    """
    semantic_masks = natsorted([os.path.join(mask_dir, os.path.basename(im)) for im in glob.glob(f"{inpainting_path}/*.png")])

    for i, mask in enumerate(semantic_masks):
        
        binary_mask = np.asarray(Image.fromarray(decode_segmap(convert_from_color(np.asarray(Image.open(mask))))).convert('1'))
        patches, _ = emp.extract_patches(binary_mask, patchsize=256, overlap=overlap_per)
        
        for j, patch in enumerate(patches):
            if np.mean(patch) == 0: # Skip all black masks
                continue
            im = Image.fromarray(patch)
            im.save(f"{out_dir}/{i}_{j}_{os.path.basename(mask)}", format='PNG')

def remove_vehicles(dataset, masks, out_dir):
    """Alter images by removing vehicles

    Args:
        dataset (str): Path to folder containing original dataset
        masks (str): Path to folder containing semantic masks corresponding to original dataset
        out_dir (str): Path to folder where output should be saved
    """
    image_paths = natsorted(os.path.join(dataset, patch) for patch in os.listdir(dataset))
    mask_paths = natsorted(os.path.join(masks, patch) for patch in os.listdir(masks))
    
    for i in range(len(image_paths)):
    
        im = Image.open(image_paths[i])
        mask = decode_segmap(convert_from_color(np.asarray(Image.open(mask_paths[i]))))
        
        height, width = mask.shape
        img = np.array(im.resize((width, height), Image.ANTIALIAS))
        new_img = remove_objects(img, mask)

        im = Image.fromarray(new_img)
        im.save(f"{out_dir}/{os.path.basename(image_paths[i])}")

def invert_masks(mask_path):
    """Inverts binary masks in directory

    Args:
        mask_path (str): Path to folder containing mask images
    """
    masks =  glob.glob(f"{mask_path}/*.png")
    for mask in masks:
        image = np.asarray(Image.open(mask).convert('1'))
        invert_image = np.invert(image)
        im = Image.fromarray(invert_image)
        im.save(mask, format='PNG')

def resize(mask_path, out_dir, size=(256,256)):
    """Resizes images to given size. Preserves aspect ratio.

    Args:
        mask_path (str): Path to images to resize
        out_dir (str): Path to save images
        size (tuple, optional): Size to resize imags to. Defaults to (256,256).
    """
    masks =  glob.glob(f"{mask_path}/*.png")
    for mask in masks:
        im = Image.open(mask)
        im.thumbnail(size, Image.ANTIALIAS)
        im = im.convert('1')
        im.save(f"{out_dir}/{os.path.basename(mask)}", format='PNG')

def generate_splits(dataset_path, out_dir, p_seed, p_ratio):
    splitfolders.ratio(dataset_path, output=out_dir, seed=p_seed, ratio=p_ratio)

def augment_masks(mask_path):
    resize(mask_path, mask_path)
    invert_masks(mask_path)

def generate_dataset():
    remove_vehicles(OG_DATASET, SEMANTIC_MASKS)
    augment_dataset(INPAINTING_DATASET, INPAINTING_OUT)
    rename(INPAINTING_DATASET, '_cleanup')
    augment_masks(IRREGULAR_MASKS_TRAIN)
    augment_masks(IRREGULAR_MASKS_VAL)
    augment_masks(IRREGULAR_MASKS_TEST)

if __name__ == "__main__":
    generate_dataset()