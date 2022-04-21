from collections import namedtuple
import numpy as np
from natsort import natsorted
import os
from skimage import io
from PIL import Image
from empatches import EMPatches
import glob

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

def decode_segmap(image):
                
    r = np.zeros_like(image).astype(np.uint8)
    for c, i in palette.items():
        idx = image == 8
        r[idx] = 255
        idx = image == 9
        r[idx] = 255
    return np.array(r)

def remove_objects(real_img,mask):
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
    """
    images = glob.glob(f"{dataset_path}/*.png")
    emp = EMPatches()
    
    for i, image_path in enumerate(images):        
        image = io.imread(image_path)
        patches, _ = emp.extract_patches(image, patchsize=256, overlap=overlap_per)
            
        for j, patch in enumerate(patches):
            im = Image.fromarray(patch)
            im.save(f"{out_dir}/{i}_{j}_{os.path.basename(image_path)}")

def rename(image_dir):
    """Rename default file names produced by inpainting tool Cleanup.Pictures to original pic name

    Args:
        image_dir (str): path to image directory
    """
    im_paths = glob.glob(f"{image_dir}/*.png")
    for im in im_paths:
        os.rename(im, im.replace('cleanup',''))

def collect_images(dataset_path):
    """Pulls images from iSAID corresponding to images produced by Cleanup.Pictures

    Args:
        dataset_path (str): file path to dataset
    """

def generate_masks(dataset_path, out_dir, overlap_per=0.4):
    """Generates masks for inpainting training
    """


if __name__ == "__main__":
    '''
    image_paths = natsorted(os.path.join('A:/Downloads/Datasets/iSAID/train/RGB Images', patch) for patch in os.listdir('A:/Downloads/Datasets/iSAID/train/RGB Images'))
    mask_paths = natsorted(os.path.join('A:/Downloads/Datasets/iSAID/train/Semantic_masks/images', patch) for patch in os.listdir('A:/Downloads/Datasets/iSAID/train/Semantic_masks/images'))
    
    for i in range(len(image_paths)):
    
        im = Image.open(image_paths[i])
        mask = decode_segmap(convert_from_color(np.asarray(Image.open(mask_paths[i]))))
        
        height, width = mask.shape
        img = np.array(im.resize((width, height), Image.ANTIALIAS))
        new_img = remove_objects(img, mask)

        im = Image.fromarray(new_img)
        im.save(f"A:/Downloads/Datasets/iSAID/train/Altered Images/{os.path.basename(image_paths[i])}")
    '''
    augment_dataset('A:/Downloads/Datasets/iSAID/train/Inpainting Dataset', 'A:/Downloads/Datasets/iSAID/train/Augmented Inpainting Dataset')