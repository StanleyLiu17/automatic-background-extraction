import numpy as np
import torch
from torch.autograd import Variable
from skimage import io
from .utils import sliding_window, grouper
from PIL import Image
from collections import namedtuple

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

WINDOW_SIZE = (256, 256) # Patch size
BATCH_SIZE = 10 # Number of samples in a mini-batch
N_CLASSES = len(palette.keys())

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

def segmentor(net, image, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    
    img = (1 / 255 * np.asarray(image, dtype='float32'))
    pred = np.zeros(img.shape[:2] + (N_CLASSES,))

    for i, coords in enumerate(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
                    
        with torch.no_grad():
            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda())
                
            # Do the inference
            outs = net(image_patches)
            outs = outs.data.cpu().numpy()
                
            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h] += out
            del(outs)
    
    mask = decode_segmap(np.argmax(pred, axis=-1))
    height, width = mask.shape
    img = np.array(image.resize((width, height), Image.ANTIALIAS))
    new_img = remove_objects(img, mask)
    return new_img, mask