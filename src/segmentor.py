import numpy as np
import torch
from torch.autograd import Variable
from skimage import io
from .utils import sliding_window, grouper
from PIL import Image

WINDOW_SIZE = (256, 256) # Patch size
BATCH_SIZE = 10 # Number of samples in a mini-batch
N_CLASSES = 6

palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)
           
def decode_segmap(image):
                
    r = np.zeros_like(image).astype(np.uint8)
    for c, i in palette.items():
        idx = image == 4
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