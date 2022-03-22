import numpy as np
import torch
from torch.autograd import Variable
from skimage import io
from networks import SegNet
from utils import sliding_window, grouper

net = SegNet()
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

def fill_gaps(values):
    searchval=[255,0,255]
    searchval2=[255,0,0,255]
    idx=(np.array(np.where((values[:-2]==searchval[0]) & (values[1:-1]==searchval[1]) & (values[2:]==searchval[2])))+1)
    idx2=(np.array(np.where((values[:-3]==searchval2[0]) & (values[1:-2]==searchval2[1]) & (values[2:-1]==searchval2[2]) & (values[3:]==searchval2[3])))+1)
    idx3=(idx2+1)
    new=idx.tolist()+idx2.tolist()+idx3.tolist()
    newlist = [item for items in new for item in items]
    values[newlist]=255
    return values

def fill_gaps2(values):
    searchval=[0,255]
    searchval2=[255,0]
    idx=(np.array(np.where((values[:-1]==searchval[0]) & (values[1:]==searchval[1]))))
    idx2=(np.array(np.where((values[:-1]==searchval[0]) & (values[1:]==searchval[1])))+1)
    
    new=idx.tolist()+idx2.tolist()
    newlist = [item for items in new for item in items]
    values[newlist]=255
    return values

def remove_patch_og(real_img,mask):
    og_data = real_img.copy()
    idx = mask == 255
    og_data[idx] = 255
    return og_data

def segmentor(net, img, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    
    net.load_state_dict(torch.load('./checkpoints/segnet_final_reference.pth'))

    # Switch the network to inference mode
    net.cuda()
    net.eval()
    
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

    return np.argmax(pred, axis=-1)