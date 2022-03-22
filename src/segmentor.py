import numpy as np

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

def segmentor(net, test_ids, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    all_preds = []
    
    # Switch the network to inference mode
    net.eval()
    
    for img in test_images:
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

        pred = np.argmax(pred, axis=-1)
        all_preds.append(pred)
    
    return all_preds