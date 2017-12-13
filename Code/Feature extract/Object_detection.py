from ctypes import *
import math
import random
import os
from matplotlib import pyplot as plt

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.2, hier_thresh=.2, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    caizhi = ['bed','sofa','chair','dinningtable']
    zhonglei = ['sink','oven','refrigerator','microwave','tvmonitor']
    net = load_net("cfg/yolo.cfg", "yolo.weights", 0)
    meta = load_meta("cfg/coco.data")
    '''r = detect(net, meta, '/home/yihan/Downloads/darknet/')
    for i in r:
        print i'''
    picture=[]
    path = os.listdir('/home/yihan/Downloads/photo/newphoto')
    path.sort()
    path.reverse()
    output_lst = list()
    for k in path:
        img_direct = '/home/yihan/Downloads/photo/newphoto/'+k
        cz_direct = '/home/yihan/Downloads/photo/caizhi/'+k
	picture = os.listdir(img_direct)
	for i in picture:
            r = detect(net, meta, img_direct+'/'+i)    
	    print img_direct+'/'+ i
            r = detect(net, meta, img_direct+'/'+i)
	    
	    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
	    m = 0
	    a = 0
            for j in r:
                print j
                if (j[0] in caizhi):
                    whole_img = plt.imread(img_direct+'/'+ i)
		    x1 = int(j[2][0]) - int(j[2][2])//2
                    if x1 < 0:
                        x1 = 0
		    y1 = int(j[2][1]) - int(j[2][3])//2
                    if y1 < 0:
                        y1 = 0
		    x2 = int(j[2][0]) + int(j[2][2])//2
                    if x2 > whole_img.shape[1]:
                        x2 = whole_img.shape[1] 
		    y2 = int(j[2][1]) + int(j[2][3])//2
                    if y2 > whole_img.shape[0]:
                        y2 = whole_img.shape[0]
		    img_seg = whole_img[y1:y2,x1:x2,:]
                    plt.imsave(cz_direct+'/'+j[0]+str(lst[m])+'.jpg',img_seg)
                    m += 1
                if (j[0] in zhonglei):
                    print j[0]
		    

