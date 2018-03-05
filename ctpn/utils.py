import numpy as np 
import xmltodict 
import os
import cv2
import matplotlib.pyplot as plt 

import tensorflow as tf 

anchor_scale = 16
#
IOU_NEGATIVE =0.3
IOU_POSITIVE = 0.7
IOU_SELECT =0.7 

RPN_POSITIVE_NUM=150
RPN_TOTAL_NUM=300

#bgr  can find from  here https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68,116.779,103.939]

DEBUG = True


def readxml(path):
    gtboxes=[]
    imgfile = ''
    with open(path,'rb') as f :
        xml = xmltodict.parse(f)
        bboxes = xml['annotation']['object']
        if(type(bboxes)!=list):
            x1 = bboxes['bndbox']['xmin']
            y1 = bboxes['bndbox']['ymin']
            x2 = bboxes['bndbox']['xmax']
            y2 = bboxes['bndbox']['ymax']
            gtboxes.append((int(x1),int(y1),int(x2),int(y2)))
        else:
            for i in bboxes:
                x1 = i['bndbox']['xmin']
                y1 = i['bndbox']['ymin']
                x2 = i['bndbox']['xmax']
                y2 = i['bndbox']['ymax']
                gtboxes.append((int(x1),int(y1),int(x2),int(y2)))

        imgfile = xml['annotation']['filename']
    return np.array(gtboxes),imgfile


def gen_anchor(featuresize,scale):

    """
    gen base anchor from feature map [HXW][9][4]
    reshape  [HXW][9][4] to [HXWX9][4]
    """
    
    heights=[11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths=[16,16,16,16,16,16,16,16,16,16]

    #gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights),1)
    widths = np.array(widths).reshape(len(widths),1)
  
    base_anchor = np.array([0,0,15,15])
    #center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5 
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2 
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5 
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5  
    base_anchor = np.hstack((x1,y1,x2,y2))

    h,w = featuresize
    shift_x = np.arange(0,w) * scale
    shift_y = np.arange(0,h) * scale
    #apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append( base_anchor + [j,i,j,i])          
    return np.array(anchor).reshape((-1,4))


def cal_iou(box1,box1_area, boxes2,boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0],boxes2[:,0])
    x2 = np.minimum(box1[2],boxes2[:,2])
    y1 = np.maximum(box1[1],boxes2[:,1])
    y2 = np.minimum(box1[3],boxes2[:,3])

    intersection = np.maximum(x2-x1,0) * np.maximum(y2-y1,0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou

def cal_overlaps(boxes1,boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box
    
    """
    area1 = (boxes1[:,0] - boxes1[:,2]) * (boxes1[:,1] - boxes1[:,3])
    area2 = (boxes2[:,0] - boxes2[:,2]) * (boxes2[:,1] - boxes2[:,3])

    overlaps = np.zeros((boxes1.shape[0],boxes2.shape[0]))

    #calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i],area1[i],boxes2,area2)
    
    return overlaps


def bbox_transfrom(anchors,gtboxes):
    """
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor 
    """
    regr = np.zeros((anchors.shape[0],2))
    Cy = (gtboxes[:,1] + gtboxes[:,3]) * 0.5
    Cya = (anchors[:,1] + anchors[:,3]) * 0.5
    h = gtboxes[:,3] - gtboxes[:,1] + 1.0
    ha = anchors[:,3] - anchors[:,1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h/ha)

    return np.vstack((Vc,Vh)).transpose()

def bbox_transfor_inv(anchor,regr):
    """
        return predict bbox
    """
    
    Cya = (anchor[:,1] + anchor[:,3]) * 0.5
    ha = anchor[:,3] - anchor[:,1] + 1 

    Vcx = regr[0,:,0]
    Vhx = regr[0,:,1]

    Cyx = Vcx * ha + Cya
    hx = np.exp(Vhx) * ha
    xt = (anchor[:,0] + anchor[:,2]) * 0.5

    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5 
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5  
    bbox = np.vstack((x1,y1,x2,y2)).transpose()
    
    return bbox


def clip_box(bbox,im_shape):
    
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox,minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


def cal_rpn(imgsize,featuresize,scale,gtboxes):

    imgh,imgw = imgsize
    
    #gen base anchor
    base_anchor = gen_anchor(featuresize,scale)

    #calculate iou
    overlaps = cal_overlaps(base_anchor,gtboxes)

    #init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)
  
    #for each GT box corresponds to an anchor which has highest IOU 
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    
 
    #the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]),anchor_argmax_overlaps]     

    #IOU > IOU_POSITIVE
    labels[anchor_max_overlaps>IOU_POSITIVE]=1
    #IOU <IOU_NEGATIVE
    labels[anchor_max_overlaps<IOU_NEGATIVE]=0
    #ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1

    #only keep anchors inside the image
    outside_anchor = np.where(
       (base_anchor[:,0]<0) |
       (base_anchor[:,1]<0) |
       (base_anchor[:,2]>=imgw)|
       (base_anchor[:,3]>=imgh)
       )[0]
    labels[outside_anchor]=-1


    #subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels==1)[0]
    if(len(fg_index)>RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index,len(fg_index)-RPN_POSITIVE_NUM,replace=False)]=-1

    #subsample negative labels 
    bg_index = np.where(labels==0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(labels==1)
    if(len(bg_index)>num_bg):
        #print('bgindex:',len(bg_index),'num_bg',num_bg)
        labels[np.random.choice(bg_index,len(bg_index)-num_bg,replace=False)]=-1
        

    #calculate bbox targets
    # debug here 
    bbox_targets = bbox_transfrom(base_anchor,gtboxes[anchor_argmax_overlaps,:])
    #bbox_targets=[]


    return [labels,bbox_targets],base_anchor


def get_session(gpu_fraction=0.6):  
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''  
  
    num_threads = os.environ.get('OMP_NUM_THREADS')  
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)  
  
    if num_threads:  
        return tf.Session(config=tf.ConfigProto(  
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))  
    else:  
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
  
class random_uniform_num():
    """
    uniform random
    """
    def __init__(self,total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self,batchsize):
        r_n=[]
        if(self.index+batchsize>self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index+batchsize)-self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
            
        else:
            r_n = self.range[self.index:self.index+batchsize]
            self.index = self.index+batchsize
        return r_n



def gen_sample(xmlpath,imgpath,batchsize=1):
    # list xml 
    xmlfiles = []
    files = os.listdir(xmlpath)
    for i in files:
        if(os.path.splitext(i)[1]=='.xml'):
            xmlfiles.append(i)
    rd = random_uniform_num(len(xmlfiles))
    xmlfiles = np.array(xmlfiles)
    
    while 1:
        shuf = xmlfiles[rd.get(1)]
        gtbox,imgfile = readxml(xmlpath + "\\" + shuf[0])
        img = cv2.imread(imgpath+ "\\"+ imgfile)
        h,w,c = img.shape

        #clip image 
        if(np.random.randint(0,100)>50):
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:,2] -1
            newx2 = w - gtbox[:,0] -1
            gtbox[:,0] = newx1
            gtbox[:,2] = newx2

        [cls,regr],_ = cal_rpn((h,w),(int(h/16),int(w/16)),16,gtbox)
        #zero-center by mean pixel 
        m_img = img - IMAGE_MEAN
        m_img = np.expand_dims(m_img,axis=0)

        regr = np.hstack([cls.reshape(cls.shape[0],1),regr])
        
        #
        cls = np.expand_dims(cls,axis=0)
        cls = np.expand_dims(cls,axis=1)
        #regr = np.expand_dims(regr,axis=1)
        regr = np.expand_dims(regr,axis=0)

       # print('imageshape:',m_img.shape,int(m_img.shape[0]/16)*int(m_img.shape[1]/16)*10,'cls shape:',cls.shape,'regr shape:',regr.shape)
        yield m_img,{'rpn_class_reshape':cls,'rpn_regress_reshape':regr}

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def testrpn():

    xmlpath = 'E:\BaiduNetdiskDownload\VOCdevkit\VOC2007\Annotations\img_4375.xml'
    imgpath = 'E:\BaiduNetdiskDownload\VOCdevkit\VOC2007\JPEGImages\img_4375.jpg'
    gtbox,_ = readxml(xmlpath)
    img = cv2.imread(imgpath)
    h,w,c= img.shape
    [cls,regr],base_anchor = cal_rpn((h,w),(int(h/16),int(w/16)),16,gtbox)

    #anchors = base_anchor[cls==1]
    ##anchors = base_anchor
    #print('anchor.shape',anchors.shape)
    #anchors = anchors.astype(int)
    #for i in anchors:
    #    cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(255,0,0),3)
    ##for i in gtbox:
    ##    cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(0,255,0),1)
    #plt.imshow(img)
    
    regr = np.expand_dims(regr,axis=0)
    inv_anchor = bbox_transfor_inv(base_anchor,regr)
    anchors = inv_anchor[cls==1]
    anchors = anchors.astype(int)
    for i in anchors:
        cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(255,0,0),3)
    plt.imshow(img)

#testrpn()


#xmlpath = 'E:\BaiduNetdiskDownload\VOCdevkit\VOC2007\Annotations'
#imgpath = 'E:\BaiduNetdiskDownload\VOCdevkit\VOC2007\JPEGImages'
#gen1 = gen_sample(xmlpath,imgpath,1)

##while(1):

#a = next(gen1)

#xmlpath = 'E:\BaiduNetdiskDownload\VOCdevkit\VOC2007\Annotations'
#imgpath = 'E:\BaiduNetdiskDownload\VOCdevkit\VOC2007\JPEGImages'
#gen1 = gen_sample(xmlpath,imgpath,1)
#next(gen1)
