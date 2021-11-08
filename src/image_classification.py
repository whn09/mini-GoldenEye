import os
import cv2
import time
import numpy as np
#import matplotlib  # only on raspberry pi
#matplotlib.use("Pdf")  # only on raspberry pi
import mxnet as mx
from mxnet.gluon import nn
from mxnet import init
from gluoncv import model_zoo, data, utils
from mxnet.gluon.data.vision import transforms

class ImageClassification():

    def __init__(self):
        self.classes = ['bedroom', 'other', 'toilet', 'kitchen', 'livingroom']
        self.net = model_zoo.get_model('ResNet50_v2', classes=len(self.classes), pretrained=False)
        param_files = ([x for x in os.listdir('.') if x.endswith('.params')])
        selected = param_files[0]
        self.net.load_parameters(selected)
        self.ctx = mx.gpu(0)
        self.net.collect_params().reset_ctx(self.ctx)
        
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def classify_image(self, img):
        x = self.transform_test(mx.nd.array(img)).expand_dims(axis=0)
        scores = self.net(x.as_in_context(self.ctx))
        return scores.asnumpy()


if __name__ == '__main__':
    imageClassification = ImageClassification()
    detect_start = time.time()
    frame = cv2.imread('../images/1.jpg')
    scores = imageClassification.classify_image(frame)
    detect_end = time.time()
    print('scores:', scores)
    print('detect time:', detect_end-detect_start)
