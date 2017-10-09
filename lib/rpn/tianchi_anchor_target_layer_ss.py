# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
import time
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
# from utils.bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

DEBUG = False

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hard_mining(neg_output, neg_labels, num_hard):
    topk = min(num_hard, len(neg_output))
    idcs = neg_output.argsort()[-topk:][::-1]
    print 'hard mining idcs: ', idcs
    return neg_output[idcs], neg_labels[idcs], idcs


class TianchiAnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        print '===================== Start to setup anchor target layer'
        print 'output bottom shape: ', bottom[0].data.shape
        print 'target bottom shape: ', bottom[1].data.shape
        layer_params = yaml.load(self.param_str)
        self._num_hard  = layer_params.get('num_hard', 2)

        output = bottom[0].data.reshape(1, -1, 5)
        self._phase = 'TRAIN'
        print 'output: ', output.shape 
        top[0].reshape(1, *output.shape)
        top[1].reshape(1, *output.shape)
        top[2].reshape(1, *output.shape)
        top[3].reshape(1, *output.shape)
        top[4].reshape(1, *output.shape)
        top[5].reshape(1, *output.shape)
        self._hard_idcs = []
        self._pos_idcs = []
        self._neg_idcs = []

    def set_phase(self, phase):
        self._phase = phase

    def forward(self, bottom, top):
        #print '==============================='
        #print 'original label shape: ', bottom[1].data.shape
        s = time.time()
        batch_size = bottom[1].data.shape[0]

        output = bottom[0].data.reshape(-1, 5)
        labels = bottom[1].data.reshape(-1, 5)
        print 'last conv output tensor: ', np.mean(output)


        pos_idcs = labels[:, 0] > 0.5
        self._pos_idcs,  = np.where(pos_idcs)
        #print 'num of pos labels: ', labels[pos_idcs].shape[0]
        #print 'pos output: ', output[pos_idcs]
        bbox_pos_output = output[pos_idcs][:, 1:]
        bbox_pos_labels = labels[pos_idcs][:, 1:]
        #print 'predict pos bbox: ', bbox_pos_output
        #print 'gt pos bbox: ', bbox_pos_labels
        #print 'bbox diff: ', bbox_pos_output-bbox_pos_labels

        cls_pos_output = output[pos_idcs][:, 0]
        cls_pos_labels = labels[pos_idcs][:, 0]

        #print 'predict pos num: ', cls_pos_output.shape
        #print 'gt pos num: ', cls_pos_labels.shape

        neg_idcs = labels[:, 0] < -0.5
        cls_neg_output = output[neg_idcs][:, 0]
        cls_neg_labels = labels[neg_idcs][:, 0]+1.0
        #print 'predict neg num: ', cls_neg_output.shape
        #print 'gt neg num: ', cls_neg_labels.shape
        #print 'before hard neg, neg labels: ', cls_neg_labels
        #print 'before hard neg, neg output: ', cls_neg_output

        if self._num_hard > 0 and self._phase == 'TRAIN': # in the phase of TRAIN
            cls_neg_output, cls_neg_labels, hard_idcs = hard_mining(cls_neg_output, cls_neg_labels, self._num_hard * batch_size)
        #    print 'after hard neg, predict neg num: ', cls_neg_output.shape
        #    print 'after hard neg, gt neg num: ', cls_neg_labels.shape

        #    print 'after hard neg, neg labels: ', cls_neg_labels
        #    print 'after hard neg, neg output: ', sigmoid(cls_neg_output)
        self._neg_idcs = hard_idcs

        #print 'bbox pos sigmoid: ', sigmoid(bbox_pos_output)
        #print 'pos sigmoid: ', sigmoid(cls_pos_output)
        #print 'cls pos labels: ', cls_pos_labels 
        #print 'cls pos output: ', cls_pos_output
        print 'neg sigmoid: ', sigmoid(cls_neg_output)
        #print 'cls neg labels: ', cls_neg_labels

        top[0].reshape(*cls_pos_output.shape)
        top[1].reshape(*cls_pos_labels.shape)
        top[2].reshape(*cls_neg_output.shape)
        top[3].reshape(*cls_neg_labels.shape)
        top[4].reshape(*bbox_pos_output.shape)
        top[5].reshape(*bbox_pos_labels.shape)
        #for i in range(6):
        #    print 'top shape: top[{}]: {}'.format(i, top[i].data.shape)
        top[0].data[...] = cls_pos_output 
        top[1].data[...] = cls_pos_labels 

        #print 'cls_pos_output [top]: ', top[0].data 
        #print 'cls_pos_labels [top]: ', top[1].data 


        top[2].data[...] = cls_neg_output
        top[3].data[...] = cls_neg_labels
        top[4].data[...] = bbox_pos_output 
        top[5].data[...] = bbox_pos_labels

        activate_pos = []
        top0 = []
        for i in cls_pos_output:
            top0.append((np.where(bottom[0].data==i)))
        activate_pos.append(top0)
        top2 = []
        for i in cls_neg_output:
            top2.append(np.where(bottom[0].data == i))
        activate_pos.append(top2)
        top4 = []
        for i in bbox_pos_output:
            top4_box = []
            for j in i:
                top4_box.append(np.where(bottom[0].data == j))
            top4.append(top4_box)
        activate_pos.append(top4)
        self.activate_pos = activate_pos
        #print '{} layer forword time: {} s'.format(__file__, (time.time()-s))

        #print 'cls_pos_output shape: ', cls_pos_output.shape
        #print 'cls_pos_labels shape: ', cls_pos_labels.shape

    def backward(self, top, propagate_down, bottom):


        """This layer does not propagate gradients."""
        #print('blob_id: {}'.format(self.blob_id))
        #print('pos_idcs: {}'.format(self._pos_idcs))
        #print('neg_idcs: {}'.format(self._neg_idcs))
        #for i in range(6):
        #    #print('bottom[{}] diff shape: {}'.format(i, bottom[i].diff.shape)) 
        #    print('top[{}] diff shape: {}'.format(i, top[i].diff.shape)) 
        #    #if not propagate_down[i]:
        #    #    continue
#bottom[#i].diff[...] = top[i].diff 
        print('top[0] diff {}'.format(top[0].diff))
        print('top[2] diff {}'.format(top[2].diff))
        print('top[4] diff {}'.format(top[4].diff))
        # pos_diff = np.insert(top[4].diff, 0, top[0].diff, axis=1)

        # a = bottom[0].diff
        # bottom[0].diff.reshape(-1, 5)[self._pos_idcs] = pos_diff
        # bottom[0].diff.reshape(-1, 5)[self._neg_idcs][:, 0] = top[2].diff
        # b = bottom[0].diff

        bottom[0].diff[...] = np.zeros(bottom[0].diff.shape)
        activate_pos = self.activate_pos
        for i in range(len(activate_pos[0])):
            bottom[0].diff[activate_pos[0][i]] -= 0.001 * top[0].diff[i]
        for i in range(len(activate_pos[1])):
            bottom[0].diff[activate_pos[1][i]] -= 0.001 * top[2].diff[i]
        for i in range(len(activate_pos[2])):
            for j in range(len(activate_pos[2][i])):
                bottom[0].diff[i][j] -=  0.001 * top[4].diff[i][j]
        # bottom[0].diff[]
        # bottom[0].diff[...] = top[2].diff.reshape(-1, 5)[self._neg_idcs][:, 0]

        print 'rpn layer bottom[0] diff mean: {}'.format(np.mean(bottom[0].diff))
        print 'rpn layer bottom[0] diff nonzero: {}'.format(np.count_nonzero(bottom[0].diff))

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


