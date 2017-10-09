# --------------------------------------------------------
# Written by HusonChen
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
import numpy as np
from  roi_data_layer import split_combine
import os
from get_pbb import get_pbb
from roi_data_layer.minibatch_detector import get_img
import time


def im_detect(net, name,data, target, coord, nzhw,save_dir):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    start_time = time.time()
    target = [np.asarray(t, np.float32) for t in target]
    lbb = target[0]
    # nzhw = nzhw[0]
    shortname = name.split('_clean')[0]
    # data = data[0][0]
    # coord = coord[0][0]
    isfeat = False

    n_per_run = cfg.TRAIN.IMS_PER_BATCH
    splitlist = range(0, len(data) + 1, n_per_run)
    if splitlist[-1] != len(data):
        splitlist.append(len(data))
    outputlist = []
    featurelist = []

    # print splitlist
    for i in range(len(splitlist) - 1):
        input = data[splitlist[i]:splitlist[i + 1]]
        inputcoord = coord[splitlist[i]:splitlist[i + 1]]

        net.blobs['data'].data[...] = input
        net.blobs['coord'].data[...] = inputcoord
        output = net.forward(start='preBlock_0', end='out_reshape2')['out_reshape2'].copy()
        outputlist.append(output)
    output = np.concatenate(outputlist, 0)
    # output = np.load('output2.npy')
    # np.save('output3.npy',output)
    output = split_combine.combine(output, nzhw=nzhw)
    if isfeat:
        feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
        feature = split_combine.combine(feature, sidelen)[..., 0]

    thresh = -3
    pbb, mask = get_pbb(output, thresh, ismask=True)
    if isfeat:
        feature_selected = feature[mask[0], mask[1], mask[2]]
        np.save(os.path.join(save_dir, shortname + '_feature.npy'), feature_selected)
    # tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
    # print([len(tp),len(fp),len(fn)])

    np.save(os.path.join(save_dir, shortname + '_pbb.npy'), pbb)
    np.save(os.path.join(save_dir, shortname + '_lbb.npy'), lbb)


    end_time = time.time()

    print('%s elapsed time is %3.2f seconds' % (shortname,end_time - start_time))


def test_net(net, imdb, max_per_image=8, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb._image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    output_dir = get_output_dir(imdb, net)
    print output_dir

    for i in xrange(num_images):
        # imgs, bboxes, coord2, nzhw,name = queue.get()
        imgs, bboxes, coord2, nzhw = get_img(i, imdb, 'TEST')
        name = imdb._image_index[i]
        im_detect(net,name, imgs, bboxes, coord2, nzhw,output_dir)
