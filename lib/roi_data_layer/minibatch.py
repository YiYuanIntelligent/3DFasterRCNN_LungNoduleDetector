# --------------------------------------------------------
# Written by HusonChen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from scipy.ndimage import zoom
import warnings
import random
from scipy.ndimage.interpolation import rotate
import split_combine

def get_minibatch(processed_ims,labels,coords):
    """Given a roidb, construct a minibatch sampled from it."""
    im_blob = im_list_to_blob(processed_ims)
    coords = np.array(coords)
    labels = np.array(labels)
    blobs = {'data': im_blob, 'target': labels, 'coord': coords}
    return blobs

def get_img(idx,_roidb,_u_roidb,phase):
    # select index I want
    isRandomImg = False
    #if phase == 'VAL':
        #print('idx:{}, _roidb:{}'.format(idx, _roidb[idx]['image']))
    if phase != 'TEST':
        if idx >= len(_roidb):
            isRandom = True
            idx = idx % len(_roidb)
            isRandomImg = np.random.randint(2)
        else:
            isRandom = False
    else:
        isRandom = False

    if phase != 'TEST':
        # select image from current roi
        if not isRandomImg:
            roi = _roidb[idx]
            bbox = roi['box']
            filename = roi['image']
            # print('idx:{}, _roidb:{}'.format(idx, filename))
            imgs = np.load(filename)
            bboxes = roi['img_boxes']
            # print bbox,filename,bboxes
            isScale = cfg.augtype['scale'] and (phase == 'TRAIN')
            sample, target, bboxes, coord = crop(imgs, bbox[1:], bboxes, isScale, isRandom)
            if phase == 'TRAIN' and not isRandom:
                sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                        ifflip=cfg.augtype['flip'], ifrotate=cfg.augtype['rotate'],
                                                        ifswap=cfg.augtype['swap'])
        # randomly select image
        else:
            randimid = np.random.randint(len(_u_roidb))
            filename = _u_roidb[randimid]['image']
            imgs = np.load(filename)
            bboxes = _u_roidb[randimid]['img_boxes']
            isScale = cfg.augtype['scale'] and (phase == 'TRAIN')
            sample, target, bboxes, coord = crop(imgs, [], bboxes, isScale=False, isRand=True)
            # print('idx:{}, _roidb:{}'.format(randimid, filename))
            # print sample.shape,target.shape,bboxes.shape,coord.shape
            # print target,bboxes
        try:
            label = label_mapping(phase, sample.shape[1:], target, bboxes)
        except:
            if not isRandomImg:
                print bbox
            print 'filename:%s' % filename
            np.save('sample', sample)
            raise Exception('size error')

        sample = (sample.astype(np.float32) - 128) / 128
        #print label[label>0.5]
        # if filename in self.kagglenames and self.phase=='train':
        #    label[label==-1]=0
        # print label.shape
        # np.save('sample',sample)
        # np.save('label',label)
        # np.save('coord',coord)
        # print filename,sample,label,coord
        return sample, label, coord
    else:
        roi = _roidb[idx]
        imgs = np.load(roi['image'])
        bboxes = roi['img_boxes']
        nz, nh, nw = imgs.shape[1:]
        pz = int(np.ceil(float(nz) / cfg.stride)) * cfg.stride
        ph = int(np.ceil(float(nh) / cfg.stride)) * cfg.stride
        pw = int(np.ceil(float(nw) / cfg.stride)) * cfg.stride
        imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                      constant_values=cfg.pad_value)

        xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / cfg.stride),
                                 np.linspace(-0.5, 0.5, imgs.shape[2] / cfg.stride),
                                 np.linspace(-0.5, 0.5, imgs.shape[3] / cfg.stride), indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
        imgs, nzhw = split_combine.split(imgs)
        coord2, nzhw2 = split_combine.split_comber.split(coord,
                                                side_len=cfg.side_len / cfg.stride,
                                                max_stride=cfg.max_stride / cfg.stride,
                                                margin=cfg.margin / cfg.stride)
        assert np.all(nzhw == nzhw2)
        imgs = (imgs.astype(np.float32) - 128) / 128
        # print imgs,bboxes,coord2,nzhw
        return imgs, bboxes, coord2, np.array(nzhw)


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord



def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        # im = cv2.imread(roidb[i]['image'])
        im = np.load(roidb[i]['image'])
        im = im.reshape(im.shape[1],im.shape[2],im.shape[3],1)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        # im_scales.append(im_scale)
        im_scales.append(1)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()



def crop( imgs, target, bboxes, isScale=False, isRand=False):
    if isScale:
        radiusLim = [8., 120.]
        scaleLim = [0.75, 1.25]
        scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
            , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
        scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
        # print self.crop_size,scale
        crop_size = (np.array(cfg.crop_size).astype('float') / scale).astype('int')
    else:
        crop_size = cfg.crop_size
    bound_size = cfg.bound_size
    target = np.copy(target)
    bboxes = np.copy(bboxes)

    start = []
    for i in range(3):
        if not isRand:
            r = target[3] / 2
            s = np.floor(target[i] - r) + 1 - bound_size
            e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
        else:
            s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
            e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
            target = np.array([np.nan, np.nan, np.nan, np.nan])
        if s > e:
            start.append(np.random.randint(e, s))  # !
        else:
            start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))

    normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
    normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
    xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], cfg.crop_size[0] / cfg.stride),
                             np.linspace(normstart[1], normstart[1] + normsize[1], cfg.crop_size[1] / cfg.stride),
                             np.linspace(normstart[2], normstart[2] + normsize[2], cfg.crop_size[2] / cfg.stride),
                             indexing='ij')
    coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

    pad = []
    pad.append([0, 0])
    for i in range(3):
        leftpad = max(0, -start[i])
        rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
        pad.append([leftpad, rightpad])
    crop = imgs[:,
           max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
           max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
           max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
    crop = np.pad(crop, pad, 'constant', constant_values=cfg.pad_value)
    for i in range(3):
        target[i] = target[i] - start[i]
    for i in range(len(bboxes)):
        for j in range(3):
            bboxes[i][j] = bboxes[i][j] - start[j]

    if isScale:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crop = zoom(crop, [1, scale, scale, scale], order=1)
        newpad = cfg.crop_size[0] - crop.shape[1:][0]
        if newpad < 0:
            crop = crop[:, :-newpad, :-newpad, :-newpad]
        elif newpad > 0:
            pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
            crop = np.pad(crop, pad2, 'constant', constant_values=cfg.pad_value)
        for i in range(4):
            target[i] = target[i] * scale
        for i in range(len(bboxes)):
            for j in range(4):
                bboxes[i][j] = bboxes[i][j] * scale
    return crop, target, bboxes, coord

def label_mapping(phase,input_size, target, bboxes):
    stride = np.array(cfg.stride)
    num_neg = int(cfg.num_neg)
    th_neg = cfg.th_neg
    anchors = np.asarray(cfg.anchors)

    if phase == 'TRAIN':
        th_pos = cfg.th_pos_train
    elif phase == 'VAL':
        th_pos = cfg.th_pos_val

    output_size = []
    for i in range(3):
        #if input_size[i] % stride != 0:
        #    raise Exception("size error!!")
        assert (input_size[i] % stride == 0)
        output_size.append(input_size[i] / stride)

    label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
    offset = ((stride.astype('float')) - 1) / 2
    oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
    oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
    ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

    for bbox in bboxes:
        for i, anchor in enumerate(anchors):
            iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
            label[iz, ih, iw, i, 0] = 0

    if phase == 'TRAIN' and num_neg > 0:
        neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
        neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
        neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
        label[:, :, :, :, 0] = 0
        label[neg_z, neg_h, neg_w, neg_a, 0] = -1

    if np.isnan(target[0]):
        return label
    iz, ih, iw, ia = [], [], [], []
    for i, anchor in enumerate(anchors):
        iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
        iz.append(iiz)
        ih.append(iih)
        iw.append(iiw)
        ia.append(i * np.ones((len(iiz),), np.int64))
    iz = np.concatenate(iz, 0)
    ih = np.concatenate(ih, 0)
    iw = np.concatenate(iw, 0)
    ia = np.concatenate(ia, 0)
    flag = True
    if len(iz) == 0:
        pos = []
        for i in range(3):
            pos.append(max(0, int(np.round((target[i] - offset) / stride))))
        idx = np.argmin(np.abs(np.log(target[3] / anchors)))
        pos.append(idx)
        flag = False
    else:
        idx = random.sample(range(len(iz)), 1)[0]
        pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
    dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
    dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
    dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
    dd = np.log(target[3] / anchors[pos[3]])
    label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
    # np.save('label.npy',label)
    return label


def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]

        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)

        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0

        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))

        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))

        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union

        mask = iou >= th
        # if th > 0.4:
        #   if np.sum(mask) == 0:
        #      print(['iou not large', iou.max()])
        # else:
        #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw
