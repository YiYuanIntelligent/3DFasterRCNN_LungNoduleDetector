# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""
import google.protobuf.text_format
import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import math
from scipy.special import expit

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb_train,roidb_val, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.roidb_train = roidb_train
        self.roidb_val = roidb_val

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb_train)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
            self.start_epoch = int(pretrained_model.split('/')[0].split('.'))

        self.solver_param = caffe_pb2.SolverParameter()

        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)


    def snapshot(self,it):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_epoch_{:d}'.format(it) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        train_len,val_len = self.solver.net.layers[0].set_roidb(self.roidb_train, self.roidb_val)
        #print 'data len: ', self.solver.net.layers[0].database_len()
        #print 'num of total samples per epoch: ', self.solver.net.layers[0].database_len()
        iter_per_epoch = int(math.ceil(float(train_len) / cfg.TRAIN.IMS_PER_BATCH))
        print 'total train bbox is %d, database length is %d, train iter per epoch is %d' % \
              (len(self.roidb_train),train_len,iter_per_epoch)
        val_per_epoch = int(math.ceil(float(val_len) / cfg.TRAIN.IMS_PER_BATCH))
        print 'total val bbox is %d, database length is %d,val iter per epoch is %d' % \
              (len(self.roidb_val), val_len, val_per_epoch)
        net_outputs = []
        metrics = []
        epoch = 0
        for it in range(max_iters):
            # Make one SGD update
            if cfg.TRAIN.RUN:
                if it > 0:
                    timer.tic()
                self.solver.step(1)
                #self.solver.net.forward(end='preBlock_0')
                net_outputs.append([
                    self.solver.net.blobs['rpn_loss_bbox'].data,
                    self.solver.net.blobs['rpn_neg_loss_cls'].data,
                    self.solver.net.blobs['rpn_pos_loss_cls'].data
                ])
                #print 'net_outputs: ', net_outputs
                metrics.append(get_loss_by_rnp(self.solver.net.blobs))
                if it>0:
                    timer.toc()
                    print 'iter timeused: {}'.format(timer.average_time)
                if it % (10 * self.solver_param.display) == 0:
                    print 'speed: {:.3f}s / iter'.format(timer.average_time)

                model_paths.append(self.snapshot(it))

            # one epoch
            if ((it + 1) % iter_per_epoch == 0) or not cfg.TRAIN.RUN:
                if cfg.TRAIN.RUN:
                    epoch += 1
                    print 'epoch %d done' % epoch
                    model_paths.append(self.snapshot(epoch))

                    net_outputs = np.asarray(net_outputs, np.float32)
                    net_outputs[:,1:] *= 0.5
                    metrics = np.asarray(metrics, np.float32)
                    print('Training: tpnr %3.8f, tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
                        100.0 * (np.sum(metrics[:, 0]) + np.sum(metrics[:, 2])) / (
                            np.sum(metrics[:, 1]) + np.sum(metrics[:, 3])),
                        100.0 * np.sum(metrics[:, 0]) / np.sum(metrics[:, 1]),
                        100.0 * np.sum(metrics[:, 2]) / np.sum(metrics[:, 3]),
                        np.sum(metrics[:, 1]),
                        np.sum(metrics[:, 3]),
                        timer.average_time))
                    print('loss %2.4f, regress loss %2.4f, neg loss %2.4f, pos loss %2.4f' % (
                        np.mean(np.sum(net_outputs, axis=1)),
                        np.mean(net_outputs[:, 0]),
                        np.mean(net_outputs[:, 1]),
                        np.mean(net_outputs[:, 2])))

                metrics = []
                net_outputs = []

                if cfg.VAL.RUN:
                    # do validation
                    self.solver.net.layers[0].set_phase('VAL')
                    self.solver.net.layers[-4].set_phase('VAL')

                    timer.tic()
                    for i in range(val_per_epoch):
                        # print 'validation at index: %d' %i
                        self.solver.net.forward()
                        net_outputs.append([
                            self.solver.net.blobs['rpn_loss_bbox'].data,
                            self.solver.net.blobs['rpn_neg_loss_cls'].data,
                            self.solver.net.blobs['rpn_pos_loss_cls'].data
                        ])
                        loss_output = get_loss_by_rnp(self.solver.net.blobs)
                        metrics.append(loss_output)
                    metrics = np.asarray(metrics, np.float32)
                    net_outputs = np.asarray(net_outputs, np.float32)
                    net_outputs[:,1:] *= 0.5
                    timer.toc()
                    print('Validation: tpnr %3.8f, tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
                        100.0 * (np.sum(metrics[:, 0]) + np.sum(metrics[:, 2])) / (
                        np.sum(metrics[:, 1]) + np.sum(metrics[:, 3])),
                        100.0 * np.sum(metrics[:, 0]) / np.sum(metrics[:, 1]),
                        100.0 * np.sum(metrics[:, 2]) / np.sum(metrics[:, 3]),
                        np.sum(metrics[:, 1]),
                        np.sum(metrics[:, 3]),
                        timer.average_time))
                    print('loss %2.4f, regress loss %2.4f, neg loss %2.4f, pos loss %2.4f' % (
                        np.mean(np.sum(net_outputs,axis=1)),
                        np.mean(net_outputs[:, 0]),
                        np.mean(net_outputs[:, 1]),
                        np.mean(net_outputs[:, 2])))

                    if not cfg.TRAIN.RUN:
                        break
                    self.solver.net.layers[-4].set_phase('TRAIN')
                    self.solver.net.layers[0].set_phase('TRAIN')
                    metrics = []
                    net_outputs = []

        # if last_snapshot_iter != it:
        #     model_paths.append(self.snapshot(it))
        return model_paths

def get_loss_by_rnp(blobs):
    pos_output = blobs['cls_pos_output'].data
    pos_labels = blobs['cls_pos_labels'].data
    neg_output = blobs['cls_neg_output'].data
    neg_labels = blobs['cls_neg_labels'].data

    neg_prob = expit(neg_output)
    pos_total = len(pos_labels)
    #print 'len: ', len(pos_output), len(pos_labels)
    if len(pos_output) > 0:
        pos_prob = expit(pos_output)

        pos_correct = (pos_prob >= 0.5).sum()

    else:
        pos_correct = 0

    neg_correct = (neg_prob < 0.5).sum()
    neg_total = len(neg_labels)
    return [pos_correct, pos_total, neg_correct, neg_total]


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb_train,roidb_val, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    roidb_train = filter_roidb(roidb_train)
    roidb_val = filter_roidb(roidb_val)
    sw = SolverWrapper(solver_prototxt, roidb_train,roidb_val, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
