# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer import minibatch_detector
from roi_data_layer import minibatch
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue


class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(self.database_len()))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= self.database_len():#len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            processed_ims = []
            labels = []
            coords = []
            for idx in db_inds:
                idx = int(idx)
                try:
                    if self._phase == 'TEST':
                        img, label, coord = minibatch_detector.get_img(idx,self._roidb, self._u_roidb, self._phase)
                    else:
                        img, label, coord = minibatch.get_img(idx, self._roidb, self._u_roidb, self._phase)
                except:
                    print 'wrong idx:%d' % idx
                    import traceback
                    traceback.print_exc()
                    db_inds.append(self._get_next_ind())
                    continue
                processed_ims.append(img)
                labels.append(label)
                coords.append(coord)
            return get_minibatch(processed_ims,labels,coords)

    def _get_next_ind(self):
        if self._cur + 1 >= self.database_len():
            self._shuffle_roidb_inds()

        db_ind = self._perm[self._cur]
        self._cur += 1
        return db_ind

    def database_len(self):
        if self._phase == 'TRAIN':
            return len(self._roidb)/(1-cfg.r_rand_crop)
        elif self._phase =='VAL':
            return len(self._roidb)
        else:
            return len(self._u_roidb)

    def cleanup(self):
        if len(self._prefetch_process_list) > 0:
            print 'Terminating BlobFetcher'

        for fetcher in self._prefetch_process_list:
            fetcher.terminate()
            fetcher.join()
        self._blob_queue = None

    def set_roidb(self, roidb,phase):
        """Set the roidb to be used by this layer during training."""
        self.cleanup()
        self._phase = phase
        self._roidb = roidb
        self._shuffle_roidb_inds()
        self._u_roidb = self._load_unique_boxes()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(8)

            for i in range(cfg.TRAIN.PREFETCH_NUM):
                print 'Starting BlobFetcher'
                fetcher = BlobFetcher(self._blob_queue,self._roidb,self._u_roidb,self._num_classes,self._phase)
                self._prefetch_process_list.append(fetcher)
                fetcher.start()
            # Terminate the child process when the parent exists
            import atexit
            atexit.register(self.cleanup)


    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']
        self._phase = layer_params['phase']


        self._name_to_top_map = {}
        self._prefetch_process_list = []
        
        # data blob: holds a batch of N images, each with 1 channels
        top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 1, cfg.crop_size_detector[0], cfg.crop_size_detector[1], cfg.crop_size_detector[2])
        self._name_to_top_map['data'] = 0

        # top[1].reshape(cfg.TRAIN.IMS_PER_BATCH, 24,24,24,3,5)
        top[1].reshape(cfg.TRAIN.IMS_PER_BATCH, cfg.crop_size_detector[0] / cfg.stride, cfg.crop_size_detector[1] / cfg.stride,
                       cfg.crop_size_detector[2] / cfg.stride, 3, 5)
        self._name_to_top_map['target'] = 1

        # top[2].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,32,32,32)
        top[2].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, cfg.crop_size_detector[0] / cfg.stride, cfg.crop_size_detector[1] / cfg.stride,
                       cfg.crop_size_detector[2] / cfg.stride)
        self._name_to_top_map['coord'] = 2


        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
            #print 'after reshape, top shape: top[{}]: {}'.format(top_ind, top[top_ind].data.shape)

        print 'forworded data layer'

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    # one tianchi image has one record (600 record)
    def _load_unique_boxes(self):
        seen = set()
        seen_add = seen.add
        return [x for x in self._roidb if not (x['image'] in seen or seen_add(x['image']))]

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, _u_roidb,num_classes,phase):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._phase = phase
        self._u_roidb = _u_roidb
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(self.database_len()))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= self.database_len():
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_ind(self):
        if self._cur + 1 >= self.database_len():
            self._shuffle_roidb_inds()

        db_ind = self._perm[self._cur]
        self._cur += 1
        return db_ind

    def database_len(self):
        if self._phase == 'TRAIN':
            return len(self._roidb)/(1-cfg.r_rand_crop)
        elif self._phase =='VAL':
            return len(self._roidb)
        else:
            return len(self._u_roidb)


    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            processed_ims = []
            labels = []
            coords = []
            for idx in db_inds:
                idx = int(idx)
                try:
                    if self._phase == 'TEST':
                        img, label, coord = minibatch_detector.get_img(idx, self._roidb, self._u_roidb, self._phase)
                    else:
                        img, label, coord = minibatch.get_img(idx, self._roidb, self._u_roidb, self._phase)
                except:
                    print 'wrong idx:%d' % idx
                    import traceback
                    traceback.print_exc()
                    db_inds.append(self._get_next_ind())
                    continue
                processed_ims.append(img)
                labels.append(label)
                coords.append(coord)

            blobs = get_minibatch(processed_ims, labels, coords)
            self._queue.put(blobs)
