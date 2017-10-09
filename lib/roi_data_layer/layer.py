# --------------------------------------------------------
# Written by HusonChen
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
from utils.timer import Timer
import os
#from multiprocessing import Process, Queue,Value,Lock,Array,JoinableQueue
from utils.custom_multiprocessing import Process, Queue,Value,Lock,Array,JoinableQueue
import psutil
import math
import time


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
            if self._phase == 'TRAIN':
                timer = Timer()
                timer.tic()
                blob = self._blob_queue_train.get()
                self._blob_queue_train.task_done()
                timer.toc()
                #print 'queue get data time used: ', timer.average_time
                return blob
            else:
                blob = self._blob_queue_val.get()
                self._blob_queue_val.task_done()
                return blob
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
            return int(len(self._roidb_train)/(1-cfg.r_rand_crop))
        elif self._phase =='VAL':
            return len(self._roidb)
        else:
            return len(self._u_roidb)

    def cleanup(self):
        if len(self._train_prefetch_process_list) > 0:
            print 'Terminating train BlobFetcher'

            for fetcher in self._train_prefetch_process_list:
                fetcher.terminate()
                fetcher.join()

        if len(self._val_prefetch_process_list) > 0:
            print 'Terminating val BlobFetcher'
            for fetcher in self._val_prefetch_process_list:
                fetcher.terminate()
                fetcher.join()
            # while not self._blob_queue.empty():
            #     self._blob_queue.get()

    def set_phase(self,phase):
        self._phase = phase

    def set_roidb(self, roidb_train,roidb_val,phase):
        """Set the roidb to be used by this layer during training."""
        #print '-------------DEBUG-----------{}'.format('set_roidb')
        self._roidb_train = roidb_train
        self._roidb_val = roidb_val
        self._u_roidb_train = self._load_unique_boxes(roidb_train)
        self._u_roidb_val = self._load_unique_boxes(roidb_val)
        self._phase = phase
        base_len = int(len(self._roidb_train) / (1 - cfg.r_rand_crop))
        if cfg.TRAIN.USE_PREFETCH:
            if len(self._train_prefetch_process_list) == 0 and phase=='TRAIN':
                print 'Starting train BlobFetcher'
                self._blob_queue_train = JoinableQueue(cfg.TRAIN.QUEUE_SIZE)
                self._cur_train = Value('i',0)
                self._lock_train = Lock()
                perm_train = np.random.permutation(np.arange(base_len)).tolist()
                perm_train = Array('i', perm_train)
                for i in range(cfg.TRAIN.PREFETCH_NUM):
                    fetcher = BlobFetcher(self._blob_queue_train,self._roidb_train,self._u_roidb_train,
                                          self._num_classes,"TRAIN",perm_train,self._cur_train,self._lock_train, is_shuffle=True)
                    self._train_prefetch_process_list.append(fetcher)
                    fetcher.start()
                    p = psutil.Process(fetcher.pid)
                    p.cpu_affinity([cfg.TRAIN.CORE_USED_FOR_CAFFE + i + 4])

            if len(self._val_prefetch_process_list) == 0 and phase=='VAL':
                print 'Starting val BlobFetcher'
                self._blob_queue_val = JoinableQueue(cfg.TRAIN.QUEUE_SIZE)
                self._cur_val = Value('i', 0)
                self._lock_val = Lock()
                perm_val = np.arange(len(self._roidb_val))
                perm_val = Array('i', perm_val)
                for i in range(cfg.TRAIN.PREFETCH_NUM):
                    fetcher = BlobFetcher(self._blob_queue_val, self._roidb_val, self._u_roidb_val,
                                          self._num_classes, "VAL", perm_val, self._cur_val, self._lock_val, is_shuffle=False)
                    self._val_prefetch_process_list.append(fetcher)
                    fetcher.start()
                    p = psutil.Process(fetcher.pid)
                    p.cpu_affinity([cfg.TRAIN.CORE_USED_FOR_CAFFE + i + 4])
        # Terminate the child process when the parent exists
        import atexit
        atexit.register(self.cleanup)
        return base_len,len(self._roidb_val)


    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']
        self._phase = layer_params['phase']


        self._name_to_top_map = {}
        self._train_prefetch_process_list = []
        self._val_prefetch_process_list = []
        # data blob: holds a batch of N images, each with 1 channels

        top[0].reshape(cfg.TRAIN.IMS_PER_BATCH,1,cfg.crop_size[0],cfg.crop_size[1],cfg.crop_size[2])
        self._name_to_top_map['data'] = 0

        #top[1].reshape(cfg.TRAIN.IMS_PER_BATCH, 24,24,24,3,5)
        top[1].reshape(cfg.TRAIN.IMS_PER_BATCH, cfg.crop_size[0]/cfg.stride,cfg.crop_size[1]/cfg.stride,cfg.crop_size[2]/cfg.stride,3,5)
        self._name_to_top_map['target'] = 1

        # top[2].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,32,32,32)
        top[2].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, cfg.crop_size[0]/cfg.stride,cfg.crop_size[1]/cfg.stride,cfg.crop_size[2]/cfg.stride)
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
            #print 'after reshape, top data: top[{}]: {}'.format(top_ind, top[top_ind].data.shape)
            #print 'data[{}] mean: {}'.format(top_ind, np.mean(top[top_ind].data))

        #print 'data layer: forworded data layer'

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    # one tianchi image has one record (600 record)
    def _load_unique_boxes(self,_roidb):
        seen = set()
        seen_add = seen.add
        return [x for x in _roidb if not (x['image'] in seen or seen_add(x['image']))]

class Cur_Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self,num):
        with self.lock:
            self.val.value += num

    def value(self):
        with self.lock:
            return self.val.value

    def set(self,val):
        with self.lock:
            self.val.value = val

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, _u_roidb,num_classes,phase,perm,cur,lock,is_shuffle=True):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._phase = phase
        self._u_roidb = _u_roidb
        # fix the random seed for reproducibility
        self._perm = perm
        #print '_perm: ', self._perm
        self._cur = cur
        self._lock = lock
        self._is_shuffle = is_shuffle


    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        with self._lock:
            # TODO(rbg): remove duplicated code
            if self._cur.value + cfg.TRAIN.IMS_PER_BATCH > self.database_len():
                more = cfg.TRAIN.IMS_PER_BATCH - (self.database_len() - self._cur.value)
                db_inds = self._perm[self._cur.value:] + \
                          self._perm[0 : more]
                self._cur.value = more
                if self._is_shuffle:
                    perm = np.random.permutation(np.arange(self.database_len()))
                    for i in range(self.database_len()):
                        self._perm[i] = perm[i]
            else:
                db_inds = self._perm[self._cur.value:self._cur.value + cfg.TRAIN.IMS_PER_BATCH]
                self._cur.value += cfg.TRAIN.IMS_PER_BATCH
            return db_inds

    def _get_next_ind(self):
        with self._lock:
            if self._cur.value  >= self.database_len() :
                self._cur.value = 0
            db_ind = self._perm[self._cur.value]
            if self._cur.value == 0:
                if self._is_shuffle:
                    perm = np.random.permutation(np.arange(self.database_len()))
                    for i in range(self.database_len()):
                        self._perm[i] = perm[i]
            self._cur.value += 1
            return db_ind

    def database_len(self):
        return len(self._perm.get_obj())


    def run(self):
        print 'BlobFetcher started'
        while True:
            #print('Current process id: {}'.format(os.getpid()))
            #print('Current affinity: {}'.format(p.cpu_affinity()))

            db_inds = self._get_next_minibatch_inds()
            processed_ims = []
            labels = []
            coords = []
            db_inds = db_inds[0:cfg.TRAIN.IMS_PER_BATCH]
            for idx in db_inds:
                idx = int(idx)
                # print idx
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
                if type(img) == type(None):
                    db_inds.append(self._get_next_ind())
                    continue

                # img = np.ones(img.shape)
                # label = np.ones(label.shape)
                # coord = np.ones(coord.shape)
                processed_ims.append(img)
                labels.append(label)
                coords.append(coord)

            blobs = get_minibatch(processed_ims, labels, coords)
            self._queue.put(blobs)
