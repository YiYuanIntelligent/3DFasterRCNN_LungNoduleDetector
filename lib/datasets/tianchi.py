import os
import errno
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from tianchi_eval import tianchi_eval
from fast_rcnn.config import cfg

class tianchi(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'tianchi_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'nodule')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['_clean.npy']
        self._image_index = self._load_image_set_index()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}


        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def label_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return os.path.join(self._data_path, 'Annotations_'+self._image_set, i + '_label.npy')

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images_'+self._image_set,
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
	return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file. enhance database
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index_temp = [x.strip() for x in f.readlines()]
            # small nodule enhansment
            labels = []
            for idx in image_index_temp:
                l = np.load(self.label_path_at(idx))
                if self._image_set != 'test':
                    if np.all(l == 0):
                        l = np.array([])
                labels.append(l)

            self.sample_bboxes = labels
            bboxes = []
            # print labels
            bboxes = []
            if self._image_set == 'train':
                for i, l in enumerate(labels):
                    if len(l) > 0:
                        # some lung has more than 1 label, enhance each label
                        for t in l:
                            if t[3] < cfg.sizelim3:
                                bboxes.append([np.concatenate([[i],t])])
                            if t[3] < cfg.sizelim2:
                                bboxes += [[np.concatenate([[i],t])]]*1
                            if t[3] < cfg.sizelim:
                                bboxes += [[np.concatenate([[i],t])]]*2
            if self._image_set in ['val','test'] :

                for i, l in enumerate(labels):
                    if len(l) > 0:
                        # some lung has more than 1 label, enhance each label
                        for t in l:
                            bboxes.append([np.concatenate([[i],t])])

            bboxes = np.concatenate(bboxes, axis=0)
            self.bboxes = bboxes

            if self._image_set == 'train':
                print 'check training set size', len(image_index_temp), len(labels), np.array(self.bboxes).shape
            if self._image_set == 'val':
                print 'check val set size', len(image_index_temp), len(labels), np.array(self.bboxes).shape

        return image_index_temp

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        #     return roidb

        gt_roidb = []

        #mark current roi_num, because each image has one box. so we should know which one.
        for i in range(len(self.bboxes)):
            gt_roidb.append(self._load_tianchi_annotation(self.image_index[int(self.bboxes[i][0])],i))

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        #roidb = self._load_rpn_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_tianchi_annotation(self, index, num):
        """
        Load image and bounding boxes info from txt files of tianchi.
        """
        filename = os.path.join(self._data_path, 'Annotations_'+self._image_set, index + '_label.npy')
        # print 'Loading: {}'.format(filename)
        objs = np.load(filename)

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4))
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # "Seg" area here is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for i in range(objs.shape[0]):
            # Make pixel indexes 0-based
            z = float(objs[i,0])
            x = float(objs[i,1])
            y = float(objs[i,2])
            size = float(objs[i, 3])
            cls = self._class_to_ind['nodule']
            boxes[i, :] = [z,x,y,size]
            gt_classes[i] = cls
            overlaps[i, cls] = 1.0
            seg_areas[i] = size

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'box' : self.bboxes[num],
                'img_boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _write_inria_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = self._get_inria_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_inria_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_inria_results_file_template().format(cls)
                os.remove(filename)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_inria_results_file_template(self):
        # INRIAdevkit/results/comp4-44503_det_test_{%s}.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        try:
            os.mkdir(self._devkit_path + '/results')
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise e
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._data_path,
            'Annotations',
            '{:s}.txt')
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_inria_results_file_template().format(cls)
            rec, prec, ap = tianchi_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _get_default_path(self):
        """
        Return the default path where tianchi is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'tianchi' )

