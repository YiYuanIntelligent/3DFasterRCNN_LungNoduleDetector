import caffe
import numpy as np
import time

class Transpose(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 1,            'requires a single layer.bottom'
        assert bottom[0].data.ndim == 3,    'requires matrix data'
        assert len(top) == 1,               'requires a single layer.top'

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[2], bottom[0].data.shape[1])

    def forward(self, bottom, top):
        #s = time.time()
        top[0].data[...] = np.transpose(bottom[0].data,(0,2,1))
        #print '{} layer forword time: {} s'.format(__file__, (time.time()-s))

    def backward(self, top, propagate_down, bottom):
        #if propagate_down:
        #    print 'diff mean: ', np.mean(top[0].diff)
        # print('top[0] diff {}'.format(top[0].diff))
        print 'trainsform layer top[0] diff mean: {}'.format(np.mean(top[0].diff))
        bottom[0].diff[...] = np.transpose(top[0].diff, (0,2,1))
        #    print 'tranpsoe diff mean: ', np.mean(bottom[0].diff)
        #print 'transpose layer: top shape: {}'.format(top[0].diff.shape)
        #print 'transpose layer: top diff mean: {}'.format(np.mean(top[0].diff))
        #print 'transpose layer bottom[0] diff nonzero: {}'.format(np.count_nonzero(top[0].diff))
        #print 'transpose layer: bottom diff mean: {}'.format(np.mean(bottom[0].diff))
        #pass

