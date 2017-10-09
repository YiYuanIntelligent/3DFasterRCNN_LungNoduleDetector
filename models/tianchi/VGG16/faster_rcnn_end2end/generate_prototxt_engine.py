import json
from collections import OrderedDict

def conv3d(name,output,kernel_size,padding,stride,bottom):
    print 'layer {'
    print '  name: "%s"' %name
    print '  type: \"Convolution\"'
    print '  bottom: "%s"' %bottom
    print '  top: "%s"' %name
    print """  param {
        lr_mult: 1
        decay_mult: 1
      }
      param {
        lr_mult: 2
        decay_mult: 0
      }"""
    print '  convolution_param {'
    print '    num_output: %d' %output
    print '    pad: %d' %padding
    print '    stride: %d' %stride
    print '    kernel_size: %d' %kernel_size
    print '''    weight_filler {
        type: "xavier"
      }'''
    print '  }'
    print '}'
    print ''


def deconv3d(name,output,kernel_size,padding,stride,bottom):
    print 'layer {'
    print '  name: "%s"' %name
    print '  type: \"Deconvolution\"'
    print '  bottom: "%s"' %bottom
    print '  top: "%s"' %name
    print """  param {
        lr_mult: 0
      }"""
    print '  convolution_param {'
    print '    num_output: %d' %output
    print '    pad: %d' %padding
    print '    stride: %d' %stride
    print '    kernel_size: %d' %kernel_size
    print '    bias_term: false'
    print '    weight_filler: { type: \"gaussian\" ' 
    print '                     std: 0.01 }' 
    print '  }'
    print '}'
    print ''

def batch_norm(name,bottom):
    print """layer {
      name: "%s"
      type: "BatchNorm"
      bottom: "%s"
      top: "%s"
      engine: "MKL2017"
}
    """ % (name, bottom, name)

def relu(name,bottom):
    print """layer {
      name: "%s"
      type: "ReLU"
      bottom: "%s"
      top: "%s"
      engine: "MKL2017"
}
    """ % (name, bottom, name)

def pooling(name,bottom,kernel_size,stride,pool='MAX'):
    print """layer {
      name: "%s"
      type: "Pooling"
      bottom: "%s"
      top: "%s"
      pooling_param {
        kernel_size: %d
        stride: %d
        pool: %s
      }
}
    """ % (name, bottom, name,kernel_size,stride,pool)

# def scale()
def eltwise(name,bottom1,bottom2):
    print """layer {
      bottom: "%s"
      bottom: "%s"
      top: "%s"
      name: "%s"
      type: "Eltwise"
}
    """ % (bottom1,bottom2,name,name)


def concate(name,bottom1,bottom2):
    print """layer {
  name: "%s"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  type: "Concat"
  concat_param {
    axis: 1
  }
}
""" % (name,bottom1,bottom2,name)

def drop(name,bottom,ratio):
    print """layer {
  name: "%s"
  type: "Dropout"
  bottom: "%s"
  top: "%s"
  dropout_param {
    dropout_ratio: %f
  }
}
""" % (name, bottom,bottom,ratio)



print '''name: "VGG_ILSVRC_16_layers"
layer {
  name: 'input-data'
  type: 'Python'
  top: 'data'
  top: 'target'
  top: 'coord'
  python_param {
    module: 'roi_data_layer.layer'
    layer: 'RoIDataLayer'
    param_str:  "{'num_classes': 2,'phase':'TRAIN'}"
  }
}'''

conv3d('preBlock_0',16,3,1,1,'data')
batch_norm('preBlock_1','preBlock_0')
relu('preBlock_relu1','preBlock_1')
conv3d('preBlock_2',16,3,1,1,'preBlock_relu1')
batch_norm('preBlock_3','preBlock_2')
relu('preBlock_relu2','preBlock_3')

def PostRes(name,bottom,n_in, n_out, stride = 1):
    short = False
    if stride != 1 or n_out != n_in:
        conv3d(name+'shortcut_conv',n_out,1,0,stride,bottom)
        batch_norm(name+'shortcut_bn',name+'shortcut_conv')
        short = True
        # bottom = name+'shortcut_bn'
    conv3d(name+'conv1',n_out,3,1,stride,bottom)
    batch_norm(name+'bn1',name+'conv1')
    relu(name+'relu1',name+'bn1')
    conv3d(name+'conv2',n_out,3,1,1,name+'relu1')
    batch_norm(name+'bn2',name+'conv2')
    if short:
        eltwise(name+'res_elt',name+'bn2',name+'shortcut_bn')
    else:
        eltwise(name + 'res_elt', name + 'bn2', bottom)
    relu(name + 'res', name + 'res_elt')


num_blocks_forw = [2,2,3,3]
num_blocks_back = [3,3]
featureNum_forw = [24,32,64,64,64]
featureNum_back =    [128,64,64]
anchors = [ 10.0, 30.0, 60.]

def back(i,bottom):
    for j in range(num_blocks_back[i]):
        if j == 0:
            if i == 0:
                addition = 3
            else:
                addition = 0
            PostRes('back_' + str(i + 2) + '_' + str(j+1)+'_', bottom, featureNum_back[i + 1] + featureNum_forw[i + 2] + addition, featureNum_back[i])
        else:
            PostRes('back_' + str(i + 2) + '_' + str(j+1)+'_', 'back_' + str(i + 2) +'_' +str(j)+'_res', featureNum_back[i], featureNum_back[i])


#	self.featureNum_back =    [64,64,64]
for i in range(len(num_blocks_forw)):
    if i == 0:
        pooling('pool_' + str(i ),"preBlock_relu2",2,2)
    else:
        pooling('pool_' + str(i ), 'forw_' + str(i) + '_' + str(j + 1) + '_res', 2, 2)
    for j in range(num_blocks_forw[i]):
        if j == 0:
            PostRes('forw_' + str(i + 1)+'_'+str(j+1)+'_','pool_'+str(i),featureNum_forw[i], featureNum_forw[i+1])
        else:
            PostRes('forw_' + str(i + 1)+'_'+str(j+1)+'_','forw_' + str(i + 1)+'_'+str(j)+'_res',featureNum_forw[i + 1], featureNum_forw[i+1])

print '#=====deconvolution part=================================='

deconv3d('path1_deconv',64,2,0,2,'forw_4_3_res')
batch_norm('path1_bn','path1_deconv')
relu('path1_relu','path1_bn')

concate('cat1','path1_relu','forw_3_3_res')
back(1,'cat1')

deconv3d('path2_deconv',64,2,0,2,'back_3_3_res')
batch_norm('path2_bn','path2_deconv')
relu('path2_relu','path2_bn')

concate('cat2','path2_relu','forw_2_2_res')
concate('cat3','cat2','coord')
back(0,'cat3')

drop('drop_1','back_2_3_res',0.5)
conv3d('output_conv1',64,1,0,1,'back_2_3_res')
relu('output_relu','output_conv1')
conv3d('output_conv2',5 * len(anchors),1,0,1,'output_relu')


print '#=====reshape'

print '''layer {
  name: "out_reshape1"
  type: "Reshape"
  bottom: "output_conv2"
  top: "out_reshape1"
  reshape_param {
    shape {
      dim: 0  # copy the dimension from below
      dim: 0
      dim: -1 # infer it from the other dimensions
    }
  }
}
'''

print '''layer {
  name: 'out_transpose1'
  type: 'Python'
  bottom: 'out_reshape1'
  top: 'out_transpose1'
  python_param {
    module: 'transform.transpose_layer'
    layer: 'Transpose'
  }
}
'''

print '''layer {
  name: "out_reshape2"
  type: "Reshape"
  bottom: "out_transpose1"
  top: "out_reshape2"
  reshape_param {
    shape {
      dim: 0  # copy the dimension from below
      dim: 24
      dim: 24 
      dim: 24 
      dim: %d
      dim: 5 # infer it from the other dimensions
    }
  }
}
'''% len(anchors)


print '''

#==== ROI layer
layer {
  name: "rpn-data"
  type: "Python"
  bottom: "out_reshape2"
  bottom: "target"
  top: "cls_pos_output"
  top: "cls_pos_labels"
  top: "cls_neg_output"
  top: "cls_neg_labels"
  top: "rpn_bbox_pred"
  top: "rpn_bbox_targets"
  python_param {
    module: 'rpn.tianchi_anchor_target_layer'
    layer: 'TianchiAnchorTargetLayer'
    param_str: "'num_hard': 2"
  }
}

#========not completed loss ====

layer {
  name: "rpn_pos_loss_cls"
  type: "SigmoidCrossEntropyLoss"
  bottom: "cls_pos_output"
  bottom: "cls_pos_labels"
  propagate_down: 1
  propagate_down: 0
  top: "rpn_pos_loss_cls"
  loss_weight: 0.5 
  loss_param {
    normalize: 1
  }
}

layer {
  name: "rpn_neg_loss_cls"
  type: "SigmoidCrossEntropyLoss"
  bottom: "cls_neg_output"
  bottom: "cls_neg_labels"
  propagate_down: 1
  propagate_down: 0
  top: "rpn_neg_loss_cls"
  loss_weight: 0.5 
  loss_param {
    normalize: 1
  }
}

layer {
  name: "rpn_loss_bbox"
  type: "SmoothL1Loss"
  bottom: "rpn_bbox_pred"
  bottom: "rpn_bbox_targets"
  top: "rpn_loss_bbox"
  loss_weight: 1
  smooth_l1_loss_param { sigma: 1.0 }
}
'''
