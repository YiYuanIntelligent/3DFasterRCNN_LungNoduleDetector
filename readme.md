# Lung CT nodule detector based on 3D faster RCNN <br>
&emsp;&emsp;This is an intel-extended caffe based 3D faster RCNN RPN training framework, which we believe is the first training framework that makes 3D faster RCNN RPN with 150-layer Deep Convolutional Network converged in CT images.  

&emsp;&emsp;The model has achieved good performance on Alibaba TIANCHI Healthcare AI Competition data (medical imaging prediction of lung nodule). You are welcome to modify it to GPU version.  

&emsp;&emsp;This open-source project is developed by Shenzhen Yiyuan Intelligence Tech Co., LTD and Hong Kong Baptist University (HKBU) GPU High Performance Computing Laboratory.

# NetWork
The 3D RPN network : models/tianchi/VGG16/faster_rcnn_end2end/train.prototxt<br>
Input data layer: lib/roi_data_layer/layer.py 
# Data Format
The training data stored in directory :data/tianchi/data 
```
[[ 201.        ,  242.        ,  112.        ,    8.12129222],
[ 231.        ,  390.        ,  132.        ,    4.43397444]]
...
```
The first three of each line are z, x, y coordinates, and the fourth number is the nodule size(mm) <br>

# Traininng  Script
```
python -u train_net.py --solver ../models/tianchi/VGG16/faster_rcnn_end2end/solver.prototxt --imdb_train tianchi_train --imdb_val tianchi_val --iters 70000 --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --rand
```
the model output in directory :output/faster_rcnn_end2end/tianchi_train<br>


# Traverse the model, calculate the tnr and tpr in the validation set
```
python -u val_net.py --solver ../models/tianchi/VGG16/faster_rcnn_end2end/solver_val.prototxt --imdb_train tianchi_train --imdb_val tianchi_val --iters 70000 --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --rand
```
# Detect nodule in Test set 
```
python -u test_net.py --def ../models/tianchi/VGG16/faster_rcnn_end2end/test.prototxt --net ../output/faster_rcnn_end2end/tianchi_faster_rcnn_iter_2204.caffemodel --imdb tianchi_test --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --max_per_image 1
```
# Contributors
Yu Wu : YiYuan Intelligent co-founder <br>
Shaohuai Shi : Hong Kong Baptist University, Phd<br>
Xiaochen Chen : Hong Kong University of Science and Technology, Master
