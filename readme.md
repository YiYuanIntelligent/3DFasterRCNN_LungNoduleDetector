# 数据存放
数据存放在data/tianchi/data下面，Annotations和Images前缀分别代表标注文件和肺3D图。标注文件为_label后缀的npy文件，文件内容是二维数组如下
```
[[ 201.        ,  242.        ,  112.        ,    8.12129222],
[ 231.        ,  390.        ,  132.        ,    4.43397444]]
```
每一行的前三个为z,x,y坐标，第四个数字为大小<br>
肺3D图片是mhd经过预处理之后的npy文件，以_clean为后缀，是1\*x\*y\*z大小的4维数组<br>
ImageSets文件夹是存放数据引用，在后续中会通过指定 imdb 来指定需要训练文件列表。比如存在一个训练数据为LKDS-00001，则Annotations_train下面有LKDS-00001_label.npy，Images文件夹下面有LKDS-00001_clean.npy,ImageSets/train.txt有一行为LKDS-00001

# 训练脚本
不使用pretrained_model<br>
在tools文件夹下面执行
```
python -u train_net.py --solver ../models/tianchi/VGG16/faster_rcnn_end2end/solver.prototxt --imdb_train tianchi_train --imdb_val tianchi_val --iters 70000 --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --rand
```
模型生成在output/faster_rcnn_end2end/tianchi_train下面<br>
使用pretrained_model
```
python -u train_net.py --solver ../models/tianchi/VGG16/faster_rcnn_end2end/solver.prototxt --weights ../output/faster_rcnn_end2end/tianchi_train/tianchi_faster_rcnn_epoch_1083.caffemodel --imdb_train tianchi_train --imdb_val tianchi_val --iters 70000 --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --rand
```
# 验证模型的脚本
遍历所有模型，计算在验证集的tnr和tpr。
```
python -u val_net.py --solver ../models/tianchi/VGG16/faster_rcnn_end2end/solver_val.prototxt --imdb_train tianchi_train --imdb_val tianchi_val --iters 70000 --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --rand
```
# 对测试集detect
```
python -u test_net.py --def ../models/tianchi/VGG16/faster_rcnn_end2end/test.prototxt --net ../output/faster_rcnn_end2end/tianchi_faster_rcnn_iter_2204.caffemodel --imdb tianchi_test --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --max_per_image 1
```
