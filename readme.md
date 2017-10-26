# Lung Nodule Detection in CT Images Using 3D faster RCNN #
This is an intel-extended caffe based 3D faster RCNN RPN training framework, which we believe is the first training framework that makes 3D faster RCNN RPN with 150-layer Deep Convolutional Network converged in CT images.

The model has achieved good performance on [Alibaba Tianchi Healthcare AI Competition](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100068.5678.1.1722b13em5oGst&raceId=231601) data (medical imaging prediction of lung nodule). You are welcome to modify it to GPU version.

This open-source project is developed by Shenzhen Yiyuan Intelligence Tech Co., LTD and Hong Kong Baptist University (HKBU) GPU High Performance Computing Laboratory.

Authors
------------
- WU Yu
  - Co-founder of Shenzhen Yi-Yuan Intelligence Tech Co., LTD. Formerly with best ad’s pCTR team of Tencent.
  - Responsible for: model training, nodule-detecting framework constructing and overall optimizing strategy planning.
- SHI Shaohuai
  - Hong Kong Baptist University, GPU High Performance Computing Laboratory, PhD student. Series Winner of NVIDIA National CUDA Contest (Highest Contribution Prize and First Prize).
  - Responsible for: model optimization (which has made significant development for several times) and model-selecting strategy planning.
- CHEN Xiaochen
  - Hong Kong University of Science and Technology (HKUST), Master Degree.
  - Major Caffe code contributor of this competition.

Dependencies
------------
## Software ##
- [Intel Extended Caffe](https://github.com/extendedcaffe/extended-caffe)

It is a highly optimized deep learning framework running on Intel CPUs like Intel Xeon Phi, and it supports many 3D layers (e.g., 3D convolution/deconvolution, 3D batch normalization, etc.). Please follow the installation instruction of extended-caffe, and make sure that ``MKLML engine`` and ``pycaffe`` have been installed successfully. Please be noted this is a not GPU-based platform, any PRs which help migrate this project to [BVLC/Caffe](https://github.com/BVLC/caffe) are welcome.

- Python Dependencies

All the python dependencies are list in ``requirements.txt``, Please make the packages have been installed in your python environment, or you can just run the following command in the root directory of the project to install all the dependencies:

```[sudo] pip install -r requirements.txt```

## Hardware ##
Maybe you need a good server with Intel CPUs to support you to run the training smoothly. Both Intel(R) Xeon(R) E serials and Intel Xeon Phi are fine. We have tested the speed are about 4 samples/s and 10 samples/s on a Xeon E2630 Dual and a Xeon Phi 7250F respectively. If you are running on a desktop PC, it could be very slow when training.

Data preparation
----------------
### Original Dataset ###
The original lung CT images can be found on the [Tianchi website](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100068.5678.2.142fc24cdJGXkU&raceId=231601) and there are detailed introduction about the dataset, so we don't repeat the information here.  

### Preprocessing ###
After you download the original dataset, it should be preprocessed to achieve higher accuracy. The preprocess method we use is not complicated. Since there are lots of usefulness information out of a lung for nodule detection, the main trick we do here is that we just segment the lung from the original image in every CT slide. The comparison of CT slides between before (left) and after (right) lung segmentation is shown as follows:

![Original](https://github.com/YiYuanIntelligent/3DFasterRCNN_LungNoduleDetector/blob/master/original_slice.jpg)
![Processed](https://github.com/YiYuanIntelligent/3DFasterRCNN_LungNoduleDetector/blob/master/preprocessed_slice.jpg)

Our　preprocessing codes refers to: <br>
grt123's code　https://github.com/lfz/DSB2017<br>
kaggle tutorail https://github.com/booz-allen-hamilton/DSB3Tutorial<br>

##### Lung Segmentation ####
```
Train Set：python lungSeg.py trainPath train_result_path
Val Set：python lungSeg.py valPath val_result_path
Test Set：python lungSeg.py testPath test_result_path
```
#### Create label file ####
```
Train Set：python make_label.py train
Val Set：python make_label.py val
Test Set：python make_label.py test
```
Every patient may have 200-500 CT slides. All the slides are needed to do lung segmentation. The preprocessed CT slides of a patient are saved in one ``$patientID_clean.npy`` file, and its ground truth is saved as ``$patientID_label.npy``. The format of ``npy`` is a ``numpy`` array which is easy to read when training. The following example shows what the ``npy`` files look like:

```
>>> import numpy as np
>>> clean = np.load('LKDS-00162_clean.npy')
>>> label = np.load('LKDS-00162_label.npy')
>>> clean.shape
(1, 371, 512, 512)
>>> label.shape
(1, 4)
>>> label
array([[ 113.        ,  334.        ,  162.        ,   23.95100386]])
```

### Get data ready for training #
After all the CT slides have been prepreprossed, put all the training samples in ``*.npy`` to the directory: ``./data/tianchi/data/Images_train/`` and the validation samples to the directory: ``./data/tianchi/data/Images_val/``.

Modify the file ``./data/tianchi/data/ImageSets/train.txt`` to specify the real training samples, our framework will read this patient list from the file, based on which to fetch the real ``*.npy`` file. It is the same for validation data, and you should specify the samples in ``./data/tianchi/data/ImageSets/val.txt``.

Note: The edge of the lung is obvious, so it is not hard to segment the lung. But the method we use in suboptimal, there exist some cases that half of the lung may be eliminated by the algorithm. For these cases, we have a manually setting of the threshold to make the segmentation correct.

Our model
-------------------------
Inspired by some state-of-the-art frameworks in the areas of object detection, image segmentation and image classification with deep learning techniques. We combine Faster-RCNN, UNet, and ResBlock from RetNet to design our deep model (PS: we have not a name for it yet, maybe latter). The architecture of the model is shown as follows:

![](https://github.com/YiYuanIntelligent/3DFasterRCNN_LungNoduleDetector/blob/master/model.png)

You can also have the details of the model via: ``models/tianchi/unet_resnet150/faster_rcnn_end2end/train.prototxt`` or (Visuable Model)[http://ethereon.github.io/netscope/#/gist/79d90c41d3a4389dc1adbf101b2a1f02].

Usage
-----
- Training

To train the detector, run:

```
$cd tools
$python -u train_net.py --solver ../models/tianchi/unet_resnet150/faster_rcnn_end2end/solver.prototxt --imdb_train tianchi_train --imdb_val tianchi_val --iters 70000 --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --rand
```

The trained models will be saved every epoch in the directory: ``./output/faster_rcnn_end2end/tianchi_train``.

- Model Selection

Due to the training speed of the model, we separate the validation as an independent part. Based on the generated models above, we can traverse all the models to do validation with validation set, and the values of TPR (true positive rate) and TNR (true negative rate) are recorded in the log file or the console. In general, the model with the highest TPR and TNR are chosen as the best model to do testing on the test dataset. To traverse the models, run:

```
$cd tools
$python -u val_net.py --solver ../models/tianchi/unet_resnet150/faster_rcnn_end2end/solver_val.prototxt --imdb_train tianchi_train --imdb_val tianchi_val --iters 70000 --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --rand
```

- Detecting with unlabeled data

At the last step, we would like to detect the nudules on the test dataset whose ground truth are unknown. Just run:
```
$cd tools
$python -u test_net.py --def ../models/tianchi/unet_resnet150/faster_rcnn_end2end/test.prototxt --net ../output/faster_rcnn_end2end/tianchi_faster_rcnn_iter_2204.caffemodel --imdb tianchi_test --cfg ../experiments/cfgs/faster_rcnn_end2end.yml --max_per_image 1
```
The detected results are saved in the directory of `output/faster_rcnn_end2end/tianchi_test/${model_name}`

Acknowledgements
----------------
We would like to thank Alibaba to conduct the AI competition with valuable dataset. We also thank Intel technical support team in this Tianchi competition, and their provided high performance computing platforms including Intel Xeon Phi and Extended-Caffe. In addition, many thanks to the authors of [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).


![](https://github.com/YiYuanIntelligent/3DFasterRCNN_LungNoduleDetector/blob/master/WechatIMG3.jpeg)
