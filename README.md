
# InsightFace: 2D and 3D Face Analysis Project

By Jia Guo and [Jiankang Deng](https://jiankangdeng.github.io/)

## 自制数据集  
这里以vgg-face为例
###下载
首先在vggface官网下载数据集，下载得到的不是打包好的图片文件，而是一些url和标注。使用网上搜索的多线程代码下载数据，其中因为有些url已经失效、有些url需要科学下载，还有一些url含有重定向等，我找到的脚本**把有返回响应的都保存成了jpg**。下载完的数据格式形如  

----PersonA  
-----Img000001.jpg  
-----Img000002.jpg  
----PersonB  
-----Img000001.jpg  
-----Img000002.jpg  

###清洗数据
使用pillow包或者其他工具判断jpg文件的完整性，删除不是jpg的文件

###裁切人脸
下载的vgg-face数据集是整张图加标注信息，其中标注信息是DPM检测的（多个人脸的图不知道是不是人工干预的，最终标注的确是是目标人），根据标注信息对每张图片进行裁切，裁切时注意标注框有非法值（大于图片长宽，或许还有负数）要加以限制，将裁切后的图片输出到某个目录。

###再次清洗
裁切后的图片中也有明显不是人脸的，甚至在检查url时还发现一些不是人像的图片。这里我使用python版本的mtcnn对crop后的人脸进行了检测，如果检测不到人脸就删除图片。（后来发现可以不做，因为后面对齐的代码中也用到了mtcnn，检测不到人脸的会跳过）

###人脸对齐
使用src/align/下的align_dataset_mtcnn文件进行对齐，查看该代码发现是没有使用mtcnn检测的关键点的，所以其实并没有对齐，只是将检测框区域的图片resize成了指定大小。（需要安装tensorflow）  需要修改导入的名字
```python
import facenet.src.facenet as facenet
import facenet.src.align.detect_face as detect_face
```
还尝试过使用align_dataset进行对齐，这里面使用了dlib的68个关键点文件对齐（文件网上可以搜到官网链接），但是对齐效果感觉很多图片都变形太严重，最后没有使用。

###生成lst文件
face2rec2需要有lst文件才能生成rec和index文件。这里我参考了src/data中的dir2lst文件，将print改为写入文件，我的目录中总是出现Thumbs.db（我是windows远程ubuntu），所以我稍微修改了代码跳过了这些文件。

###后续
运行face2rec2，运行train.py，其中的参数怎么写可以看代码中的定义以及使用。运行这些py文件还需要安装facenet，后来逛issue看到其实作者代码中有一部分就是facenet的。。。。总之我是又用pip装了一个facenet。。。。

###

## License

The code of InsightFace is released under the MIT License. There is no limitation for both acadmic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

## CVer Presentation 

[Slides](https://pan.baidu.com/s/1v9fFHBJ8Q9Kl9Z6GwhbY6A)
 
## ArcFace Video Demo

[![ArcFace Demo](https://github.com/deepinsight/insightface/blob/master/resources/facerecognitionfromvideo.PNG)](https://www.youtube.com/watch?v=y-D1tReryGA&t=81s)

Please click the image to watch the Youtube video. For Bilibili users, click [here](https://www.bilibili.com/video/av38041494?from=search&seid=11501833604850032313).

## Recent Update

**`2019.04.30`**: Our Face detector ([RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)) obtains state-of-the-art results on [the WiderFace dataset](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html).

**`2019.04.14`**: We will launch a [Light-weight Face Recognition challenge/workshop](https://github.com/deepinsight/insightface/tree/master/iccv19-challenge) on ICCV 2019.

**`2019.04.04`**: Arcface achieved state-of-the-art performance (7/109) on the NIST Face Recognition Vendor Test (FRVT) (1:1 verification)
[report](https://www.nist.gov/sites/default/files/documents/2019/04/04/frvt_report_2019_04_04.pdf) (name: Imperial-000 and Imperial-001). Our solution is based on [MS1MV2+DeepGlintAsian, ResNet100, ArcFace loss]. 

**`2019.02.08`**: Please check [https://github.com/deepinsight/insightface/tree/master/recognition](https://github.com/deepinsight/insightface/tree/master/recognition) for our parallel training code which can easily and efficiently support one million identities on a single machine (8* 1080ti).

**`2018.12.13`**: Inference acceleration [TVM-Benchmark](https://github.com/deepinsight/insightface/wiki/TVM-Benchmark).

**`2018.10.28`**: Light-weight attribute model [Gender-Age](https://github.com/deepinsight/insightface/tree/master/gender-age). About 1MB, 10ms on single CPU core. Gender accuracy 96% on validation set and 4.1 age MAE.

**`2018.10.16`**: We achieved state-of-the-art performance on [Trillionpairs](http://trillionpairs.deepglint.com/results) (name: nttstar) and [IQIYI_VID](http://challenge.ai.iqiyi.com/detail?raceId=5afc36639689443e8f815f9e) (name: WitcheR). 

## Contents
[Deep Face Recognition](#deep-face-recognition)
- [Introduction](#introduction)
- [Training Data](#training-data)
- [Train](#train)
- [Pretrained Models](#pretrained-models)
- [Verification Results On Combined Margin](#verification-results-on-combined-margin)
- [Test on MegaFace](#test-on-megaface)
- [512-D Feature Embedding](#512-d-feature-embedding)
- [Third-party Re-implementation](#third-party-re-implementation)

[Face Alignment](#face-alignment)

[Face Detection](#face-detection)

[Citation](#citation)

[Contact](#contact)

## Deep Face Recognition

### Introduction

In this repository, we provide training data, network settings and loss designs for deep face recognition.
The training data includes the normalised MS1M, VGG2 and CASIA-Webface datasets, which were already packed in MXNet binary format.
The network backbones include ResNet, MobilefaceNet, MobileNet, InceptionResNet_v2, DenseNet, DPN.
The loss functions include Softmax, SphereFace, CosineFace, ArcFace and Triplet (Euclidean/Angular) Loss.


![margin penalty for target logit](https://github.com/deepinsight/insightface/raw/master/resources/arcface.png)

Our method, ArcFace, was initially described in an [arXiv technical report](https://arxiv.org/abs/1801.07698). By using this repository, you can simply achieve LFW 99.80%+ and Megaface 98%+ by a single model. This repository can help researcher/engineer to develop deep face recognition algorithms quickly by only two steps: download the binary dataset and run the training script.

### Training Data

All face images are aligned by [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) and cropped to 112x112:

Please check [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) for detail information and dataset downloading.


* Please check *src/data/face2rec2.py* on how to build a binary face dataset. Any public available *MTCNN* can be used to align the faces, and the performance should not change. We will improve the face normalisation step by full pose alignment methods recently.

### Train

1. Install `MXNet` with GPU support (Python 2.7).

```
pip install mxnet-cu90
```

2. Clone the InsightFace repository. We call the directory insightface as *`INSIGHTFACE_ROOT`*.

```
git clone --recursive https://github.com/deepinsight/insightface.git
```

3. Download the training set (`MS1M-Arcface`) and place it in *`$INSIGHTFACE_ROOT/datasets/`*. Each training dataset includes at least following 6 files:

```Shell
    faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

The first three files are the training dataset while the last three files are verification sets.

4. Train deep face recognition models.
In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/recognition/`*.
```Shell
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
```

Place and edit config file:
```Shell
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```

We give some examples below. Our experiments were conducted on the Tesla P40 GPU.

(1). Train ArcFace with LResNet100E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore
```
It will output verification results of *LFW*, *CFP-FP* and *AgeDB-30* every 2000 batches. You can check all options in *config.py*.
This model can achieve *LFW 99.80+* and *MegaFace 98.3%+*.

(2). Train CosineFace with LResNet50E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore
```

(3). Train Softmax with LMobileNet-GAP.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss softmax --dataset emore
```

(4). Fine-turn the above Softmax model with Triplet loss.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss triplet --lr 0.005 --pretrained ./models/m1-softmax-emore,1
```


5. Verification results.

*LResNet100E-IR* network trained on *MS1M-Arcface* dataset with ArcFace loss:

| Method  | LFW(%) | CFP-FP(%) | AgeDB-30(%) |  
| ------- | ------ | --------- | ----------- |  
|  Ours   | 99.80+ | 98.0+     | 98.20+      |   



### Pretrained Models

You can use `$INSIGHTFACE/src/eval/verification.py` to test all the pre-trained models.

**Please check [Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) for more pretrained models.**



### Verification Results on Combined Margin

A combined margin method was proposed as a function of target logits value and original `θ`:

```
COM(θ) = cos(m_1*θ+m_2) - m_3
```

For training with `m1=1.0, m2=0.3, m3=0.2`, run following command:
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss combined --dataset emore
```

Results by using ``MS1M-IBUG(MS1M-V1)``

| Method           | m1   | m2   | m3   | LFW   | CFP-FP | AgeDB-30 |
| ---------------- | ---- | ---- | ---- | ----- | ------ | -------- |
| W&F Norm Softmax | 1    | 0    | 0    | 99.28 | 88.50  | 95.13    |
| SphereFace       | 1.5  | 0    | 0    | 99.76 | 94.17  | 97.30    |
| CosineFace       | 1    | 0    | 0.35 | 99.80 | 94.4   | 97.91    |
| ArcFace          | 1    | 0.5  | 0    | 99.83 | 94.04  | 98.08    |
| Combined Margin  | 1.2  | 0.4  | 0    | 99.80 | 94.08  | 98.05    |
| Combined Margin  | 1.1  | 0    | 0.35 | 99.81 | 94.50  | 98.08    |
| Combined Margin  | 1    | 0.3  | 0.2  | 99.83 | 94.51  | 98.13    |
| Combined Margin  | 0.9  | 0.4  | 0.15 | 99.83 | 94.20  | 98.16    |

### Test on MegaFace

Please check *`$INSIGHTFACE_ROOT/Evaluation/megaface/`* to evaluate the model accuracy on Megaface. All aligned images were already provided.


### 512-D Feature Embedding

In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/deploy/`*. The input face image should be generally centre cropped. We use *RNet+ONet* of *MTCNN* to further align the image before sending it to the feature embedding network.

1. Prepare a pre-trained model.
2. Put the model under *`$INSIGHTFACE_ROOT/models/`*. For example, *`$INSIGHTFACE_ROOT/models/model-r100-ii`*.
3. Run the test script *`$INSIGHTFACE_ROOT/deploy/test.py`*.

For single cropped face image(112x112), total inference time is only 17ms on our testing server(Intel E5-2660 @ 2.00GHz, Tesla M40, *LResNet34E-IR*).

### Third-party Re-implementation

- TensorFlow: [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
- TensorFlow: [tf-insightface](https://github.com/AIInAi/tf-insightface)
- PyTorch: [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- PyTorch: [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
- Caffe: [arcface-caffe](https://github.com/xialuxi/arcface-caffe)
- Caffe: [CombinedMargin-caffe](https://github.com/gehaocool/CombinedMargin-caffe)
- Tensorflow: [InsightFace-tensorflow](https://github.com/luckycallor/InsightFace-tensorflow)


## Face Alignment

Please check the [Menpo](https://github.com/jiankangdeng/MenpoBenchmark) Benchmark and [Dense U-Net](https://github.com/deepinsight/insightface/tree/master/alignment) for more details.

## Face Detection

Please check [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) for more details.

## Citation

If you find *InsightFace* useful in your research, please consider to cite the following related papers:

```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}

@inproceedings{guo2018stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={BMVC},
  year={2018}
}

@article{deng2018menpo,
  title={The Menpo benchmark for multi-pose 2D and 3D facial landmark localisation and tracking},
  author={Deng, Jiankang and Roussos, Anastasios and Chrysos, Grigorios and Ververas, Evangelos and Kotsia, Irene and Shen, Jie and Zafeiriou, Stefanos},
  journal={IJCV},
  year={2018}
}

@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```

## Contact

```
[Jia Guo](guojia[at]gmail.com)
[Jiankang Deng](jiankangdeng[at]gmail.com)
```
