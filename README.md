# 简介

用keras实现ocr定位、识别，后端tensorflow

* 环境 win10 titanx
# 识别
* crnn：vgg + blstm + blsmt + ctc 原版crnn
测试速度较慢 32X280 耗时 60ms ,放弃优化。
数据集是自己造的随机数据，且比较简单。如果要测试请去这里下载 https://github.com/senlinuc/caffe_ocr 语料数据集
而且caffe_ocr的crnn更快 9ms，不知道tf和caffe差在哪里？

* densenet-ocr ：densent + ctc 无lstm
是对caffe_cor的复现 https://github.com/senlinuc/caffe_ocr (包含数据集)


# 定位
* CTPN：效果很好，且已有tensorflow实现，但是框架太重。下一步用keras实现或者用其他框架，目前还没确定



![demo](https://github.com/xiaomaxiao/keras_ocr/blob/master/demo/demo1.jpg)
