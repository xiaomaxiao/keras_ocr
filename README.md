# 简介

用keras实现ocr定位、识别，后端tensorflow.

* 环境 win10 titanx
# 识别
* crnn：vgg + blstm + blsmt + ctc 原版crnn
测试速度较慢 32X280 耗时 60ms ,放弃优化。
数据集是自己造的随机数据，且比较简单。

* densenet-ocr ：densent + ctc 无lstm

| 网格结构  | GPU | 准确率 | 模型大小 |
| ---------- | -----------| ---------- | -----------|
| crnn | 60ms | 0.972 |  |
| densent+ctc | 8ms | 0.982 | 18.9MB |


# 定位
* CTPN：效果很好，且已有tensorflow实现https://github.com/eragonruan/text-detection-ctpn，
但是框架太重。下一步用keras实现或者用其他框架，目前还没确定




![demo](https://github.com/xiaomaxiao/keras_ocr/blob/master/demo/demo1.jpg)

# 参考
[1]https://github.com/eragonruan/text-detection-ctpn

[2]https://github.com/senlinuc/caffe_ocr
