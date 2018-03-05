# 简介

用keras实现ocr定位、识别，后端tensorflow.

* 环境 win10 titanx

# 识别
* 数据集链接: https://pan.baidu.com/s/1jJWfDmm 密码: vh8p (中英数300W+,语料不均衡)

* crnn：vgg + blstm + blstm + ctc 

* densenet-ocr ：densent + ctc 

| 网格结构  | GPU | 准确率 | 模型大小 |
| ---------- | -----------| ---------- | -----------|
| crnn | 60ms | 0.972 |  |
| densent+ctc | 8ms | 0.982 | 18.9MB |


# 定位

* 数据集链接: https://pan.baidu.com/s/1mj5Dj0c 密码: 6257 (ICDR 2013+少量中文)

* CTPN：代码持续整理上传中。

1. 即使大部分数据集基于英文，但在中文定位中也表现良好。
2. 各位如有中文标注的数据集愿意分享，可提issues

![demo](https://github.com/xiaomaxiao/keras_ocr/blob/master/demo/ctpn_demo2.png) 


---
![demo](https://github.com/xiaomaxiao/keras_ocr/blob/master/demo/demo1.jpg)

# 参考
[1]https://github.com/eragonruan/text-detection-ctpn

[2]https://github.com/senlinuc/caffe_ocr
