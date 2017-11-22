# 简介

用keras实现ocr定位、识别，后端tensorflow

* crnn：vgg + blstm + blsmt + ctc 原版crnn
测试速度较慢 32X280 耗时 60ms ,放弃
数据集是自己造的随机数据，且比较简单。如果要测试请去这里下载 https://github.com/senlinuc/caffe_ocr 预料数据集

* densenet-ocr ：densent + ctc 无lstm
是对caffe_cor的复现 https://github.com/senlinuc/caffe_ocr (包含数据集)



