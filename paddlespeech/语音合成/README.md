# paddleOCR获取文字
```python
import IPython.display as dp
from PIL import Image
img_path = 'download/ocr_result.jpg'
im = Image.open(img_path)
dp.display(im)
```

[代码详情](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/story_talker)

# paddleGAN合成唇形
[代码详情](https://github.com/PaddlePaddle/PaddleGAN)

# 背景知识
文本转语音，又称语音合成(Speech Sysnthesis)

发展历史

## 主流方法
基于统计参数的语音合成、波形拼接语音合成、混合方法以及端到端神经网络语音合成

## 流水线三模块
文本前端（Text Frontend）、声学模型（Acoustic Model、声码器（Vocoder）

# 实践
获取paddlepaddle预训练模型

## 文本前端

## 声学模型

## 声码器

## PaddleSpeech训练TTS模型
基于CSMCS数据集FastSpeech2模型

基于CSMCS数据集训练Parallel WaveGAN模型
