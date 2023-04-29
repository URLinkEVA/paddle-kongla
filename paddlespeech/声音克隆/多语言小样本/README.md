# 概念简介
语音合成、中英混合语音合成、小样本合成

# 中英混合语音合成
难点：一致性，中英文衔接不自然

采用数据：csmsc、ljspeech

文本前端加停顿

Speaker Embedding提取

## 数据准备

| 语言 | 数据集 |音频信息 | 描述 |
| -------- | -------- | -------- | -------- |
| 中文 | [CSMSC](https://www.data-baker.com/open_source.html) | 48KHz, 16bit | 单说话人，女声，约12小时，具有高音频质量 |
| 中文 | [AISHELL-3](http://www.aishelltech.com/aishell_3) | 44.1kHz，16bit | 多说话人（218人），约85小时，音频质量不一致（有的说话人音频质量较高）|
| 英文 | [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) | 22050Hz, 16bit | 单说话人，女声，约24小时，具有高音频质量|
| 英文 | [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | 48kHz, 16bit | 多说话人（110人）， 约44小时，音频质量不一致（有的说话人音频质量较高）|

## 模型训练
训练声学模型：使用上述数据训练 fastspeech2 的24KHz模型

预训练声码器：使用aishell3 训练的 HiFiGAN 模型

run.sh 包含预处理、训练、合成、静态图推理等步骤
```
都在dump下，先做一个均值方差，再做一个normalize，再用train下的npy文件训练
```

## 文本前端设计
