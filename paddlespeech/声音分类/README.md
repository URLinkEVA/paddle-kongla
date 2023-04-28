# 识别声音
人类可以根据不同声音的特征（频率，音色等）进行区分，对声音进行分类，还可以根据用途进行细分。

用PaddleSpeech的预训练模型对一段音频进行实时的声音检测

# 音频和特征提取
## 数字音频
以一段音频为例
```python
import IPython
import numpy as np
import matplotlib.pyplot as plt

# 获取示例音频
IPython.display.Audio('./dog.wav')

from paddleaudio import load
data, sr = load(file='./dog.wav', mono=True, dtype='float32')  # 单通道，float32音频样本点
print('wav shape: {}'.format(data.shape))
print('sample rate: {}'.format(sr))

# 展示音频波形
plt.figure()
plt.plot(data)
plt.show()
```
wav shape: (97280,)

sample rate: 44100

```python
!paddlespeech cls --input ./dog.wav
```
得出识别结果Dog 0.7919359803199768 

### paddlespeech功能
```
!paddlespeech help
```
```
Usage:
    paddlespeech <command> <options>

Commands:
    help                   Show help for commands.
    version                Show version and commit id of current package.
    stats                  Get speech tasks support models list.
    asr                    Speech to text infer command.
    cls                    Audio classification infer command.
    st                     Speech translation infer command.
    text                   Text command.
    tts                    Text to Speech infer command.
    vector                 Speech to vector embedding infer command.
    kws                    Keyword Spotting infer command.
```
## 音频特征提取
### 短时傅里叶变换
技术讲解+配图

分帧加窗

paddle.signal.stft演示提取可视化

### logfbank
mel频谱提出背景

计算方法并用paddleaudio.features.LogMelSpectrogram演示提取

## 声音分类方法
### 传统机器学习方法
决策树、svm和随机森林等

### 深度学习方法
优点


