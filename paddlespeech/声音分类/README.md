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

### Pretrain+Finetune
预训练基础上微调


# 实践环境声音分类
选取 `PANNs` 中的预训练模型 `cnn14` 作为 backbone，用于提取声音的深层特征，`SoundClassifer`创建下游的分类网络，实现对输入音频的分类。

数据集采用ESC-50

# 参考文献

[1] Guzhov, A., Raue, F., Hees, J., & Dengel, A.R. (2021). AudioCLIP: Extending CLIP to Image, Text and Audio. ArXiv, abs/2106.13043.
  
[2] Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M.D. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 2880-2894.
  
[3] Gong, Y., Chung, Y., & Glass, J.R. (2021). AST: Audio Spectrogram Transformer. ArXiv, abs/2104.01778.
  
[4] Gemmeke, J.F., Ellis, D.P., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M., & Ritter, M. (2017). Audio Set: An ontology and human-labeled dataset for audio events. 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 776-780.

[5] Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd ACM international conference on Multimedia.
