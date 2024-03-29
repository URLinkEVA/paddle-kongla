# Prompt优化思路

例：帮我做一幅很好看的画，一只猫猫趴在透明的泡泡上，眼睛盯着前方看，泡泡上还打着光非常可爱，整体上是粉色系为主的动画风格

- 缺乏细节描述：找到更多"细节词"进行补充
- 未把握重点：提升大模型对重点词敏感度来解决
- prompt过长：寻找不依赖"细节词"的方法提升效果



# Prompt优化原理

通俗来说：给模型输入什么数据，模型就会尝试学习什么内容

模仿分词方式表达需求：

画一幅画，呆萌的小猫躺在大泡泡中，可爱温柔，动漫风格，暖系色调，居中，面下镜头，虚幻引擎，棉花糖质感，光线追踪，极致细节，质感细腻，8K，超高清，超广角，极致清晰，丁达尔效应



# 十个技巧高效优化Prompt

## 迭代法

- 创造评估
- 基础创作
- 多轮次交互

## Trick法

- 戴高帽
- 增加引导语
- Few-shot
- 增加假设

## 工具法

- 检索类工具
- 优化类工具
- 收纳类工具



### 提强调

以专业影评为例，一般会覆盖情节、主题和基调、演技和角色、方向、配乐、电影摄影、制作设计、特效、剪辑、节奏、对话等主题

### 提预设

交互更多是细节的展示以及内容延伸

不建议作为细碎的需求修改反式，因为模型能够记忆的内容可能会随着多次对话而失去信息

例：

你是一个文本分类模型，我将粘贴文本，您需要通过对我输入的文本进行分类，分类可以是”积极“、”消极“、”普通“，适当给出分类的原因和解释。

文本：balabala...

### 戴高帽

实际上并不需要加一堆细节词，只需要加入一句"你就是游戏大作的特效师"就能得到很不错的效果，这就叫戴高帽

### 说好话

多想一想也适用于大模型，学习”思维链“的数据

例：

计算1+1=2，请给出步骤与答案

### 给提示

上面提到的都属于zero-shot，就是不给模型额外数据的情况下让模型作答，给出少量样例后模型能做出更好的结果

### 做假设

尽管使用很多方式让模型生成效果尽可能更加准确，但由于数据和采样策略的原因，我们很难保证大模型在生成时不会说胡话，使用”增加假设“的方式就可以让模型在有些犹豫时不进行瞎说

例：如果你的数据存在问题，例如数据不准确、缺乏时效性等，那么可以给出否定答复，例如：目前没有相关数据可供参考



## 问题

- 缺乏细节描述
- 未把握重点
- prompt过长



### 做搜索

检索网站Lexia，PromptHero...

### 看优化

平台润色prompt

promptperfect.jinaai.cn

### 搞收纳

PromptBox用来组织和保存人工智能提示的工具，结合插件保存共享分类prompt
