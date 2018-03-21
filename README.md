# NLP-Beginner：自然语言处理入门练习

新加入本实验室的同学，请按要求完成下面练习，并提交报告。

*请完成每次练习后把report上传到QQ群中的共享文件夹中的“Reports of nlp-beginner”目录，文件命名格式为“task 1+姓名”。*

参考：

1. [深度学习上手指南](https://github.com/nndl/nndl.github.io/blob/master/md/DeepGuide.md)
2. 《[神经网络与深度学习](https://nndl.github.io/)》 
3. 不懂问google





### 任务一：基于机器学习的文本分类

实现基于logistic/softmax regression的文本分类

1. 参考
   1. [文本分类](文本分类.md)
   2. 《[神经网络与深度学习](https://nndl.github.io/)》 第2/3章
2. 数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
3. 实现要求：NumPy
4. 需要了解的知识点：

   1. 文本特征表示：Bag-of-Word，N-gram
   2. 分类器：logistic regression，损失函数、（随机）梯度下降、特征选择
   3. 数据集：训练集/验证集/测试集的划分
5. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch 
6. 时间：两周

### 任务二：基于词嵌入的文本分类

熟悉Pytorch，用Pytorch重写《任务一》的分类器

1. 参考

   1. https://www.tensorflow.org/
   2. 词嵌入
      1. word2vec
      2. glove https://nlp.stanford.edu/projects/glove/

2. 词用embedding 的方式初始化；

  （1）随机embedding的初始化方式
  （2）用glove 训练出来的文本初始化

3. 实现Continuous BOW模型、CNN、RNN的文本分类；

4. 时间：三周

### 任务三：基于神经网络的语言模型

用LSTM、GRU来训练字符级的语言模型，计算困惑度

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、15章
2. 数据集：poetryFromTang.txt
3. 实现要求：TensorFlow
4. 知识点：
   1. 语言模型：困惑度等
   2. 文本生成
5. 时间：两周


### 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、11章
   2. https://arxiv.org/pdf/1603.01354.pdf
   3. https://arxiv.org/pdf/1603.01360.pdf
2. 数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/
3. 实现要求：TensorFlow
4. 知识点：
   1. 评价指标：precision、recall、F1
   2. 无向图模型、CRF
5. 时间：两周