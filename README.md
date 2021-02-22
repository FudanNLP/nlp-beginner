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
   2. 分类器：logistic/softmax  regression，损失函数、（随机）梯度下降、特征选择
   3. 数据集：训练集/验证集/测试集的划分
5. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch 
6. 时间：两周

## Report
date:2021-2-19:2021-2-20  
1.reference:  
   None  
2.dataset:  
   train.tsv/test.tsv:  
3.lib:  
   numpy;pandas;matplotlib  
4.feature:  
   BOW;Ngram (high level 19422/136648)  
5.details:  
   softmax;argmax;shuffle;batch;iteration;  
6.result:  
  2000:epoch:10 acc:0.67
  20000:epoch:10 acc:0.55 (loss no longer decreases while epoch grows)  
7.conclusion:  
   problem 1：我认为softmax + fc学到的只是目标数据集的分布，用dataloader.check_dataset()实验发现数据集分布及其不均匀，而softmax训练后的结果也只是略高于最大的那一项，我认为19422*5这么多的参数面对复杂语言问题表现的极限就是猜出目标分布+一点点记忆（5%），调了很久参数，或是pytorch都很难得到收敛的loss曲线，并且我在尝试使用均匀化的数据集后，准确率最多达到了0.22左右，为了严谨我今晚会看一下别人的表现并用pytorch彻底地重写这部分，来验证我的观点，当然也可能是我的问题。
8.thinking:
   之前有个小bug是BOW的维度我舍得过高，改了之后pytorch run了一下，能复现别人的0.8train 0.5 test的acc，不过感觉没啥意义，我在自己的dataloader上得到均匀的数据集来训练，overfitting，train很高，test很低。到此为止task1结束，很有限，模型学到的东西很少，只是在硬背罢了，Ngrams实现了但是没run，意义不大。


### 任务二：基于深度学习的文本分类

熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；

1. 参考

   1. https://pytorch.org/
   2. Convolutional Neural Networks for Sentence Classification <https://arxiv.org/abs/1408.5882>
   3. <https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/>
2. word embedding 的方式初始化
1. 随机embedding的初始化方式
  2. 用glove 预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/
3. 知识点：

   1. CNN/RNN的特征抽取
   2. 词嵌入
   3. Dropout
4. 时间：两周

### 任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考[ESIM]( https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第7章
   2. Reasoning about Entailment with Neural Attention <https://arxiv.org/pdf/1509.06664v1.pdf>
   3. Enhanced LSTM for Natural Language Inference <https://arxiv.org/pdf/1609.06038v3.pdf>
2. 数据集：https://nlp.stanford.edu/projects/snli/
3. 实现要求：Pytorch
4. 知识点：
   1. 注意力机制
   2. token2token attetnion
5. 时间：两周


### 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、11章
   2. https://arxiv.org/pdf/1603.01354.pdf
   3. https://arxiv.org/pdf/1603.01360.pdf
2. 数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/
3. 实现要求：Pytorch
4. 知识点：
   1. 评价指标：precision、recall、F1
   2. 无向图模型、CRF
5. 时间：两周

### 任务五：基于神经网络的语言模型

用LSTM、GRU来训练字符级的语言模型，计算困惑度

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、15章
2. 数据集：poetryFromTang.txt
3. 实现要求：Pytorch
4. 知识点：
   1. 语言模型：困惑度等
   2. 文本生成
5. 时间：两周