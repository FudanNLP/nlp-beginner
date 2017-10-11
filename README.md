# NLP-Beginner
自然语言处理入门教程



参考：[深度学习上手指南](https://github.com/nndl/nndl.github.io/blob/master/md/DeepGuide.md)



### 任务一：基于机器学习的文本分类

实现基于logistic/softmax regression的[文本分类](文本分类.md)

1. 数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
2. 实现要求：NumPy
3. 需要了解的知识点：

   1. 文本特征表示：Bag-of-Word，N-gram
   2. 分类器：logistic regression，损失函数、（随机）梯度下降、特征选择
   3. 数据集：训练集/验证集/测试集的划分
4. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch 
5. 时间：一周

### 任务二：基于词嵌入的文本分类

1. 熟悉tensorflow，用tensowflow 重写任务一的分类器；

2. 词用embedding 的方式初始化；

  （1）随机embedding的初始化方式
  （2）用glove 训练出来的文本初始化

3. 实现Continuous BOW模型的文本分类；

4. 时间：两周

### 任务三：基于神经网络的语言模型

1. 数据集：poetryFromTang.txt
2. 实现要求：用LSTM、GRU来训练字符级的语言模型
3. 知识点：
   1. [语言模型](https://nndl.github.io/chap-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%AF%8D%E5%B5%8C%E5%85%A5.pdf)
   2. 文本生成
4. 时间：两周

