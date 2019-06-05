# python-algorithm

https://github.com/ljpzzz/machinelearning#2

https://github.com/MLEveryday/100-Days-Of-ML-Code

(1) 数据预处理

1. 导入数据集

dataset = pd.read_csv('Data.csv')//读取csv文件

X = dataset.iloc[:,:-1].values// 全部行,第0-倒数第二列

Y = dataset.iloc[:,3].values  // : 全部行,第4列

2. 处理丢失数据

from sklearn.preprocessing import Imputer

3. 解析分类数据

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

4. 拆分数据集为训练集合和测试集合

from sklearn.model_selection import train_test_split

5. 特征归一化

from sklearn.preprocessing import StandardScaler


(2) 回归算法：线性回归，多项式回归，以及广义线性回归

1. 根据损失函数J，利用梯度下降（Gradient Descent），迭代更新参数θ

2. 根据损失函数J对θ的偏导等于θ，利用最小二乘（Least squares），直接计算θ

3. 根据状态方程，根据θ+h(θ)到目标状态target的差值d1与θ到目标状态target的差值d0的差值d=d1-d0，数值近似计算对参数0的雅克比矩阵，利用牛顿法，迭代更新参数θ

4. 交叉验证(Cross Validation)

5. 精确率与召回率，RoC曲线与PR曲线

6. 线性回归的正则化（Regularization）

7. scikit-learn与pandas


(3) 分类算法：逻辑回归，决策树，K近邻法(KNN)，朴素贝叶斯，支持向量机

1. scikit-learn 逻辑回归

2. scikit-learn 决策树算法

3. scikit-learn K近邻法（KD树）

4. scikit-learn 朴素贝叶斯

5. scikit-learn 感知机

6. scikit-learn 支持向量机算法（拉格朗日函数，SMO算法）


(4) 聚类算法：K-Means，BIRCH，DBSCAN，spectral

1. scikit-learn K-Means聚类

2. scikit-learn BIRCH聚类

3. scikit-learn DBSCAN密度聚类

4. scikit-learn 谱聚类（spectral clustering)


(5) 集成学习算法：Adaboost，梯度提升树，随机森林

1. scikit-learn Adaboost

2. scikit-learn 梯度提升树(GBDT)

3. scikit-learn 随机森林


(6) 降维算法：主成分分析，线性判别分析，局部线性嵌入

1. scikit-learn 主成分分析(PCA)

2. scikit-learn 线性判别分析(LDA)

3. scikit-learn 局部线性嵌入(LLE)


(7) 其他算法

1. 隐马尔科夫模型HMM(EM算法原理)

2. 条件随机场CRF

3. word2vec原理


(8) 深度学习算法：DNN，CNN，RNN，RBM

1. 模型结构，前向传播算法与反向传播算法(BP)

2. 损失函数和激活函数的选择

3. 正则化


(9) 强化学习：MDP，SARSA，Q-Learning，Deep Q-Learning，Policy Gradient

1. 马尔科夫决策过程(MDP)

2. 时序差分在线控制算法SARSA

3. 时序差分离线控制算法Q-Learning

4. 价值函数的近似表示与Deep Q-Learning

5. Nature DQN，DDQN，Prioritized Replay DQN，Dueling DQN

6. 策略梯度(Policy Gradient)

7. Actor-Critic，A3C，DDPG，Dyna，MCTS
