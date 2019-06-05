import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel('../data/CCPP/Folds5x2_pp.xlsx')
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#模型拟合测试集
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, y, cv=9)
from sklearn import metrics
# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y, predicted))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()