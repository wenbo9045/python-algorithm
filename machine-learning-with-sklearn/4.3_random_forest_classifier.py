import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import matplotlib.pylab as plt

train = pd.read_csv('../data/train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'

x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns].values
# X = train[['Existing_EMI', 'Loan_Amount_Applied']].values
y = train['Disbursed'].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

rf2 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
rf2.fit(X,y)

y_pred = rf2.predict(X)
y_predprob = rf2.predict_proba(X)[:,1]
print ("Accuracy : %.4g" % metrics.accuracy_score(y, y_pred))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

from sklearn.model_selection import GridSearchCV
param_test = {'n_estimators':[10,20,30,40,50,60,70]}
gsearch = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),
                       param_grid = param_test, scoring='roc_auc',cv=5)
gsearch.fit(X,y)
print(gsearch.best_params_, gsearch.best_score_)

# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                      np.arange(y_min, y_max, 0.02))
#
# Z = gbm1.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# cs = plt.contourf(xx, yy, Z)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
# plt.show()