from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
ss = StandardScaler()
X, y = iris.data[:, :3], iris.data[:, 3]
X = ss.fit_transform(X)
model_lasso = LassoCV(alphas=[1, 0.5, 0.1, 0.05, 0.001], cv=5).fit(X, y)
print(model_lasso.alpha_)
print(model_lasso.coef_)  # 若为0表示剔除
