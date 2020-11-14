from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#下載資料
digits = datasets.load_digits()
print(digits.images)
n_sample = len(digits.images)
#把資料轉換為二維資料，x的行資料是不同樣本資料，列是樣本屬性。
x = digits.images.reshape(n_sample, -1)#取資料的所有行第一列資料
y = digits.target
#print(x)

#以下方法確定解釋變數只能有一個，但是多個解釋變數該怎麼處理呢,答案是x包含了眾多解釋變數
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']

for score in scores:
    print('Tuning hyper-parameters for %s'%score)
    print()
    #利用網格搜尋演算法構建評估器模型，並且對資料進行評估
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro'%score)
    clf.fit(x_train, y_train)
    print('最優引數：',clf.best_params_)
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('網格資料得分：','%0.3f (+/-%0.3f) for %r'%(mean, std, params))
    #這個std有的文章乘以2，但個人不知道為什麼需要乘以2，如有明白的朋友，求指點。
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(y_true)
    print(classification_report(y_true, y_pred))

#在獲取最優超引數之後， 用5折交叉驗證來評估模型
clf = SVC(kernel='rbf', C=1, gamma=1e-3)#最優模型
#對模型進行評分
scores = cross_val_score(clf, x, y, cv=5)
print(scores)