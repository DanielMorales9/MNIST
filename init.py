import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA

# from chart import create_charts
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random

from sklearn.preprocessing import StandardScaler

my_data = genfromtxt('data/train.csv', dtype=int, delimiter=',', skip_header=1)
test = genfromtxt('data/test.csv', dtype=int, delimiter=',',  skip_header=1)

pick_index = random.random() * len(my_data)
pick_index = int(pick_index)

train_x = np.array(my_data[:, 1:])
train_x = train_x.astype('float64')
train_y = my_data[:, 0]
# create_charts(train_y[pick_index], train_x[pick_index], "BEFORE RESCALING")
# train_x = np.log(train_x+1)/10
data = np.append(train_x, test, axis=0)


scaler = StandardScaler(with_std=False)
data = scaler.fit_transform(data)

# Normalizing by st deviation
# train_x /= np.std(train_x, axis=0) not important: already normalized between (0-255)

# PCA whitening
pca = PCA()
data = pca.fit_transform(data)
variance = pca.explained_variance_ratio_.cumsum()


def getDimensionToReduce(variance):
    k = 0
    vlen = len(variance)
    while k < vlen:
        if variance[k] >= 0.99:
            return k
        k+=1
    return vlen


n_components = getDimensionToReduce(variance)


pca = PCA(n_components=n_components, whiten=True)

data = pca.fit_transform(data)
train_rows = train_x.shape[0]
test_rows = test.shape[0]
train_x = data[:train_rows, :]
test = data[train_rows:train_rows+test_rows, :]

clf = RandomForestClassifier(n_estimators=100)
# clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(784, 784), random_state=1)
#scores = cross_val_score(clf, train_x, train_y, cv=10)
#print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

clf.fit(train_x, train_y)
prediction = clf.predict(test)

plen = len(prediction)

id = np.arange(1,plen+1, 1)

mat = np.zeros((plen,2))

mat[:,0] = id
mat[:,1] = prediction

np.savetxt('data/submission.csv',
            mat,
           delimiter=',',
            fmt='%d',
           newline='\n',
           header='ImageId,Label')
