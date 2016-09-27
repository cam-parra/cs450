from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split as tsp
from collections import Counter as ct
import numpy as np
from collections import Counter as co

iris = ds.load_iris()
train, test, train_target, test_target = tsp(iris.data, iris.target, test_size=.50, random_state=12)

items = iris.data.shape[1]
columns = []
for i in range(items):
    columns.append(test[:, i])

entropy = []
for index in columns:
    entropy.append(sum(list(map(lambda r: -r * np.log2(r) if r != 0 else 0, index))))

my = np.unique(columns[0], return_index=True)
uniquevalues = []
for i in my[1]:
    uniquevalues.append(test_target[i])

print(co(uniquevalues))

