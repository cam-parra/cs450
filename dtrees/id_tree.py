from sklearn import datasets as ds
import pandas as pd
import numpy as np
from collections import Counter as co
from sklearn.cross_validation import train_test_split as tts
import sys

pd.options.mode.chained_assignment = None


class Node:
    def __init__(self, feature_name, child_nodes):
        self.feature_name = feature_name
        self.child_nodes = child_nodes


class DecisionTree:
    def __init__(self):
        self.classes = self.target = self.data = None
        self.tree = None

    def set_classes(self, classes):
        self.classes = classes

    def train(self, train_data, train_target):
        self.data = train_data
        self.target = train_target
        self.tree = self.make_tree(range(train_data.shape[1]), range(train_target.shape[0]))

    def make_tree(self, features_left, indices):
        # get list of each class result for the indices
        classes_list = list(map(lambda i: self.target[i], indices))

        # base case if there is only one unique class in list
        if np.unique(classes_list).size == 1:
            return classes_list[0]

        # base case if there are no more features
        if len(features_left) == 0:
            return co(classes_list).most_common(1)[0][0]

        # which feature has the lowest entropy
        best_feature = self.best_info_gain(features_left, indices)
        values_of_feature = np.unique(list(map(lambda i: self.data[:, best_feature][i], indices)))
        value_indices = list(map(
            lambda val: [ind for ind in indices if self.data[:, best_feature][ind] == val], values_of_feature))
        remaining = [i for i in features_left if i != features_left[best_feature]]
        return Node(features_left[best_feature],
                    {x: self.make_tree(remaining, y) for x in values_of_feature for y in value_indices})

    def best_info_gain(self, features, indices):
        """
        This function will calculate figure out which feature has the most info gain
        :param features: The features you have to look between
        :param indices: Which spots we are looking at right now
        :return: The index of the feature with the best info gain or least entropy
        """
        return np.argmin(list(map(lambda feature: self.entropy_of_feature(feature, indices), features)))

    def entropy_of_feature(self, feature, indices):
        # get the possible values for the feature
        values_of_feature = np.unique(self.data[:, feature])

        # get the indices of each of those values
        value_indices = list(map(
            lambda val: [ind for ind in indices if self.data[:, feature][ind] == val], values_of_feature))
        cnt = co()
        # total number for weighted average
        num_total = sum(map(lambda val_i: len(val_i), value_indices))
        total_entropy = 0

        for vi in value_indices:
            num = len(vi)
            for i in vi:
                cnt[self.target[i]] += 1
            total_entropy += (num / num_total) * sum(map(
                lambda c: -cnt[c] / num * np.log2(cnt[c] / num) if cnt[c] != 0 else 0, self.classes))
            cnt.clear()
        return total_entropy


def get_dataset(convert_nominal=True):
    which = int(input("Please choose a Dataset:\n1 - Iris\n2 - Cars\n>> "))
    if which == 1:
        iris = ds.load_iris()
        return iris.data, iris.target, iris.target_names
    else:
        my_read_in = pd.read_csv("../GoodNeighbor/cars.csv", dtype=str,
                                 names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"])
        car_data = my_read_in.ix[:, :-1]
        car_target = my_read_in.target
        if convert_nominal:
            car_target = car_target.replace("unacc", 0).replace("acc", 1).replace("good", 2).replace("vgood", 3)
            car_data.buying = car_data.buying.replace("low", 1).replace("med", 2).replace("high", 3).replace("vhigh", 4)
            car_data.maint = car_data.maint.replace("low", 1).replace("med", 2).replace("high", 3).replace("vhigh", 4)
            car_data.doors = car_data.doors.replace("2", 2).replace("3", 3).replace("4", 4).replace("5more", 5)
            car_data.persons = car_data.persons.replace("2", 2).replace("4", 4).replace("more", 6)
            car_data.lug_boot = car_data.lug_boot.replace("small", 1).replace("med", 2).replace("big", 3)
            car_data.safety = car_data.safety.replace("low", 1).replace("med", 2).replace("high", 3)

        return car_data.values, car_target.values, ["unacc", "acc", "good", "vgood"]


def main(argv):
    # process_data()
    d, t, ta = get_dataset(False)

    my_classifier = DecisionTree()
    train, test, t_target, test_target = tts(d, t, train_size=.8, random_state=20123)
    my_classifier.set_classes(ta)
    my_classifier.train(train, t_target)
    print(my_classifier.tree.feature_name, my_classifier.tree.child_nodes,
          my_classifier.tree.child_nodes['low'].feature_name, my_classifier.tree.child_nodes['low'].child_nodes,
          my_classifier.tree.child_nodes['low'].child_nodes['more'].feature_name,
          my_classifier.tree.child_nodes['low'].child_nodes['more'].child_nodes,
          my_classifier.tree.child_nodes['low'].child_nodes['more'].child_nodes['low'].feature_name,
          sep='\n')


if __name__ == '__main__':
    main(sys.argv)
