from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split as tsp
import numpy as np
import pandas as pd
from collections import Counter as co


class Node():
    def __init__(self, feature_name, child_nodes):
        self.feature_name = feature_name
        self.child_nodes = child_nodes

class DTree:
    # classes = target = data = train = test = train_target = test_target = None
    classes = None
    target = None
    data = None
    train = None
    test = None
    train_target = None
    test_target = None

    initiated = False

    def start(self, user_data, user_target):
        """
        This function is meant to work as an __init__ to initiate data
        and make it into workable chunks

        :param user_data: sets class data
        :param user_target: sets class target
        :return: None
        """
        self.data, self.target = user_data, user_target
        self.train, self.test, self.train_target, self.test_target = tsp(user_data, user_target, test_size=.33)
        self.initiated = True

    # ################################################################################################################ #
    #                                                      ID3                                                         #
    # ################################################################################################################ #
    def entropy(self):
        return sum(list(
            (map(lambda x: -x * np.log2(x), (list(map(lambda x: x / len(self.target), co(self.target).values())))))))

    def sub_entropy(self, target_values):
        return sum(
            list(map(lambda x: -x * np.log2(x), (map(lambda x: x / len(target_values), co(target_values).values())))))

    def info_gain(self, feature, indicies):
        values_of_feature = np.unique(self.data[:, feature])
        print(values_of_feature)


    # ################################################################################################################ #
    #                                                    TREE FUNCS                                                    #
    # ################################################################################################################ #
    def make_tree(self, data, target, target_names):
        default = np.array(co(target).values()).argmax()


def get_dataset():
    my_tree = DTree()
    which = int(input("Please choose a Dataset:\n1 - Iris\n2 - Cars\n>> "))
    if which == 1:
        iris = ds.load_iris()
    else:
        my_read_in = pd.read_csv("../GoodNeighbor/cars.csv", dtype=str,
                                 names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"])
        car_data = my_read_in.ix[:, :-1]
        car_target = my_read_in.target.replace("unacc", 0).replace("acc", 1).replace("good", 2).replace("vgood", 3)
        car_data.buying = car_data.buying.replace("low", 1).replace("med", 2).replace("high", 3).replace("vhigh", 4)
        car_data.maint = car_data.maint.replace("low", 1).replace("med", 2).replace("high", 3).replace("vhigh", 4)
        car_data.doors = car_data.doors.replace("2", 2).replace("3", 3).replace("4", 4).replace("5more", 5)
        car_data.persons = car_data.persons.replace("2", 2).replace("4", 4).replace("more", 6)
        car_data.lug_boot = car_data.lug_boot.replace("small", 1).replace("med", 2).replace("big", 3)
        car_data.safety = car_data.safety.replace("low", 1).replace("med", 2).replace("high", 3)

    return car_data.values, car_target.values, ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"]

my_tree = DTree()
car_data , car_target, values = get_dataset()
my_tree.start(car_data, car_target)
my_tree.info_gain(, car_data.size()[0])
print(car_data)