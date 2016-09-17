from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split as tsp
from sklearn import svm
from sklearn.naive_bayes import GaussianNB as GNB
import random


iris = ds.load_iris()
iris.data.shape, iris.target.shape

# --------------------------Do the Shuffle------------------------------------ #
iris_data = sh(iris.data, random_state=0)
iris_target = sh(iris.target, random_state=0)
iris_names = sh(iris.target_names, random_state=0)


# ############################################################################# #
# #                                                                           # #
# #         This function just ensures user picks a valid option              # #
# #                                                                           # #
# ############################################################################# #


def prompt_user_des_set():
    print("What data would you like to see? Choose one of the following:")
    print("1. Numerical\n2. Target\n3. Names")
    user_set = int(input(">> "))
    while 3 < user_set < 1:
        prompt_user_des_set()
    return user_set


# ############################################################################# #
# #                                                                           # #
# #    show_right_set(var): handles user input and gives correct output       # #
# #                                                                           # #
# ############################################################################# #


def show_right_set(num_set):
    randomized = str(input("Would you like these randomized?[Y/n]\n>> ")).capitalize()
    if num_set == 1 and randomized == "Y":
        print("This is the current numerical set:\n")
        print(sh(iris.data), sep=',')
    elif num_set == 1 and randomized == "N":
        print("This is the current numerical set:\n", iris.data)
    elif num_set == 2 and randomized == "Y":
        print("This is the current target set:\n", sh(iris.target))
    elif num_set == 2 and randomized == "N":
        print("This is the current target set:\n", iris.target)
    elif num_set == 3 and randomized == "Y":
        print("This is the current names set:\n", sh(iris.targer_names))
    elif num_set == 3 and randomized == "N":
        print("This is the current target set:\n", iris.target_names)


def show_training_set_test_set():
    print("Please choose one of the following:")
    user_option = int(input("1. Training\n2. Test\n3. Training and Test\n>> "))
    user_percent = 0
    if user_option == 3:
        double_train, double_test = map(float, (input(
            "Please enter the desired percentage for training and test(i.e. .90 .10)\n>> ").split()))
        iris_train, iris_test = tsp(iris.data, train_size=double_train, test_size=double_test, random_state=0)
        print("The training set is split into ", double_train * 100, "%\n", iris_train)
        print("The test set is split into ", double_test * 100, "%\n", iris_test)
    else:
        user_percent = float(input("Please enter desired percentage(i.e. .90): "))
    if user_option == 1:
        iris_train, iris_test = tsp(iris.data, train_size=user_percent, random_state=0)
        print("The training set is split into ", user_percent * 100, "%\n", iris_train)
    elif user_option == 2:
        iris_train, iris_test = tsp(iris.data, test_size=user_percent, random_state=0)
        print("The test set is split into ", user_percent * 100, "%\n", iris_train)


class HardCodedClassifier:
    def __init__(self):
        self.data = iris.data
        self.target = iris.target

    @staticmethod
    def train_machine(train_set, train_target):
        print("Machine is trained, me Lord.")

    @staticmethod
    def predict(train_set):
        x = []
        for i in train_set:
            x.append("setsoa")
        percent = (len(x) / 150) * 100
        print(len(train_set), "where tested. ", percent, "% are setosa.")

        return x

# x_train, x_test, target_train, target_test = tsp(iris.data, iris.target, test_size=0.45, random_state=23)
# new_classi = HardCodedClassifier()
# new_classi.train_machine(x_test, target_test)
# new_classi.predict(x_test)



#print(iris.target)
#print(iris_data)
show_training_set_test_set()
show_right_set(prompt_user_des_set())

# ------------Make a Test and Training array---------------------------------- #
# iris_training, iris_test = tsp(iris.data, train_size=.70, random_state=0)

# -----------------------------------testing---------------------------------- #
# print("print the length of the whole set: ", len(iris_data))
# print("print the length of the training set: ", len(iris_training))
# print("print the length of the test set: ", len(iris_test), "\n")
# print("The training set is split into 70%\n", iris_training)
# print("The test set is split into 30%\n", iris_test)


class HardCodedClassifier:
    def __init__(self):
        self.data = iris.data
        self.target = iris.target

    @staticmethod
    def train_machine(train_set, train_target):
        print("Machine is trained, me Lord.")

    def predict_one(self,data):
        return "setsoa"

    @staticmethod
    def predict(train_set):
        x = []
        for i in train_set:
            x.append(predict_one(i))
        percent = (len(x) / 150) * 100
        print(len(train_set), "where tested. ", percent, "% are setosa.")

        return x


size = input
state = random.randint(1,100)
train_data, test_data , train_target, test_target = tsp(iris.data, iris.target, test_size=size, random_state=state)

def svm_predicter():
    clf = svm.SVC(kernel='linear', C=1).fit(train_data, train_target)
    print("Your results: ", clf.score(test_data,test_target))


def naive_bayes():
    clf = GNB()
    clf.fit(iris.data, iris.target)
    print(clf.predict(test_data))


def what_would_you_like():
    user_input = int(input("Please select from the following:\n1.Cam's Classifier\n2.SVM\n3.Naive Bayes\n>>> "))
    if user_input > 3 or user_input < 1:
        print("Error please choose a valid answer.")
        what_would_you_like()
    elif user_input == 1:
        print("fix me")
    elif user_input == 2:
        svm_predicter()
    elif user_input == 3:
        naive_bayes()


what_would_you_like()