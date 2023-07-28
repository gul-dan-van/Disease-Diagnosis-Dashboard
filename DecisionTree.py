import pandas as pd
import math
from collections import Counter


# Defining function that will convert a list of symptoms to a multi-dimensional vector
def vectorize(symptoms, columns):
    vector = [0] * len(columns)  # Initialising all values of vector to be zero
    for x in symptoms:
        vector[columns.index(x)] = 1  # Setting values of corresponding symptoms

    return vector


# Node class to hold important values of tree
class Node:
    def __init__(self, entropy, sample_cnt, dominant_sample):
        self.feature = None  # Value of this feature will be used to split the data
        self.centre = 0.5  # Centre value about which data will be splitted
        self.entropy = entropy  # Entropy of the data that it holds
        self.sample_cnt = (
            sample_cnt  # Count array of diseases in data that is passed to the node
        )
        self.dominant_class = (
            dominant_sample  # Most frequent disease among the given data
        )
        self.left = None  # Left child
        self.right = None  # Right child


class TreeClassifier:
    def __init__(self, max_depth=5, min_samples=5):
        self.max_depth = max_depth
        self.min_samples = min_samples  # Min samples that a node must have so that we could split it further
        self.tree = None

    # Function to calculate entropy using formula -p * log(p)
    def calculate_entropy(self, samples):
        cnt = [*Counter(samples).values()]
        total = sum(cnt)
        n = len(samples)

        if n < 2:
            return 0

        E = 0
        for x in cnt:
            p = x / total
            if p > 0:
                E += -p * math.log(p, n)

        return E

    # Function Calculating contribution of each newly distributed dataset into the new entropy of data
    def weighted_entropy(self, left, right):
        l = len(left)
        r = len(right)
        n = l + r
        return l / n * self.calculate_entropy(left) + r / n * self.calculate_entropy(
            right
        )

    # Function Evaluating a node
    def evaluate(self, dataY):
        E = self.calculate_entropy(dataY)
        values = [*Counter(dataY).values()]

        max_cnt = -1
        dominant_sample = None
        for k, v in Counter(dataY).items():

            if max_cnt < v:
                max_cnt = v
                dominant_sample = k

        node = Node(E, values, dominant_sample)
        return node

    # Function that will return best feature that should be used to split the data
    def best_splitting_feature(self, dataX, dataY):

        entropies = []
        centre = 0.5  # feature value around which we will distribute the data

        for j in range(len(dataX[0])):
            partA = [dataY[i] for i in range(len(dataX)) if dataX[i][j] < centre]
            partB = [dataY[i] for i in range(len(dataX)) if dataX[i][j] >= centre]

            entropies += [(self.weighted_entropy(partA, partB), j)]

        return sorted(entropies)[
            0
        ]  # Returning the minimum of all entropies possible after splitting around different feature

    # Function to create tree recursively
    def create_tree(self, dataX, dataY, depth=0):

        if depth >= self.max_depth:
            return self.evaluate(dataY)
        if len(dataY) <= self.min_samples:
            return self.evaluate(dataY)

        intital_entropy = self.calculate_entropy(dataY)
        final_entropy, best_feature = self.best_splitting_feature(dataX, dataY)

        leftX = []
        rightX = []
        leftY = []
        rightY = []
        for i in range(len(dataX)):

            # Features with value 0 will go into the left child
            if dataX[i][best_feature] < 0.5:
                leftX.append(dataX[i])
                leftY.append(dataY[i])

            # Features with value 1 will go into the right child
            else:
                rightX.append(dataX[i])
                rightY.append(dataY[i])

        node = self.evaluate(
            dataY
        )  # Defining current node with the help of evaluate function
        node.feature = best_feature  # Storing the feature that helped us in getting maximum information gain

        # Recursively creating a tree from left child
        if len(leftY) > 0:
            node.left = self.create_tree(leftX, leftY, depth + 1)
        else:
            node.left = None

        # Recursively creating a tree from right child
        if len(rightX) > 0:
            node.right = self.create_tree(rightX, rightY, depth + 1)
        else:
            node.right = None

        return node

    # Function to intiate tree formation
    def fit(self, trainX, trainY):
        self.tree = self.create_tree(trainX, trainY)

    # Function that will help us to take use of trained model
    def predict(self, data):
        predictions = []

        for x in data:  # Iterating over all test cases

            # Considering case where no symptoms are passed
            if sum(x) == 0:
                predictions.append("You are fine!")
                continue

            cur = self.tree  # starting from root node
            while True:
                if (
                    cur.left == None and cur.right == None
                ):  # If there are no child of current node, we can't go deeper into the tree
                    break
                if (
                    x[cur.feature] < cur.centre
                ):  # If node feature is smaller than the node centre (0.5 for every case for this model), we will go to the left child
                    if cur.left != None:
                        cur = cur.left
                    else:  # We will break the traversal if there is no left node
                        break
                else:  # If node feature is greater than the node centre, we will go to the right child
                    if cur.right != None:
                        cur = cur.right
                    else:  # We will break the traversal if there is no right node
                        break

            predictions += [
                cur.dominant_class
            ]  # Taking the most frequent class in the current node as final answer

        return predictions

    # Function to calculate accuracy of the model
    def accuracy_score(self, dataX, dataY):

        truePos = 0
        predictions = self.predict(dataX)  # Getting predictions for the given data

        for predicted, actual in zip(predictions, dataY):
            if (
                predicted == actual
            ):  # Adding one to the count of correct responses if predicted value is same as actual value
                truePos += 1

        return truePos / len(dataY)


if __name__ == "__main__":
    # Loading Data
    svr = pd.read_csv("archive/Symptom-severity.csv")
    df = pd.read_csv("archive/dataset.csv")

    columns = svr["Symptom"].tolist()  # Extracting names of all symptoms
    r, c = df.shape
    trainX, trainY = [], []
    dataX, dataY = [], []

    new_df = pd.DataFrame(
        columns=columns
    )  # Defining a dataframe that will hold vectorized values of symptoms of corresponding diseases

    for i in range(r):  # Iterating over all rows
        disease, *symptoms = df.iloc[i, :].dropna()  # Dropping Null values
        symptoms = ["".join(x.split()) for x in symptoms]
        vector = vectorize(symptoms, columns)

        dataY += [disease]
        dataX += [vector]

        new_df.loc[i] = vector

    # Splitting dataset randomly to test our model with unseen data
    from sklearn.model_selection import train_test_split

    trainX, testX, trainY, testY = train_test_split(
        dataX, dataY, test_size=0.2, random_state=100
    )

    model = TreeClassifier(max_depth=10, min_samples=5)
    model.fit(trainX, trainY)
    print(model.accuracy_score(testX, testY))

    # Loading model to a pickle file so that we will not have to run the model for training each time the web app will be launched
    import pickle

    with open("DecisionTreeModel.txt", "wb") as file:
        pickle.dump(model, file)

    with open("columns.txt", "wb") as file:
        pickle.dump(columns, file)

    with open("DecisionTreeModel.txt", "rb") as file:
        c = pickle.load(file)
