import csv
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

class WeightClassifier:

    def __init__(self, hidden_neurons = 6, epsilon = 0.23, output_neurons = 2):
        self.output_neurons = max(min(output_neurons, 3), 1)
        self.hidden_neurons = hidden_neurons
        self.net = MultiplayerPerceptron(6, hidden_neurons, output_neurons, epsilon)
        self.sample_count = 0
        self.classes = ["Untergewicht", "Normalgewicht", "Uebergewicht"]
    
    def load_data(self, filename = "data_a_2_2016242.csv"):
        self.xs = np.zeros((0,6))
        self.targets = np.zeros((0,self.output_neurons))

        print("Lese CSV-Datei ...")
        with open(filename, newline='') as file:
            data = csv.reader(file, delimiter=';')
            next(data) # skip first line
            for row in data:
                gender = 1 if row[0] == 'w' else 0
                height = self.normalize(int(row[1]), 140, 200)
                age = self.normalize(int(row[2]), 18, 100)
                weight = self.normalize(int(row[3]), 20, 150)
                strength_sports = 1 if row[4] == "Kraftsport" else 0
                endurance_sports = 1 if row[4] == "Ausdauersport" else 0
                self.xs = np.append(self.xs, [[gender, height, age, weight, strength_sports, endurance_sports]], 0)

                if self.output_neurons == 3:
                    underweight = 1 if row[5] == self.classes[0] else 0
                    normal = 1 if row[5] == self.classes[1] else 0
                    overweight = 1 if row[5] == self.classes[2] else 0
                    self.targets = np.append(self.targets, [[underweight, normal, overweight]], 0)

                if self.output_neurons == 2:
                    underweight = 1 if row[5] == self.classes[0] else 0
                    overweight = 1 if row[5] == self.classes[2] else 0
                    self.targets = np.append(self.targets, [[underweight, overweight]], 0)

                if self.output_neurons == 1:
                    target = 0.5
                    if row[5] == self.classes[0]:
                        target = 0
                    if row[5] == self.classes[2]:
                        target = 1
                    self.targets = np.append(self.targets, [[target]], 0)

        self.sample_count = self.targets.shape[0]
        print(str(self.sample_count) + " Beispiele aus Datei " + filename + " ausgelesen.")
        
    def normalize(self, value, upper, lower):
        normalized = (value - lower) / (upper - lower)
        return min(max(normalized, 0), 1)

    def train(self, from_index = 0, to_index = 10000):
        start = np.datetime64('now')
        train_range = range(max(from_index, 0), min(to_index, self.sample_count))
        print("Trainiere mit " + str(len(train_range)) + " Beispielen ...")
        self.net.train(self.xs, self.targets, train_range)
        end = np.datetime64('now')
        print("Trainingsdauer: " + str(end - start))

    def test(self, from_index = 0, to_index = 10000, print_classes = False):
        test_range = range(max(from_index, 0), min(to_index, self.sample_count))
        count = len(test_range)
        correct_count = 0

        for i in test_range:
            expected_class = self.classify(self.targets[i])
            (_, y) = self.net.propagate(self.xs[i])
            predicted_class = self.classify(y)
            correct = expected_class == predicted_class
            if correct:
                correct_count += 1
            if print_classes:
                print(str(i) + ": " + predicted_class + (" [richtig]" if correct else " [falsch]"))

        accuracy = correct_count / count
        print("Test-Ergebnis: {0}/{1} richtig ({2:.0%})".format(correct_count, count, accuracy))

    def classify(self, y):
        if self.output_neurons == 3:
            if (y[0] > 0.5):
                return self.classes[0]
            if (y[1] > 0.5):
                return self.classes[1]
            if (y[2] > 0.5):
                return self.classes[2]
            return "ERROR"

        if self.output_neurons == 2:
            if (y[0] > 0.5):
                return self.classes[0]
            if (y[1] > 0.5):
                return self.classes[2]
            return self.classes[1]

        if self.output_neurons == 1:
            if (y[0] < 0.25):
                return self.classes[0]
            if (y[0] > 0.75):
                return self.classes[2]
            return self.classes[1]

    def sk(self, max_iter = 5000):
        mlp = MLPClassifier(hidden_layer_sizes=(self.hidden_neurons), activation='relu', solver='adam', max_iter=max_iter)
        xs_train, xs_test, targets_train, targets_test = train_test_split(self.xs, self.targets, test_size=0.30)
        mlp.fit(xs_train, targets_train)
        targets_predicted = mlp.predict(xs_test)
        print(classification_report(targets_test, targets_predicted))

class MultiplayerPerceptron:

    def __init__(self, input_dim, hidden_dim, output_dim, epsilon):
        self.INPUT = 0
        self.HIDDEN = 1
        self.OUTPUT = 2
        self.output_dim = output_dim
        weights_from_in = 2 * np.random.random((input_dim + 1, hidden_dim)) - 1
        weights_from_hidden = 2 * np.random.random((hidden_dim + 1, output_dim)) - 1
        self.weights_from = [weights_from_in, weights_from_hidden]
        self.epsilon = epsilon

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return x * (1-x)

    def propagate_layer(self, layer, levels):
        levels_ext = np.append(levels, 1) # treshold
        w = self.weights_from[layer]
        return self.sigmoid(np.dot(levels_ext, w))

    def propagate(self, x):
        h = self.propagate_layer(self.INPUT, x)
        y = self.propagate_layer(self.HIDDEN, h)
        return (h, y)

    def overall_error(self, y, target):
        return np.sum(np.square(target - y)) / 2

    def adjust_weights(self, layer, levels_left, levels_right, error, learning_rate):
        scaled_error = error * self.sigmoid_deriv(levels_right)
        delta = learning_rate * np.outer(levels_left, scaled_error)
        delta_ext = np.append(delta, np.zeros((1, error.shape[0])), 0)
        self.weights_from[layer] += delta_ext
        return scaled_error # reuse for next layer

    def backpropagate(self, x, h, y, target, learning_rate):
        error = target - y
        scaled_error = self.adjust_weights(self.HIDDEN, h, y, error, learning_rate)
        weights_from_hidden = self.weights_from[self.HIDDEN]
        hidden_error = scaled_error.dot(weights_from_hidden[:-1].T)
        self.adjust_weights(self.INPUT, x, h, hidden_error, learning_rate)

    def train(self, xs, targets, sample_range):
        count = len(sample_range)
        learning_rate = 1.0
        round = 0
        pending = 1
        while (pending > 0):
            round += 1

            # shuffle example order and decrease learning rate every 10 rounds
            if round % 10 == 0:
                sample_range = random.sample(list(sample_range), count)
                learning_rate *= 0.66

            for i in sample_range:
                x = xs[i]
                target = targets[i]
                (h, y) = self.propagate(x)
                overall_error = self.overall_error(y, target)
                while overall_error > self.epsilon:
                    self.backpropagate(x, h, y, target, learning_rate)
                    (h, y) = self.propagate(x)
                    overall_error = self.overall_error(y, target)

            pending = count
            total_error = 0
            for i in sample_range:
                (_, y) = self.propagate(xs[i])
                overall_error = self.overall_error(y, targets[i])
                if (overall_error < self.epsilon):
                    pending -= 1
                total_error += overall_error

            print("Runde " + str(round) + " - Ausstehend: " + str(pending) + " Fehler: " + str(total_error))