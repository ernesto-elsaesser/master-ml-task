import numpy as np

class FeedForwardNetwork:

    def __init__(self, in_count, hidden_count, out_count, epsilon = 0.01):
        self.syn0 = 2 * np.random.random((in_count + 1,hidden_count)) - 1
        self.syn1 = 2 * np.random.random((hidden_count + 1,out_count)) - 1
        self.epsilon = epsilon

    @staticmethod
    def transfer(x):   # sigmoid function
        return (1/(1 + np.exp(-x)))

    @staticmethod
    def transfer_deriv(x):   # derived sigmoid function
        return x*(1-x)
    
    def feed(self, x, y, backpropagate):
        l0 = np.append(x, 1)
        l1p = transfer(np.dot(l0, self.syn0))
        l1 = np.append(l1p, 1)
        l2 = transfer(np.dot(l1, self.syn1))
        l2_error = y - l2

        l2_square_error = l2_error ^ 2
        error_sum = np.sum(l2_square_error) / 2

        if backpropagate and error_sum < self.epsilon:
            l2_delta = l2_error * transfer_deriv(l2)
            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error * transfer_deriv(l1)
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += l0.T.dot(l1_delta)

        return error_sum

    def train(self, xs, ys, range_from, range_to):
        sample_range = range(range_from, range_to)
        self.train(xs, ys, sample_range)

    def train(self, xs, ys, sample_range):
        sample_range = range(range_from, range_to)
        round = 0
        wrong_classifications = True
        while (wrong_classifications):
            round += 1

            for i in sample_range:
                x = xs[i]
                y = ys[i]
                while self.feed(x, y, True) < self.epsilon:
                    continue

            (correct, total_error) = self.test(xs, ys, sample_range)
            wrong_classifications = correct < len(sample_range)

            print("Runde " + str(round) + " - Korrekte: " + str(correct) + " Fehler : " + str(total_error))

    def test(self, xs, ys, sample_range):
        correct = 0
        total_error = 0

        for i in sample_range:
            error = self.feed(xs[i], ys[i], False)
            total_error += error
            if (error < self.epsilon):
                correct += 1

        return (correct, total_error)
    
    def export(self, filename = "trained.net"):
        with open(filename, 'w') as file:
            file.write(repr(self.syn0).replace("\n", "") + "\n")
            file.write(repr(self.syn1).replace("\n", "") + "\n")

    def import_(self, filename = "trained.net"):
        with open(filename, 'r') as file:
            self.syn0 = eval(file.readline())
            self.syn1 = eval(file.readline())