import nn

n = nn.WeightClassifier()
n.load_data()
n.test(0,10, True)
n.train(0,10)
n.save_weights("test.net")
n.load_weights("test.net")
n.test(0,10, True)