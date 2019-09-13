# TM40507 Pruefungsleistung von Ernesto Elsaesser (Matrikelnummer 2016242)

import classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
# documentation: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

db = classifier.WeightClassifier() # reuse CSV import
db.load_data()

mlp = MLPClassifier(hidden_layer_sizes=(12,12), activation="relu", solver="sgd", momentum=0.9)
mlp.fit(db.xs[0:7000], db.targets[0:7000])

targets_predicted = mlp.predict(db.xs[7000:10000])
print(classification_report(db.targets[7000:10000], targets_predicted))