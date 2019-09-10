# Ernesto Elsaesser - Matrikelnummer 2016424 - fuer Python 3

import nn

classifier = nn.WeightClassifier()

answer = input("Vortrainierte Gewichte laden (j/n)? ")
if answer == "j":
    classifier.load_weights()
else:
    train_filename = input("Pfad zu den Trainingsdaten (CSV): ")
    classifier.load_data(train_filename)
    sample_count = int(input("Zahl zu nutzender Trainingsbeispiele: "))
    classifier.train(to_index = sample_count)

test_filename = input("Pfad zu den Testdaten (CSV): ")
classifier.load_data(test_filename)
answer = input("Klassen ausgeben (j/n)? ")
print_classes = answer == "j"
classifier.test(print_classes = print_classes)
