# TM40507 Pruefungsleistung von Ernesto Elsaesser (Matrikelnummer 2016424)

import classifier

wc = classifier.WeightClassifier()

train_filename = input("Pfad zu den Trainingsdaten (CSV): ")
wc.load_data(train_filename)
wc.train()

test_filename = input("Pfad zu den Testdaten (CSV): ")
wc.load_data(test_filename)
verbose = input("Klassen ausgeben (j/n)? ") == "j"
wc.test(verbose = verbose)
