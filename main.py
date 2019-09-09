# Lösung von Matrikelnummer 2016424

import train
import math # needed?


# Eingabe einer Datei von Trainings- daten ein Konzept erlernt und anschließend bei Eingabe einer Datei von Testdaten eine Klassifikation vornimmt und ausgibt.


    trainIn = numpy.zeros((number,2))
    teach = numpy.zeros((number,1))

    step = 2 * math.pi / number
    for i in range(0,number):
        x = i * step
        y1 = (math.sin(x) + 1) / 2
        y2 = (math.sin(x+step) + 1) / 2
        y3 = (math.sin(x+step+step) + 1) / 2
        trainIn[i][0] = y1
        trainIn[i][1] = y2
        teach[i][0] = y3
        print("["+str(i)+"] "+ str(y1) + " "+ str(y2) + " -> "+ str(y3))