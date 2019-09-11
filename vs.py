import csv
import numpy as np

classes = ["Untergewicht", "Normalgewicht", "Uebergewicht"]
genders = ["w", "m"]
heights = ["<40", "40-55", "56-70", "71-85", "86-90", "91-105", ">105"]
ages = ["18-25", "26-39", "40-59", "60-79", "80+"]
weights = ["<150", "150-159", "160-169", "170-179", "180-189", ">190"]
sports = ["keinSport", "Kraftsport", "Ausdauersport"]
options = [genders, heights, ages, weights, sports]

def load(filename = "data_a_2_2016242.csv"):
    p = []
    n = []

    print("Lese CSV-Datei ...")
    with open(filename, newline='') as file:
        data = csv.reader(file, delimiter=';')
        next(data) # skip first line
        for row in data:
            gender = row[0]

            h = int(row[1])
            if h < 150:
                height = "<150"
            elif h < 160:
                height = "150-159"
            elif h < 170:
                height = "160-169"
            elif h < 180:
                height = "170-179"
            elif h < 190:
                height = "180-189"
            else:
                height = ">190"

            a = int(row[2])
            if a < 26:
                age = "18-25"
            elif a < 40:
                age = "26-39"
            elif a < 60:
                age = "40-59"
            elif a < 80:
                age = "60-79"
            else:
                age = "80+"

            w = int(row[3])
            if w < 40:
                weight = "<40"
            elif w < 56:
                weight = "40-55"
            elif w < 71:
                weight = "56-70"
            elif w < 86:
                weight = "71-85"
            elif w < 91:
                weight = "86-90"
            elif w < 106:
                weight = "91-105"
            else:
                weight = ">105"

            sport = row[4]
            example = ":".join([gender, height, age, weight, sport])

            if row[5] == classes[0]: # classes[2]:
                p.append(example)
            else:
                n.append(example)

    return (p,n)

def merged_stars(pes, nes):
    g = set()
    for pe in pes:
        ng = star(pe, nes)
        g = g.union(ng)
    return g

def star(pe, nes):
    pex = pe.split(":")
    g = set()
    g.add("*:*:*:*:*")
    for ne in nes:
        g = step(g, pe, pex, ne)
        if len(g) == 0:
            break # collapsed
    return g

def step(g, pe, pex, ne):
    nex = ne.split(":")
    ng = set(g)
    for h in g:
        hx = h.split(":")
        if contains(hx, nex):
            ng.remove(h)
            nhs = most_general_specialization(h, hx, nex, pe, pex)
            for nh in nhs:
                ng.add(nh)

    # remove more general duplicates
    toremove = redundancies(ng)
    ng = ng.difference(toremove)

    return ng

def redundancies(g):
    hs = set()

    for h1 in g:
        redundant = False
        for h2 in g:
            if h1 == h2:
                continue
            if more_general(h1, h2):
                redundant = True
                break
        
        if redundant:
            hs.add(h1)

    return hs

def more_general(h1, h2):
    h1x = h1.split(":")
    h2x = h2.split(":")

    more_general = False
    for i in range(0, 5):
        if h1x[i] == h2x[i]:
            continue
        if h1x[i] == "*" and h2x[i] != "*":
            more_general = True
        else:
            more_general = False
            break

    return more_general


def most_general_specialization(h, hx, nex, pe, pex):
    nhs = []

    # find most general specializations
    for i in range(0,5):
        na = nex[i]
        ha = hx[i]
        pa = pex[i]

        if ha != "*":
            continue

        if pa == na:
            continue

        nhx = h.split(":")
        nhx[i] = pa
        nh = ":".join(nhx)

        if nh != pe:
           nhs.append(nh)

    return nhs

def contains(hx, ex):
    for j in range(0, 5):
        if hx[j] == "*":
            continue
        if hx[j] != ex[j]:
            return False
    return True

def any_contains(g, e):
    ex = e.split(":")
    for h in g:
        hx = h.split(":")
        if contains(hx, ex):
            return True
    return False

def all_contain(g, e):
    ex = e.split(":")
    for h in g:
        hx = h.split(":")
        if not contains(hx, ex):
            return False
    return True