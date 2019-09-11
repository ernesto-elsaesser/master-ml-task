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

def classic(pes, nes):
    g = set()
    g.add("*:*:*:*:*")
    s = set()
    first = True

    print("POSITIVES")

    for pe in pes:
        if first:
            s.add(pe)
            first = False
            continue

        pex = pe.split(":")

        ng = set(g)
        for h in g:
            hx = h.split(":")
            if not contains(hx, pex):
                print("GREM " + h)
                ng.remove(h)
        g = ng

        if len(g) == 0:
            print("G COLLAPSED for " + str(pe))
            return (s,g)

        ns = set(s)
        for h in s:
            hx = h.split(":")
            if not contains(hx, pex):
                print("SREM " + h)
                ns.remove(h)
                nh = most_special_general(h, hx, pex)
                add = True
                for gh in g:
                    if more_general(nh, gh):
                        add = False
                        break
                if add:
                    print("SADD " + nh)
                    ns.add(nh)

        # remove more special duplicates
        toremove = set()
        for h1 in ns:
            redundant = False
            for h2 in ns:
                if h1 == h2:
                    continue
                if more_general(h2, h1):
                    redundant = True
                    break
            
            if redundant:
                print("SRED " + h1)
                toremove.add(h1)
                
        s = ns.difference(toremove)

        if len(s) == 0:
            print("S COLLAPSED for " + str(pe))
            return (s,g)
        
        if g == s:
            print("CONVERGED")
            return (s,g)

    print("NEGATIVES")

    for ne in nes:

        nex = ne.split(":")

        ns = set(s)
        for h in s:
            hx = h.split(":")
            if contains(hx, nex):
                print("SREM " + h)
                ns.remove(h)
        s = ns

        if len(s) == 0:
            print("S COLLAPSED for " + str(ne))
            return (s,g)

        ng = set(g)
        for h in g:
            hx = h.split(":")
            if contains(hx, nex):
                print("GREM " + h)
                ng.remove(h)
                nhs = most_general_special(h, hx, nex, pes)
                for nh in nhs:
                    add = True
                    for sh in s:
                        if more_general(sh, nh):
                            add = False
                            break
                    if add:
                        print("GADD " + nh)
                        ng.add(nh)

        # remove more general duplicates
        toremove = set()
        for h1 in ng:
            redundant = False
            for h2 in ng:
                if h1 == h2:
                    continue
                if more_general(h1, h2):
                    redundant = True
                    break
            
            if redundant:
                print("GRED " + h1)
                toremove.add(h1)
                
        g = ng.difference(toremove)

        if len(g) == 0:
            print("G COLLAPSED for " + str(ne))
            return (s,g)

        if g == s:
            print("CONVERGED")
            return (s,g)

    return (s,g)

def most_special_general(h, hx, pex):

    nhx = h.split(":")

    for i in range(0,5):
        ha = hx[i]
        pa = pex[i]

        if ha == "*":
            continue

        if ha == pa:
            continue

        nhx[i] = "*"

    nh = ":".join(nhx)
    return nh

def most_general_special(h, hx, nex, s):
    # TODO
    return update()

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
            break
    return g

def step(g, pe, pex, ne):
    nex = ne.split(":")
    ng = set(g)
    for h in g:
        hx = h.split(":")
        if contains(hx, nex):
            ng.remove(h)
            nhs = update(h, hx, nex, pe, pex)
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


def update(h, hx, nex, pe, pex):
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

def contains(hx, ex):
    for j in range(0, 5):
        if hx[j] == "*":
            continue
        if hx[j] != ex[j]:
            return False
    return True

def has_more_special(g, nhx):

    for h in g:
        hx = h.split(":")
        is_more_special = False

        for j in range(0, 5):
            if nhx[j] == "*":
                if hx[j] != "*":
                    is_more_special = True
                else:
                    continue
            elif hx[j] != nhx[x]:
                break

        if is_more_special:
            return True

    return False

