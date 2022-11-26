#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

filepath = "runs/val/exp25/"

def printmetrics():
    stats = []

    with open(filepath + "metrics.npy", 'rb') as f:
        for i in range(4):
            stats.append(np.load(f))
        tp = np.load(f)
        fp = np.load(f)
        p  = np.load(f)
        r  = np.load(f)
        f1 = np.load(f)
        ap = np.load(f)
        ap_class = np.load(f)
        mp = np.load(f)
        mr = np.load(f)
        map50 = np.load(f)
        map = np.load(f)

    print("---stats---")
    for l in stats:
        print("---")
        print(l)
        print(l.shape)
    print("---end-stats---")

    print("\nTP:")
    print(tp)
    print("\nFP:")
    print(fp)
    print("\nP:")
    print(p)
    print("\nR:")
    print(r)
    print("\nF1:")
    print(f1)
    print("\nAP:")
    print(ap)
    print("\nAPC:")
    print(ap_class)
    print("\nMP:")
    print(mp)
    print("\nMR:")
    print(mr)
    print("\nmAP50:")
    print(map50)
    print("\nmAP:")
    print(map)



def plotcurves():
    dic = np.load(filepath + "curves.npy", allow_pickle=True).item(0)
    px = np.array(dic['px'])
    py = np.array(dic['pr'])
    f1 = np.array(dic['f1'])
    p  = np.array(dic['p'])
    r  = np.array(dic['r'])

    py = np.stack(py, axis=1)

    plt.plot(px, py.mean(1))
    plt.show()

printmetrics()
plotcurves()