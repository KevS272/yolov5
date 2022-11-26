#!/usr/bin/env python

import glob
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



switcher_path = glob.glob("runs/val/switcher*/curves.npy")[-1]
ensemble_path = glob.glob("runs/val/ensemble*/curves.npy")[-1]
mono_path = glob.glob("runs/val/mon*/curves.npy")[-1]

def plotcurves():
    s_dic = np.load(switcher_path, allow_pickle=True).item(0)
    s_px = np.array(s_dic['px'])
    s_py = np.array(s_dic['pr'])
    s_f1 = np.array(s_dic['f1'])
    s_p  = np.array(s_dic['p'])
    s_r  = np.array(s_dic['r'])

    m_dic = np.load(mono_path, allow_pickle=True).item(0)
    m_px = np.array(m_dic['px'])
    m_py = np.array(m_dic['pr'])
    m_f1 = np.array(m_dic['f1'])
    m_p  = np.array(m_dic['p'])
    m_r  = np.array(m_dic['r'])

    e_dic = np.load(ensemble_path, allow_pickle=True).item(0)
    e_px = np.array(e_dic['px'])
    e_py = np.array(e_dic['pr'])
    e_f1 = np.array(e_dic['f1'])
    e_p  = np.array(e_dic['p'])
    e_r  = np.array(e_dic['r'])


    s_py = np.stack(s_py, axis=1)
    m_py = np.stack(m_py, axis=1)
    e_py = np.stack(e_py, axis=1)

    s_f1 = np.stack(s_f1, axis=1)
    m_f1 = np.stack(m_f1, axis=1)
    e_f1 = np.stack(e_f1, axis=1)

    s_p = np.stack(s_p, axis=1)
    m_p = np.stack(m_p, axis=1)
    e_p = np.stack(e_p, axis=1)

    s_r = np.stack(s_r, axis=1)
    m_r = np.stack(m_r, axis=1)
    e_r = np.stack(e_r, axis=1)

    plt.title("PR curves")
    plt.plot(s_px, s_py.mean(1), label="switch")
    plt.plot(m_px, m_py.mean(1), label="mono")
    plt.plot(e_px, e_py.mean(1), label="ens")
    plt.legend()
    plt.show()

    plt.title("F1 curves")
    plt.plot(s_px, s_f1.mean(1), label="switch")
    plt.plot(m_px, m_f1.mean(1), label="mono")
    plt.plot(e_px, e_f1.mean(1), label="ens")
    plt.legend()
    plt.show()

    plt.title("P curves")
    plt.plot(s_px, s_p.mean(1), label="switch")
    plt.plot(m_px, m_p.mean(1), label="mono")
    plt.plot(e_px, e_p.mean(1), label="ens")
    plt.legend()
    plt.show()

    plt.title("R curves")
    plt.plot(s_px, s_r.mean(1), label="switch")
    plt.plot(m_px, m_r.mean(1), label="mono")
    plt.plot(e_px, e_r.mean(1), label="ens")
    plt.legend()
    plt.show()

printmetrics()
plotcurves()