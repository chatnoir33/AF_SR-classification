from glob import glob
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

np.random.seed(seed=0)
N = 100

'''
SR vs. SR
'''
files = sorted(glob('SR/*.txt'))
sr = np.empty((0,520))
for file in files: sr = np.vstack((sr, np.loadtxt(file)))

idx_sr = np.random.choice(np.arange(len(sr)), size=N, replace=False)
sr = sr[idx_sr]

cnt_geq = 0
cnt_ls = 0
test= []
for i in range(N):
    for j in range(i+1, N):
        if i != j:
            if ks_2samp(sr[i], sr[j])[1] > 0.05: cnt_geq += 1
            else: cnt_ls += 1

print("SR vs. SR: cnt_geq=",cnt_geq, "cnt_ls=", cnt_ls, "cnt_geq/(cnt_geq+cnt_ls)=",cnt_geq/(cnt_geq+cnt_ls))


'''
AF vs. AF
'''
files = sorted(glob('AF/*.txt'))
af = np.empty((0,520))
for file in files: af = np.vstack((af, np.loadtxt(file)))

idx_af = np.random.choice(np.arange(len(af)), size=N, replace=False)
af = af[idx_af]

cnt_geq = 0
cnt_ls = 0
for i in range(N):
    for j in range(i+1, N):
        if i != j:
            if ks_2samp(af[i], af[j])[1] > 0.05: cnt_geq += 1
            else: cnt_ls += 1

print("AF vs. AF: cnt_geq=",cnt_geq, "cnt_ls=", cnt_ls, "cnt_geq/(cnt_geq+cnt_ls)=",cnt_geq/(cnt_geq+cnt_ls))


'''
SR vs. AF
'''
cnt_geq = 0
cnt_ls = 0
for i in range(N):
    for j in range(N):
        if ks_2samp(sr[i], af[j])[1] >= 0.05: cnt_geq += 1
        else: cnt_ls += 1

print("SR vs. AF: cnt_geq=",cnt_geq, "cnt_ls=", cnt_ls, "cnt_geq/(cnt_geq+cnt_ls)=",cnt_geq/(cnt_geq+cnt_ls))
