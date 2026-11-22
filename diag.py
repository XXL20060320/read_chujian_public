# -*- coding: utf-8 -*-
"""诊断：kNN / 原型分类 / 线性头 在不同类别频次下的表现"""
import json, os
import numpy as np
import pandas as pd
import torch
import model as M

B = "bundle"
df = pd.read_csv("data/splits.csv")
df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
df = df.groupby("label", group_keys=False).head(40).reset_index(drop=True)

tr = df[df.split_doc == "train"].reset_index(drop=True)
va = df[df.split_doc == "val"].reset_index(drop=True)
te = df[df.split_doc == "test"].reset_index(drop=True)

ftr = np.load(f"{B}/feat_train.npy")
fva = np.load(f"{B}/feat_val.npy")
fte = np.load(f"{B}/feat_test.npy")
assert len(ftr) == len(tr) and len(fte) == len(te), (len(ftr), len(tr), len(fte), len(te))

with open(f"{B}/meta.json", encoding="utf-8") as _fh:
    labels = json.load(_fh)["labels"]
lab2id = {l: i for i, l in enumerate(labels)}

mte = te.label.isin(lab2id).values
yte = np.array([lab2id[l] for l in te.label[mte]])
Fte = fte[mte]

# 1) kNN
idx = M.l2norm(ftr)
q = M.l2norm(Fte)
sims = q @ idx.T                      # (n, 26964)
order = np.argsort(-sims, axis=1)[:, :10]
tr_labels = tr.label.values
for k in (1, 5, 10):
    hit = np.array([yte[i] in [lab2id.get(tr_labels[j], -1) for j in order[i, :k]]
                    for i in range(len(yte))])
    print(f"kNN top{k}: {hit.mean():.4f}")

# 2) 原型（类均值）
proto, plabels = [], []
for l, g in tr.groupby("label"):
    proto.append(ftr[g.index.values].mean(0))
    plabels.append(l)
proto = M.l2norm(np.stack(proto))
ps = M.l2norm(Fte) @ proto.T
porder = np.argsort(-ps, axis=1)
pl = np.array(plabels)
pids = np.array([lab2id.get(x, -1) for x in pl])
for k in (1, 5):
    hit = np.array([yte[i] in pids[porder[i, :k]] for i in range(len(yte))])
    print(f"原型 top{k}: {hit.mean():.4f}")

# 3) 线性头（已有）
with open(f"{B}/meta.json", encoding="utf-8") as _fh:
    meta = json.load(_fh)
print("线性头 top1/top5:", meta["test_top1"], meta["test_top5"])

# 4) 按频次分桶
cnt = tr.label.value_counts()
print("\n频次桶  n_class  样本数   kNN-1   原型-1")
buckets = [(5, 9), (10, 19), (20, 39), (40, 80)]
knn1 = np.array([yte[i] == lab2id.get(tr_labels[order[i, 0]], -1) for i in range(len(yte))])
proto1 = np.array([yte[i] == pids[porder[i, 0]] for i in range(len(yte))])
# 简单按测试样本所属类在训练集里的频次分桶
freq = np.array([cnt.get(l, 0) for l in te.label[mte]])
for lo, hi in buckets:
    m = (freq >= lo) & (freq <= hi)
    if m.sum() == 0:
        continue
    print(f"{lo:>2}-{hi:<3} {len(set(te.label[mte][m])):6d} {m.sum():7d}   "
          f"{knn1[m].mean():.3f}   {proto1[m].mean():.3f}")
