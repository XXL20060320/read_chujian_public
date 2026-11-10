# -*- coding: utf-8 -*-
"""基线：直接用像素当特征，看"形状相似"到底能做到多少。
   如果像素 kNN 明显强于 ResNet18 冻结特征，说明问题出在特征，不在数据。"""
import os
import numpy as np, pandas as pd
from PIL import Image, ImageOps
import model as M

df = pd.read_csv("data/splits.csv")
df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
cnt = df.label.value_counts()
keep = [l for l, c in cnt.items() if c >= 20]
tr = df[(df.split_doc == "train") & df.label.isin(keep)]
te = df[(df.split_doc == "test") & df.label.isin(keep)]
print(f"类别 {len(keep)}；训练 {len(tr)}，测试 {len(te)}")

tr = tr.groupby("label", group_keys=False).head(15)
te = te.groupby("label", group_keys=False).head(5)


def vec(p, s):
    im = Image.open(p).convert("L")
    im = ImageOps.autocontrast(im).resize((s, s), Image.BILINEAR)
    a = np.asarray(im, dtype="float32") / 255.0
    return a.ravel()


for s in (32, 48):
    Xtr = np.stack([vec(p, s) for p in tr.path])
    Xte = np.stack([vec(p, s) for p in te.path])
    ytr, yte = tr.label.values, te.label.values
    A = M.l2norm(Xtr - Xtr.mean(1, keepdims=True))
    B = M.l2norm(Xte - Xte.mean(1, keepdims=True))
    sims = B @ A.T
    order = np.argsort(-sims, 1)[:, :5]
    for k in (1, 5):
        hit = np.array([yte[i] in ytr[order[i, :k]] for i in range(len(yte))])
        print(f"{s}x{s} 像素 kNN top{k}: {hit.mean():.4f}")
