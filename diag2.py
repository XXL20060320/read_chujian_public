# -*- coding: utf-8 -*-
"""假设检验：特征是不是主要编码了"批次/扫描风格"而不是字形？
   并测试：先二值化再提特征，能否削弱批次信号。"""
import numpy as np, pandas as pd, torch, json, os
import torch.nn as nn
from PIL import Image, ImageOps
import model as M

torch.set_num_threads(os.cpu_count() or 4)
df = pd.read_csv("data/splits.csv")
df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
df = df.groupby("label", group_keys=False).head(40).reset_index(drop=True)
tr = df[df.split_doc == "train"].reset_index(drop=True)
te = df[df.split_doc == "test"].reset_index(drop=True)
Ftr = np.load("bundle/feat_train.npy")

# ---- 1) 批次探针：能不能从特征认出"这张图来自哪一批" ----
batches = sorted(tr.batch.unique())
b2i = {b: i for i, b in enumerate(batches)}
yb = np.array([b2i[b] for b in tr.batch])
head = M.train_head(Ftr, yb, len(batches), epochs=30)
pb = M.predict(head, Ftr)
print(f"批次探针（{len(batches)} 类）训练集 top1 = {M.topk_acc(pb, yb)[1]:.4f}")
print("  （若明显高于 1/15≈0.067，说明特征主要编码批次风格）")

# ---- 2) 二值化前后对比 ----
net, tf, dim = M.build("resnet18")


def otsu(im):
    g = np.asarray(im.convert("L"))
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    tot = hist.sum()
    w = np.cumsum(hist)
    s = np.cumsum(hist * np.arange(256))
    mu = s[-1] / tot
    denom = w * (tot - w)
    denom[denom == 0] = 1
    var = (mu * w - s) ** 2 / denom
    t = int(np.argmax(var))
    return g > t


def prep_bin(path):
    im = Image.open(path).convert("L")
    im = ImageOps.autocontrast(im)
    m = otsu(im)
    out = np.where(m, 0, 255).astype("uint8")   # 笔画黑、底白
    return Image.fromarray(out).convert("RGB")


def embed_apply(paths, fn):
    feats = []
    for i in range(0, len(paths), 64):
        ch = paths[i:i + 64]
        x = torch.stack([tf(fn(p)) for p in ch])
        with torch.no_grad():
            feats.append(torch.flatten(net(x), 1).numpy())
    return np.concatenate(feats, 0).astype("float32")


# 采样：取训练/测试共有标签的前 300 类，每类 20 张训练 / 6 张测试
common = sorted(set(tr.label) & set(te.label))[:300]
sub_tr = pd.concat([tr[tr.label == l].head(20) for l in common])
sub_te = pd.concat([te[te.label == l].head(6) for l in common])

ftr_bin = embed_apply(sub_tr.path.values, prep_bin)
fte_bin = embed_apply(sub_te.path.values, prep_bin)
ftr_raw = embed_apply(sub_tr.path.values, lambda p: Image.open(p).convert("RGB"))
fte_raw = embed_apply(sub_te.path.values, lambda p: Image.open(p).convert("RGB"))

ytr = np.array(sub_tr.label.values)
yte = np.array(sub_te.label.values)


def knn_acc(a, b, ya, yb, k=5):
    A = M.l2norm(a); B = M.l2norm(b)
    sims = B @ A.T
    order = np.argsort(-sims, 1)[:, :k]
    hit = np.array([yb[i] in ya[order[i]] for i in range(len(yb))])
    return hit.mean()


for name, (a, b) in [("原图", (ftr_raw, fte_raw)), ("二值化", (ftr_bin, fte_bin))]:
    print(f"{name}: kNN top1={knn_acc(a,b,ytr,yte,1):.4f} top5={knn_acc(a,b,ytr,yte,5):.4f}")

