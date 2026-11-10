# -*- coding: utf-8 -*-
"""简牍识字 —— 特征提取与线性探针（CPU 友好）

思路：ImageNet 预训练网络冻结，只当"特征提取器"；在上面接一个线性分类头，
再用同一批特征做 kNN 形近字检索。CPU 上几分钟到半小时能跑完，不必微调。
"""
import os
import numpy as np
import torch
import torch.nn as nn

BACKBONES = {
    "resnet18": ("resnet18", "ResNet18_Weights", 512),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights", 576),
    "resnet34": ("resnet34", "ResNet34_Weights", 512),
}


def build(backbone="resnet18"):
    import torchvision.models as tvm

    name, wname, dim = BACKBONES[backbone]
    ctor = getattr(tvm, name)
    weights = getattr(tvm, wname).IMAGENET1K_V1
    net = ctor(weights=weights)

    # 去掉分类层，保留池化后的向量
    if hasattr(net, "fc"):
        net.fc = nn.Identity()
    elif hasattr(net, "classifier"):
        net.classifier = nn.Identity()

    for p in net.parameters():
        p.requires_grad = False
    net.eval()

    tf = weights.transforms()
    return net, tf, dim


@torch.no_grad()
def embed_paths(net, tf, paths, batch=64, log_every=2000):
    from PIL import Image

    out = []
    bad = 0
    n = len(paths)
    for i in range(0, n, batch):
        chunk = paths[i:i + batch]
        tens = []
        for p in chunk:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                bad += 1
                img = Image.new("RGB", (224, 224))
            tens.append(tf(img))
        x = torch.stack(tens)
        f = net(x)
        f = torch.flatten(f, 1)
        out.append(f.numpy().astype("float32"))
        if log_every and (i % log_every == 0 or i + batch >= n):
            print(f"    {min(i + batch, n)}/{n}", flush=True)
    if bad:
        print(f"  [warn] {bad}/{n} 张图无法读取，已用空白图代替", flush=True)
    return np.concatenate(out, 0) if out else np.zeros((0, 0), "float32")


class LinearHead(nn.Module):
    def __init__(self, dim, ncls):
        super().__init__()
        self.fc = nn.Linear(dim, ncls)

    def forward(self, x):
        return self.fc(x)


def l2norm(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def train_head(x, y, ncls, epochs=30, lr=1e-2, wd=1e-4, bs=256, seed=0):
    torch.manual_seed(seed)
    head = LinearHead(x.shape[1], ncls)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y).long()
    n = xt.shape[0]
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(head(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs} loss={tot/cnt:.4f}", flush=True)
    head.eval()
    return head


@torch.no_grad()
def predict(head, x, bs=4096):
    if len(x) == 0:
        return np.zeros((0, head.fc.out_features), "float32")
    out = []
    for i in range(0, x.shape[0], bs):
        out.append(torch.softmax(head(torch.from_numpy(x[i:i + bs])), -1).numpy())
    return np.concatenate(out, 0)


def topk_acc(prob, y, k=(1, 5)):
    order = np.argsort(-prob, axis=1)
    res = {}
    for kk in k:
        hit = (order[:, :kk] == y[:, None]).any(1)
        res[kk] = float(hit.mean())
    return res
