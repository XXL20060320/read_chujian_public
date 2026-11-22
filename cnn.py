# -*- coding: utf-8 -*-
"""楚简字形专用的小 CNN（CPU 可训）

诊断结论：ImageNet 冻结特征的"批次探针"准确率 95%（15 类），说明特征主要在编码
扫描风格而不是字形；像素模板 kNN 与它持平。故改为在 64x64 灰度图上从零训练。

为提速：先把所有图片一次性缩放缓存成 uint8 数组，训练时只读内存。
X_train.npy 同时供 app.py 显示形近字缩略图，交付时不可裁去。
"""
import argparse, hashlib, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

import model as M

SIZE = 64


def file_sig(path):
    """数据文件内容指纹，用于判断图像缓存是否仍然有效。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class GlyphNet(nn.Module):
    def __init__(self, ncls, ch=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.b1, self.b2 = blk(1, ch), blk(ch, ch * 2)
        self.b3, self.b4 = blk(ch * 2, ch * 4), blk(ch * 4, ch * 8)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(ch * 8, ncls)

    def features(self, x):
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.fc(self.drop(self.features(x)))


def to_array(paths, size=SIZE):
    out = np.zeros((len(paths), size, size), dtype="uint8")
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L")
        im = ImageOps.autocontrast(im).resize((size, size), Image.Resampling.BILINEAR)
        out[i] = np.asarray(im, dtype="uint8")
    return out


class Arr(Dataset):
    def __init__(self, X, y, train=False):
        self.X, self.y, self.train = X, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].astype("float32") / 255.0)[None]
        x = (x - 0.5) / 0.5
        if self.train:
            ang = float(torch.empty(1).uniform_(-7, 7))
            sc = float(torch.empty(1).uniform_(0.92, 1.06))
            tx, ty = (float(torch.empty(1).uniform_(-0.04, 0.04)) for _ in range(2))
            a = np.radians(ang)
            tf = torch.tensor([[[np.cos(a) / sc, -np.sin(a) / sc, tx],
                                [np.sin(a) / sc, np.cos(a) / sc, ty]]], dtype=torch.float)
            grid = F.affine_grid(tf, x.unsqueeze(0).shape, align_corners=False)
            x = F.grid_sample(x.unsqueeze(0), grid, align_corners=False, padding_mode="border")[0]
        return x, int(self.y[i])


@torch.no_grad()
def collect(model, loader):
    model.eval()
    feats, logits, ys = [], [], []
    for x, y in loader:
        f = model.features(x)
        feats.append(f.numpy())
        logits.append(model.fc(f).numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--out", default="bundle_cnn")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=None,
                    help="拒识阈值；不指定时取验证集置信度的 10 分位数")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out, exist_ok=True)
    splits_sig = file_sig(args.splits)

    df = pd.read_csv(args.splits)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        raise SystemExit(
            f"{args.splits} 里的 path 在本机全部不存在（该列是相对路径 glyphs/<标签>/<文件名>，由 prep.py 生成）。\n"
            "请先运行：python prep.py --root <本机 CHUBS/glyphs 目录> --out data")
    cnt = df.label.value_counts()
    df = df[df.label.isin([l for l, c in cnt.items() if c >= args.min_count])]
    df = df.groupby("label", group_keys=False).head(args.per_class).reset_index(drop=True)

    tr = df[df.split_doc == "train"].reset_index(drop=True)
    va = df[df.split_doc == "val"].reset_index(drop=True)
    te = df[df.split_doc == "test"].reset_index(drop=True)
    labels = sorted(tr.label.unique())
    lab2id = {l: i for i, l in enumerate(labels)}
    va = va[va.label.isin(lab2id)].reset_index(drop=True)
    te = te[te.label.isin(lab2id)].reset_index(drop=True)
    print(f"类别 {len(labels)}｜训练 {len(tr)} 验证 {len(va)} 测试 {len(te)}"
          f"｜seed={args.seed}", flush=True)

    def cache(name, d):
        f = os.path.join(args.out, f"X_{name}.npy")
        mf = f + ".meta.json"
        sig = {"splits": splits_sig, "min_count": args.min_count,
               "per_class": args.per_class, "size": SIZE, "n": len(d),
               "first": str(d.path.iloc[0]) if len(d) else "",
               "last": str(d.path.iloc[-1]) if len(d) else ""}
        if os.path.exists(f):
            try:
                if os.path.exists(mf):
                    with open(mf, encoding="utf-8") as fh:
                        if json.load(fh) == sig:
                            return np.load(f)
                print(f"  [缓存失效] {name}: 数据或参数已变，重新生成", flush=True)
            except (OSError, ValueError):
                print(f"  [缓存元信息损坏] {name}: 重新生成", flush=True)
        t0 = time.time()
        X = to_array(d.path.tolist())
        np.save(f, X)
        with open(mf, "w", encoding="utf-8") as fh:
            json.dump(sig, fh, ensure_ascii=False, indent=2)
        print(f"  缓存 {name}: {X.shape} {time.time()-t0:.0f}s", flush=True)
        return X

    Xtr, Xva, Xte = cache("train", tr), cache("val", va), cache("test", te)
    ytr = np.array([lab2id[l] for l in tr.label])
    yva = np.array([lab2id[l] for l in va.label])
    yte = np.array([lab2id[l] for l in te.label])

    gen = torch.Generator().manual_seed(args.seed)
    dl = lambda X, y, sh=False, bs=256: DataLoader(
        Arr(X, y, sh), batch_size=bs, shuffle=sh, num_workers=0,
        generator=gen if sh else None)

    net = GlyphNet(len(labels), args.ch)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, t_all = 0.0, time.time()
    for ep in range(args.epochs):
        net.train(); t0, tot, n = time.time(), 0.0, 0
        for x, y in dl(Xtr, ytr, True):
            opt.zero_grad(); out = net(x); loss = lossf(out, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        _, lg, yy = collect(net, dl(Xva, yva))
        acc = M.topk_acc(torch.softmax(torch.from_numpy(lg), -1).numpy(), yy)
        print(f"ep{ep+1:02d} loss={tot/n:.3f} val@1={acc[1]:.3f} val@5={acc[5]:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if acc[1] >= best:
            best = acc[1]
            torch.save(net.state_dict(), os.path.join(args.out, "model.pt"))

    state = torch.load(os.path.join(args.out, "model.pt"), map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    fva, lg, yy = collect(net, dl(Xva, yva))
    pva = torch.softmax(torch.from_numpy(lg), -1).numpy()
    _, lg, yy = collect(net, dl(Xte, yte))
    acc = M.topk_acc(torch.softmax(torch.from_numpy(lg), -1).numpy(), yy)
    print(f"测试 top1={acc[1]:.4f} top5={acc[5]:.4f}（总用时 {(time.time()-t_all)/60:.0f} 分）")

    if args.threshold is not None:
        thr, thr_src = float(args.threshold), "cli"
    else:
        thr, thr_src = float(np.quantile(pva.max(1), 0.10)), "val_q10"
    ftr, _, _ = collect(net, dl(Xtr, ytr))
    np.save(os.path.join(args.out, "index.npy"), M.l2norm(ftr).astype("float32"))
    for col, name in [("path", "paths"), ("label", "labels"), ("batch", "batch"),
                      ("doc", "doc"), ("slip", "slip")]:
        with open(os.path.join(args.out, f"index_{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(tr[col].astype(str).tolist(), fh, ensure_ascii=False)
    meta = {"kind": "cnn", "ch": args.ch, "labels": labels, "threshold": thr,
            "threshold_source": thr_src, "seed": args.seed, "size": SIZE,
            "test_top1": float(acc[1]), "test_top5": float(acc[5]),
            # 溯源：便于日后判断某个 model.pt 由哪版脚本、什么时候产出
            "script_sha": file_sig(os.path.abspath(__file__)),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}（threshold={thr:.4f}, 来源 {thr_src}）")


if __name__ == "__main__":
    main()
