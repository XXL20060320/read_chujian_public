# -*- coding: utf-8 -*-
"""补充评测：切分协议对照 / 批次探针 / 检索命中率 / 误判案例。

依赖已有的训练产物（bundle/、bundle_cnn/、data/splits.csv），不重新训练 CNN。
用法：python eval_extras.py    输出：eval_extras.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import model as M
from cnn import GlyphNet

torch.set_num_threads(os.cpu_count() or 4)
OUT = {}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--skip-legacy", action="store_true",
                help="不复算 ResNet 第一版；精简交付包没有 bundle/ 时使用")
ap.add_argument("--output", default="eval_extras.json", help="评测 JSON 输出路径")
args = ap.parse_args()
legacy_files = [os.path.join("bundle", f"feat_{s}.npy") for s in ("train", "val", "test")]
HAS_LEGACY = not args.skip_legacy and all(os.path.exists(f) for f in legacy_files)


def load_splits():
    # 切分对照与 CNN 复算都使用随包缓存，不读取原始字图。
    # 因此不能按 path 是否存在过滤，否则换机器后会得到空数据集。
    return pd.read_csv("data/splits.csv").reset_index(drop=True)


# ------------------------------------------------------- 1-2. 第一版切分对照与批次探针
df = load_splits()
if HAS_LEGACY:
    F = np.zeros((len(df), 512), "float32")
    for sp, fn in [("train", "feat_train.npy"), ("val", "feat_val.npy"),
                   ("test", "feat_test.npy")]:
        idx = df.index[df.split_doc == sp].to_numpy()
        F[idx] = np.load(os.path.join("bundle", fn))
    print("[1] ResNet 特征矩阵", F.shape, flush=True)

    def linear_probe(col):
        tr = df[df[col] == "train"]
        te = df[df[col] == "test"]
        labels = sorted(tr.label.unique())
        l2i = {l: i for i, l in enumerate(labels)}
        ytr = np.array([l2i[l] for l in tr.label])
        m = te.label.isin(l2i).values
        yte = np.array([l2i[l] for l in te.label[m]])
        head = M.train_head(F[tr.index.to_numpy()], ytr, len(labels), epochs=30)
        p = M.predict(head, F[te.index.to_numpy()][m])
        a = M.topk_acc(p, yte)
        return {"n_classes": len(labels), "n_train": int(len(tr)),
                "n_test_closed": int(m.sum()),
                "top1": round(float(a[1]), 4), "top5": round(float(a[5]), 4)}

    OUT["resnet_by_doc"] = linear_probe("split_doc")
    print("    按篇分组:", OUT["resnet_by_doc"], flush=True)
    OUT["resnet_by_rand"] = linear_probe("split_rand")
    print("    随机切分:", OUT["resnet_by_rand"], flush=True)

    tr = df[df.split_doc == "train"]
    batches = sorted(tr.batch.unique())
    b2i = {b: i for i, b in enumerate(batches)}
    yb = np.array([b2i[b] for b in tr.batch])
    bhead = M.train_head(F[tr.index.to_numpy()], yb, len(batches), epochs=30)
    pb = M.predict(bhead, F[tr.index.to_numpy()])
    OUT["batch_probe"] = {"n_batches": len(batches),
                          "train_top1": round(float(M.topk_acc(pb, yb)[1]), 4),
                          "chance": round(1 / len(batches), 4)}
    print("[2] 批次探针:", OUT["batch_probe"], flush=True)
else:
    print("[1-2] 跳过第一版对照（--skip-legacy 或 bundle/ 特征缓存不存在）", flush=True)

# ---------------------------------------------------------------- 3. CNN 推理
meta = json.load(open("bundle_cnn/meta.json", encoding="utf-8"))
labels = meta["labels"]
l2i = {l: i for i, l in enumerate(labels)}

d = load_splits()
cnt = d.label.value_counts()
d = d[d.label.isin([l for l, c in cnt.items() if c >= 20])]
d = d.groupby("label", group_keys=False).head(30).reset_index(drop=True)
tr_c = d[d.split_doc == "train"].reset_index(drop=True)
te = d[(d.split_doc == "test") & d.label.isin(l2i)].reset_index(drop=True)

Xte = np.load("bundle_cnn/X_test.npy")
assert len(Xte) == len(te), (Xte.shape, len(te))
net = GlyphNet(len(labels), meta.get("ch", 32))
state = torch.load("bundle_cnn/model.pt", map_location="cpu", weights_only=True)
net.load_state_dict(state)
net.eval()


def infer(X):
    fs, ps = [], []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = torch.from_numpy(X[i:i + 256].astype("float32") / 255.0)[:, None]
            x = (x - 0.5) / 0.5
            f = net.features(x)
            fs.append(f.numpy().astype("float32"))
            ps.append(torch.softmax(net.fc(f), -1).numpy())
    return np.concatenate(fs, 0), np.concatenate(ps, 0)


fte, prob = infer(Xte)
yte = np.array([l2i[l] for l in te.label])
acc = M.topk_acc(prob, yte)
print("[3] CNN 复算 top1/top5:", {k: round(v, 4) for k, v in acc.items()}, flush=True)

thr = meta["threshold"]
conf = prob.max(1)
accepted = conf >= thr
OUT["cnn"] = {
    "n_labels": len(labels), "n_train": int(len(tr_c)), "n_test": int(len(te)),
    "test_top1": round(float(acc[1]), 4), "test_top5": round(float(acc[5]), 4),
    "threshold": round(float(thr), 4),
    "accept_rate": round(float(accepted.mean()), 4),
    "top1_on_accepted": round(float((prob[accepted].argmax(1) == yte[accepted]).mean()), 4),
    "top1_on_rejected": round(float((prob[~accepted].argmax(1) == yte[~accepted]).mean()), 4)
    if (~accepted).sum() else None,
    "n_rejected": int((~accepted).sum()),
}
print("    ", OUT["cnn"], flush=True)

# 阈值扫描：覆盖与作答准确率的折中曲线
sweep = []
correct = prob.argmax(1) == yte
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    acc_mask = conf >= t
    if acc_mask.sum() == 0:
        continue
    sweep.append({"threshold": t,
                  "coverage": round(float(acc_mask.mean()), 4),
                  "n_accepted": int(acc_mask.sum()),
                  "top1_on_accepted": round(float(correct[acc_mask].mean()), 4)})
OUT["threshold_sweep"] = sweep
print("[3b] 阈值扫描:", sweep, flush=True)

# ---------------------------------------------------------------- 4. 检索命中
index = np.load("bundle_cnn/index.npy")
ilabels = json.load(open("bundle_cnn/index_labels.json", encoding="utf-8"))
sims = M.l2norm(fte) @ index.T
# 一次排序取前 8，后续各 top-k 直接切片；argpartition 不保证第 0 位是最大值
order8 = np.argsort(-sims, axis=1)[:, :8]
hit8 = np.array([te.label.values[i] in [ilabels[j] for j in order8[i]]
                 for i in range(len(te))])
hit1 = np.array([te.label.values[i] == ilabels[order8[i][0]] for i in range(len(te))])
rng = np.random.default_rng(0)
sub = rng.choice(len(te), min(200, len(te)), replace=False)
OUT["retrieval"] = {
    "index_shape": list(index.shape), "n_query": int(len(te)),
    "top1_label_hit": round(float(hit1.mean()), 4),
    "top8_label_hit": round(float(hit8.mean()), 4),
    "top8_hit_sample200": round(float(hit8[sub].mean()), 4),
}
print("[4] 检索:", OUT["retrieval"], flush=True)

# ---------------------------------------------------------------- 5. 分批与扩展指标
by_batch = {}
for b in sorted(set(te.batch.values)):
    mask = te.batch.values == b
    if mask.sum() >= 5:
        by_batch[str(b)] = {
            "n": int(mask.sum()),
            "top1": round(float((prob[mask].argmax(1) == yte[mask]).mean()), 4),
        }
OUT["by_batch"] = by_batch

topk_hit = {}
for k in (1, 3, 5, 8):
    nn = order8[:, :k]
    hit = np.array([te.label.values[i] in [ilabels[j] for j in nn[i]]
                    for i in range(len(te))])
    topk_hit[str(k)] = round(float(hit.mean()), 4)
    if k == 8:
        OUT["retrieval_on_rejected"] = {
            "n": int((~accepted).sum()),
            "top8_hit": round(float(hit[~accepted].mean()), 4) if (~accepted).sum() else None,
        }
OUT["retrieval_topk"] = topk_hit

OUT["cnn"]["top5_on_accepted"] = round(float(
    (np.argsort(-prob[accepted], axis=1)[:, :5] == yte[accepted][:, None]).any(1).mean()), 4)
OUT["params"] = int(sum(p.numel() for p in net.parameters()))

va = d[(d.split_doc == "val") & d.label.isin(l2i)].reset_index(drop=True)
Xva = np.load("bundle_cnn/X_val.npy")
_, pva = infer(Xva)
yva = np.array([l2i[l] for l in va.label])
ava = M.topk_acc(pva, yva)
OUT["val"] = {"n": int(len(va)), "top1": round(float(ava[1]), 4),
              "top5": round(float(ava[5]), 4)}
print("[5] 分批/检索/验证:", OUT["by_batch"], OUT["retrieval_topk"], OUT["val"], flush=True)

# ---------------------------------------------------------------- 6. 误判案例
order = np.argsort(-prob, axis=1)
wrong = np.where(order[:, 0] != yte)[0]
cases = []
for i in wrong[:8]:
    cases.append({
        "true": str(te.label.values[i]),
        "batch": str(te.batch.values[i]),
        "doc": str(te.doc.values[i]),
        "slip": str(te.slip.values[i]),
        "path": str(te.path.values[i]),
        "conf": round(float(prob[i].max()), 4),
        "top5": [{"label": labels[j], "p": round(float(prob[i, j]), 4)}
                 for j in order[i, :5]],
        "nn_labels": [ilabels[j] for j in np.argsort(-sims[i])[:5]],
    })
OUT["failure_cases"] = cases
print("[5] 误判案例 %d 例，首例：" % len(cases), flush=True)
if cases:
    print("   ", cases[0]["true"], "->", [x["label"] for x in cases[0]["top5"]], flush=True)

json.dump(OUT, open(args.output, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"写出 {args.output}", flush=True)
