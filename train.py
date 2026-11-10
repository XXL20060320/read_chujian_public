# -*- coding: utf-8 -*-
"""第一版（对照用）：提取特征 -> 训练线性分类头 -> 调开放集阈值 -> 保存模型包

用法:
    python train.py --splits data/splits.csv --out bundle --backbone resnet18
    python train.py --limit 2000            # 冒烟测试

需 torchvision 与 ResNet18 预训练权重。交付版为 cnn.py / bundle_cnn。
"""
import argparse, hashlib, json, os, time
import numpy as np
import pandas as pd
import torch

import model as M


def load_splits(path):
    df = pd.read_csv(path)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    return df


def file_sig(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_features(net, tf, paths, cache, tag, sig):
    mf = cache + ".meta.json"
    if os.path.exists(cache):
        try:
            if os.path.exists(mf) and json.load(open(mf, encoding="utf-8")) == sig:
                print(f"  [cache] {tag} <- {cache}", flush=True)
                return np.load(cache)
            print(f"  [缓存失效] {tag}: 骨干或数据已变，重新提取", flush=True)
        except (OSError, ValueError):
            print(f"  [缓存元信息损坏] {tag}: 重新提取", flush=True)
    t0 = time.time()
    print(f"  [抽取] {tag}: {len(paths)} 张", flush=True)
    f = M.embed_paths(net, tf, list(paths))
    np.save(cache, f)
    json.dump(sig, open(mf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  [完成] {tag}: {f.shape} 用时 {time.time()-t0:.0f}s", flush=True)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--out", default="bundle")
    ap.add_argument("--backbone", default="resnet18", choices=list(M.BACKBONES))
    ap.add_argument("--limit", type=int, default=0, help="冒烟测试：每类只取前 N 张")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    os.makedirs(args.out, exist_ok=True)

    df = load_splits(args.splits)
    if args.limit:
        df = df.groupby("label", group_keys=False).head(args.limit).reset_index(drop=True)
    print(f"样本 {len(df)} 张，类别 {df['label'].nunique()} 个")

    # 类别表（只用训练集出现过的类）
    tr = df[df["split_doc"] == "train"]
    va = df[df["split_doc"] == "val"]
    te = df[df["split_doc"] == "test"]
    labels = sorted(tr["label"].unique())
    lab2id = {l: i for i, l in enumerate(labels)}
    print(f"训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}；类别 {len(labels)}")

    net, tf, dim = M.build(args.backbone)
    print(f"backbone={args.backbone} dim={dim}")

    def sig_for(paths):
        return {"backbone": args.backbone, "splits": file_sig(args.splits),
                "n": len(paths), "first": str(paths.iloc[0]) if len(paths) else "",
                "last": str(paths.iloc[-1]) if len(paths) else ""}

    ftr = cache_features(net, tf, tr["path"], os.path.join(args.out, "feat_train.npy"),
                         "train", sig_for(tr["path"]))
    fva = cache_features(net, tf, va["path"], os.path.join(args.out, "feat_val.npy"),
                         "val", sig_for(va["path"]))
    fte = cache_features(net, tf, te["path"], os.path.join(args.out, "feat_test.npy"),
                         "test", sig_for(te["path"]))

    # 验证/测试里可能出现训练集没有的类 -> 先剔掉，单独统计"开集"部分
    ytr = np.array([lab2id[l] for l in tr["label"]])
    mva = va["label"].isin(lab2id).values
    mte = te["label"].isin(lab2id).values
    yva = np.array([lab2id[l] for l in va["label"][mva]])
    yte = np.array([lab2id[l] for l in te["label"][mte]])
    print(f"闭集评估：验证 {mva.sum()}/{len(va)}，测试 {mte.sum()}/{len(te)}"
          f"（其余 {int((~mte).sum())} 张属于训练集没见过的字）")

    head = M.train_head(ftr, ytr, len(labels), epochs=args.epochs)
    if mva.sum() == 0 or mte.sum() == 0:
        print("警告：验证/测试集里没有训练集见过的类，跳过闭集评估")
        pva = M.predict(head, fva[:0])
        pte = M.predict(head, fte[:0])
        yva = np.zeros(0, "int64")
        yte = np.zeros(0, "int64")
    else:
        pva = M.predict(head, fva[mva])
        pte = M.predict(head, fte[mte])
        print("验证 top1/top5:", {k: round(v, 4) for k, v in M.topk_acc(pva, yva).items()})
        print("测试 top1/top5:", {k: round(v, 4) for k, v in M.topk_acc(pte, yte).items()})

    # 开放集阈值：让验证集上 90% 的样本被接受
    thr = float(np.quantile(pva.max(1), 0.10)) if len(pva) else 0.5
    cover = float((pte.max(1) >= thr).mean()) if len(pte) else 0.0
    print(f"开放集阈值={thr:.4f}（验证集保留90%） 测试集接受率={cover:.3f}")

    # 检索库：用训练集特征（已归一化）
    idx = M.l2norm(ftr).astype("float32")
    np.save(os.path.join(args.out, "index.npy"), idx)
    json.dump(list(tr["path"]), open(os.path.join(args.out, "index_paths.json"), "w",
                                     encoding="utf-8"), ensure_ascii=False)
    json.dump(list(tr["label"]), open(os.path.join(args.out, "index_labels.json"), "w",
                                      encoding="utf-8"), ensure_ascii=False)
    json.dump(list(tr["batch"]), open(os.path.join(args.out, "index_batch.json"), "w",
                                      encoding="utf-8"), ensure_ascii=False)
    json.dump(list(tr["doc"]), open(os.path.join(args.out, "index_doc.json"), "w",
                                    encoding="utf-8"), ensure_ascii=False)
    json.dump(list(tr["slip"].astype(str)), open(os.path.join(args.out, "index_slip.json"), "w",
                                                 encoding="utf-8"), ensure_ascii=False)

    torch.save(head.state_dict(), os.path.join(args.out, "head.pt"))
    meta = {
        "backbone": args.backbone, "dim": dim, "labels": labels,
        "threshold": thr, "test_top1": float(M.topk_acc(pte, yte)[1]),
        "test_top5": float(M.topk_acc(pte, yte)[5]),
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("已保存 ->", args.out)


if __name__ == "__main__":
    main()
