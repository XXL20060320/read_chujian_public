# -*- coding: utf-8 -*-
"""CHUBS 数据准备：建索引 -> 过滤伪标签 -> 按篇分组切分 -> 输出 splits.csv

用法:
    python prep.py --root C:\\Users\\natou\\CHUBS\\glyphs --out data
"""
import argparse, csv, hashlib, os, random, sys
from collections import Counter, defaultdict

PSEUDO = {"None", "△", "○"}          # 不是字，不能做类别
MIN_PER_CLASS = 5                     # 每类至少多少张才进训练
MAX_PER_CLASS = 40                    # 每类最多取多少张（控制 CPU 成本）


def parse_name(fn):
    stem = fn[:-4]
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    batch, doc = parts[0], parts[1]
    docid, _, docname = doc.partition("-")
    return batch, docid, docname, parts[2]


def stable_bucket(key, buckets=10):
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for label in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, label)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(".png"):
                continue
            p = parse_name(fn)
            if p is None:
                continue
            batch, docid, docname, slip = p
            rows.append({
                "path": os.path.join(d, fn), "label": label,
                "batch": batch, "doc": f"{docid}-{docname}", "slip": slip,
                "doc_key": f"{batch}|{docid}-{docname}",
            })

    print(f"[1] 扫描到 {len(rows)} 张图，{len(set(r['label'] for r in rows))} 个标签")

    # 过滤伪标签
    rows = [r for r in rows if r["label"] not in PSEUDO]
    print(f"[2] 去掉 None/△/○ 后 {len(rows)} 张")

    cnt = Counter(r["label"] for r in rows)
    keep = {l for l, c in cnt.items() if c >= MIN_PER_CLASS}
    rows = [r for r in rows if r["label"] in keep]
    print(f"[3] 保留 ≥{MIN_PER_CLASS} 张的类别：{len(keep)} 类，{len(rows)} 张")

    # 每类上限，控制训练量
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    rng = random.Random(42)
    capped = []
    for l, items in by.items():
        rng.shuffle(items)
        capped.extend(items[:MAX_PER_CLASS])
    rows = capped
    print(f"[4] 每类最多 {MAX_PER_CLASS} 张 -> {len(rows)} 张")

    # 两种切分：随机（乐观） / 按篇分组（真实）
    docs = sorted({r["doc_key"] for r in rows})
    doc_split = {}
    for d in docs:
        b = stable_bucket(d, 10)
        doc_split[d] = "train" if b < 8 else ("val" if b == 8 else "test")
    for r in rows:
        r["split_doc"] = doc_split[r["doc_key"]]

    # 随机切分：按类分层
    rng2 = random.Random(7)
    groups = defaultdict(list)
    for r in rows:
        groups[r["label"]].append(r)
    for l, items in groups.items():
        rng2.shuffle(items)
        n = len(items)
        n_val = max(1, n // 10)
        n_test = max(1, n // 10)
        for i, it in enumerate(items):
            it["split_rand"] = "val" if i < n_val else ("test" if i < n_val + n_test else "train")

    # 统计：按篇分组后，测试集里有多少类没在训练集出现过
    train_cls = {r["label"] for r in rows if r["split_doc"] == "train"}
    test_cls = {r["label"] for r in rows if r["split_doc"] == "test"}
    unseen = test_cls - train_cls
    print(f"[5] 按篇切分：train {sum(1 for r in rows if r['split_doc']=='train')} / "
          f"val {sum(1 for r in rows if r['split_doc']=='val')} / "
          f"test {sum(1 for r in rows if r['split_doc']=='test')}")
    print(f"    测试集出现但训练集没见过的类别：{len(unseen)} / {len(test_cls)}（这些只能靠检索兜底）")

    out = os.path.join(args.out, "splits.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "batch", "doc", "slip", "doc_key",
                                           "split_doc", "split_rand"])
        w.writeheader()
        w.writerows(rows)
    print(f"[6] 写出 {out}")


if __name__ == "__main__":
    main()
