# -*- coding: utf-8 -*-
"""楚简相与析 —— 楚简字形识别与形近字检索

启动:
    python app.py --bundle bundle_cnn
"""
import argparse, json, os

# 本机若开着系统代理（如 127.0.0.1:7892），gradio 启动时的自检请求会走代理，
# 对 localhost 返回 502，启动因此失败。必须在 import gradio 之前把本地地址排除。
for _key in ("NO_PROXY", "no_proxy"):
    _parts = [p.strip() for p in os.environ.get(_key, "").split(",") if p.strip()]
    for _host in ("127.0.0.1", "localhost", "::1"):
        if _host not in _parts:
            _parts.append(_host)
    os.environ[_key] = ",".join(_parts)

import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageOps

import model as M

STATE = {}


def _load_cnn(bundle, meta):
    from cnn import GlyphNet, SIZE
    net = GlyphNet(len(meta["labels"]), meta.get("ch", 32))
    state = torch.load(os.path.join(bundle, "model.pt"), map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()

    def prep(path, size=meta.get("size", SIZE)):
        im = Image.open(path).convert("L")
        im = ImageOps.autocontrast(im).resize((size, size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype="float32") / 255.0)[None, None]
        return (x - 0.5) / 0.5

    def embed(path):
        with torch.no_grad():
            f = net.features(prep(path))
            logit = net.fc(f)
        return f.numpy(), torch.softmax(logit, -1).numpy()[0]

    return embed


def _load_resnet(bundle, meta):
    net, tf, _ = M.build(meta["backbone"])
    head = M.LinearHead(meta["dim"], len(meta["labels"]))
    state = torch.load(os.path.join(bundle, "head.pt"), map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    head.eval()

    def embed(path):
        f = M.embed_paths(net, tf, [path])
        with torch.no_grad():
            logit = head(torch.from_numpy(f))
        return f, torch.softmax(logit, -1).numpy()[0]

    return embed


def load(bundle):
    with open(os.path.join(bundle, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    def j(n):
        with open(os.path.join(bundle, f"index_{n}.json"), encoding="utf-8") as f:
            return json.load(f)

    ilabels = j("labels")
    kind = meta.get("kind", "resnet")

    # 交付版形近字直接取训练时的 64x64 缓存，与检索索引同序。
    # index_paths.json 仅用于开发期溯源，运行与交付均不需要本机 CHUBS 路径。
    thumb_path = os.path.join(bundle, "X_train.npy")
    thumbs = np.load(thumb_path, mmap_mode="r") if os.path.exists(thumb_path) else None
    if thumbs is not None and len(thumbs) != len(ilabels):
        raise ValueError(f"{thumb_path} 与检索索引长度不符，请重新导出模型包")
    if kind == "cnn" and thumbs is None:
        raise FileNotFoundError(f"缺少运行必需文件：{thumb_path}")

    path_file = os.path.join(bundle, "index_paths.json")
    paths = j("paths") if os.path.exists(path_file) else None
    if thumbs is None and paths is None:
        raise FileNotFoundError("X_train.npy 与 index_paths.json 均缺失，无法显示形近字")

    index = np.load(os.path.join(bundle, "index.npy"))
    if index.ndim != 2 or len(index) != len(ilabels):
        raise ValueError("index.npy 与 index_labels.json 长度不符，请重新导出模型包")

    STATE.clear()
    STATE.update({
        "meta": meta,
        "embed": _load_cnn(bundle, meta) if kind == "cnn" else _load_resnet(bundle, meta),
        "index": index,
        "thumbs": thumbs,
        "paths": paths, "ilabels": ilabels, "ibatch": j("batch"),
        "idoc": j("doc"), "islip": j("slip"),
    })


def recognise(img_path):
    if not img_path:
        return {}, [], "请先上传一张楚简字形图片，再点「识读」。"
    meta = STATE["meta"]
    try:
        f, prob = STATE["embed"](img_path)
    except (OSError, ValueError):
        return {}, [], "图片无法读取。请换用 PNG / JPG 格式的单字图后重试。"
    order = np.argsort(-prob)[:5]
    top = {meta["labels"][i]: float(prob[i]) for i in order}
    best = float(prob[order[0]])

    q = M.l2norm(f)[0]
    sims = STATE["index"] @ q
    nn = np.argsort(-sims)[:8]
    thumbs = STATE.get("thumbs")
    gallery = []
    for j in nn:
        cap = (f"{STATE['ilabels'][j]}｜{STATE['ibatch'][j]}·{STATE['idoc'][j]}·简{STATE['islip'][j]}"
               f"（{sims[j]:.2f}）")
        if thumbs is not None and j < len(thumbs):
            media = Image.fromarray(np.asarray(thumbs[j]))
        else:
            media = STATE["paths"][j] if STATE["paths"] is not None else None
        if media is not None:
            gallery.append((media, cap))

    if best < meta["threshold"]:
        verdict = (f"**判不准。** 最高置信度 {best:.2f}，低于阈值 {meta['threshold']:.2f}。\n\n"
                   "置信度偏低，可能是本字库未收、写法特殊或图像质量影响。"
                   "请比对下方形近字，或查《楚文字编》《楚系简帛文字编》。")
    else:
        verdict = (f"**最可能：{meta['labels'][order[0]]}**（置信度 {best:.2f}）\n\n"
                   "下面是字形最接近的 8 个已知字例，附出处，可逐一比对笔势。")
    return top, gallery, verdict


def build_ui():
    meta = STATE["meta"]
    with gr.Blocks(title="楚简相与析") as demo:
        gr.Markdown(
            "# 楚简相与析\n"
            "奇文共欣赏，疑义相与析　—— 陶渊明《移居》\n\n"
            "楚简字形识别与形近字检索助手\n\n"
            f"字库：{len(meta['labels'])} 字｜材料：清华简、上博简、包山简、郭店简等 15 批\n\n"
            "**用法**：上传一张裁好的楚简单字图（整简照片请先自行裁出单字），再点「识读」。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="字形图片")
                btn = gr.Button("识读", variant="primary")
                example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
                example_files = [os.path.join(example_dir, f"example_{i:02d}.png") for i in (1, 2, 3)]
                example_files = [p for p in example_files if os.path.exists(p)]
                if example_files:
                    gr.Examples(examples=example_files, inputs=img,
                                label="示例字图（点击载入，再点「识读」）")
            with gr.Column(scale=2):
                out_md = gr.Markdown()
                out_lab = gr.Label(num_top_classes=5, label="候选（置信度）")
        gr.Markdown("### 形近字（按字形相似度排序，附出处）")
        gal = gr.Gallery(label="形近字", columns=8, height=220)
        btn.click(recognise, inputs=img, outputs=[out_lab, gal, out_md])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="bundle_cnn")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = ap.parse_args()
    load(args.bundle)
    build_ui().launch(server_name="127.0.0.1", server_port=args.port,
                      inbrowser=not args.no_browser)
