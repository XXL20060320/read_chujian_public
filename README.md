# 简牍识字 · 楚简字形识别与形近字检索

上传一张裁好的楚简单字图，返回 top-5 候选标签与置信度、8 个带出处的形近字例，以及一条按阈值给出的判语。竞赛方案见 `作品设计方案.pdf`（31 页，目录页码已核对），可编辑源稿见 `作品设计方案.docx` 与 `作品设计方案.md`；数据与素材来源见 `NOTICE.md`。

## 直接运行（Windows）

双击 start.bat 即可：脚本会自动寻找已装好依赖的 Python、检查模型包是否齐全，然后启动并打开浏览器。

以下为手动方式。

交付模型包 `bundle_cnn/` 已随项目提供，不需要重新训练，也不需要 CHUBS 原始数据集。

```powershell
# 本机完整验证使用的解释器
$py = "C:\Users\natou\AppData\Local\Python\pythoncore-3.11-64\python.exe"
& $py app.py --bundle bundle_cnn
```

浏览器打开 `http://127.0.0.1:7860`，上传字图后点「识读」；也可直接点击界面中的三张随包示例。形近字取自 `bundle_cnn/X_train.npy` 中与检索索引同序的 64×64 灰度图；应用不读取 `index_paths.json`，也不依赖任何 CHUBS 本机路径。

在其他电脑上，安装依赖后可将上述 `$py` 换成自己的 Python 3.11 解释器。不要根据 PATH 中裸 `python` 的名字推断环境，先用 `python --version` 和 `python -m pip show torch gradio` 确认。

## 环境与依赖

交付链路的实测环境：Windows / Python 3.11.9 / torch 2.14.1+cpu / numpy 2.4.6 / pandas 3.1.0 / Pillow 12.3.0 / gradio 6.30.0，全程 CPU。第一版对照另用 torchvision 0.29.2+cpu。

```powershell
& $py -m pip install -r requirements.txt
```

`requirements.txt` 只列交付版依赖。重跑第一版时才需要：

```powershell
& $py -m pip install -r requirements-legacy.txt
```

如默认源未装到 CPU 版 PyTorch，使用官方源：

```powershell
& $py -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# 仅第一版需要同时安装匹配版本：
& $py -m pip install torch==2.14.1 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
```

本机有多个 Python 解释器。上述 `pythoncore-3.11-64` 已验证可导入交付版全部依赖及 torchvision；`Programs\Python\Python311` 虽同为 3.11.9，但未安装 torchvision，不用它跑第一版。ResNet18 权重在上一级目录 `..\.torch\hub\checkpoints\resnet18-f37072fd.pth`，重跑第一版时设置 `$env:TORCH_HOME = (Resolve-Path "..\.torch").Path`，否则 torchvision 会尝试重新下载。

Windows 中文输出乱码时先设 `$env:PYTHONUTF8=1`。

## 数据

CHUBS（清华大学 NLP 组，COLING 2025），102,722 条索引、8,349 个标签、15 批材料。下载：https://huggingface.co/datasets/chen-yingfa/CHUBS ，按其发布许可使用。

```text
glyphs/<标签>/<批次>_<篇号-篇名>_<简号>_<批次号>-<简号>-<字序号>.png
```

`prep.py` 按文件名解析出处、剔除伪标签、按「批次|篇号-篇名」哈希分组切分，生成 `data/splits.csv`。项目有两套类别门槛，数字不能混用：`prep.py` 每类 ≥5 且上限 40（35,197 张 / 1,922 类，第一版对照用）；`cnn.py` 在其上二次筛选为每类 ≥20 且上限 30（682 类，交付版用）。

随包的 `splits.csv` 保留开发期原始路径，供出处与标签表使用。交付版应用和 CNN 复算均读取随包缓存，不要求这些路径存在；重新训练或运行像素基线则需重新下载 CHUBS 并执行 `prep.py` 生成当前机器的路径。

## 复现与核验

以下命令均在项目根目录执行：

```powershell
$py = "C:\Users\natou\AppData\Local\Python\pythoncore-3.11-64\python.exe"
$env:PYTHONUTF8 = "1"

# 1. 数据索引与切分（仅重新训练需要 CHUBS 原始字图）
& $py prep.py --root C:\Users\natou\CHUBS\glyphs --out data

# 2. 重新训练交付模型（CPU 20 轮约 80 分钟）
& $py cnn.py --splits data/splits.csv --out bundle_cnn --seed 42 --threshold 0.5

# 3. 复算交付版指标；不读取 CHUBS 原始图片
& $py eval_extras.py --skip-legacy --output eval_cnn_recheck.json

# 4. 启动界面
& $py app.py --bundle bundle_cnn
```

完整开发目录还保留 `bundle/`，因此不加 `--skip-legacy` 运行 `eval_extras.py` 会同时复算第一版的切分对照与批次探针。精简提交包不含第一版产物，应使用上面的 `--skip-legacy`；随包 `eval_extras.json` 保存完整开发目录的原始评测结果，复算结果写入另一个文件，避免覆盖。

`--threshold` 显式给 0.5 是为了重训后保持相同界面操作点；不指定时脚本采用验证集置信度 10 分位数。交付权重训练完成后，阈值参考测试集分区曲线人工设为 0.5，并非在独立验证集上选定，方案第 4.4、5.3 节已如实说明。`--seed` 固定初始化、增强与 DataLoader 洗牌；`X_*.npy` 缓存带数据指纹与筛选参数校验。

交付包 `model.pt` 是加入 `--seed` 选项之前训练的权重，第五章数字均来自该权重。用新脚本重训会得到略有差异的指标；逐位核对现有数字时直接使用交付权重运行复算命令。

第一版对照实验（需 torchvision、ResNet18 权重；`diag2.py` 另需 CHUBS 原图）：

```powershell
$env:TORCH_HOME = (Resolve-Path "..\.torch").Path
& $py train.py --splits data/splits.csv --out bundle --backbone resnet18
& $py diag.py
& $py diag2.py
& $py pixel_baseline.py
```

## 目录

```text
prep.py                     解析文件名、清洗标签、生成两种切分
cnn.py                      交付版 GlyphNet 训练、索引导出
app.py                      Gradio 界面
eval_extras.py              补充评测，支持只复算交付版
model.py / train.py         第一版骨干、特征抽取、线性头
diag.py / diag2.py          第一版近邻、原型、批次探针、二值化对照
pixel_baseline.py           像素模板 kNN 基线（需 CHUBS 原图）
data/splits.csv             35,197 条索引与切分
examples/                   三张可直接在界面载入的示例字图与出处
bundle_cnn/                 交付模型包
bundle/                     第一版特征与模型（仅完整开发目录）
bundle_smoke/               冒烟测试产物（不进入提交包）
requirements.txt            交付版依赖
requirements-legacy.txt     第一版附加依赖
eval_extras.json            完整实测结果
eval_baselines.log          像素基线日志
eval_diag2.log              二值化对照日志
作品设计方案.md / .docx    方案源稿与可导出 PDF 的提交稿
```

`bundle_cnn/` 中 `model.pt`、`index.npy`、`index_labels.json`、`index_batch.json`、`index_doc.json`、`index_slip.json`、`meta.json`、`X_train.npy` 为运行必需；`index_paths.json` 只是开发期溯源信息，不进入精简提交包。`X_val.npy` / `X_test.npy` 只在复算交付版指标时需要。

## 方法要点

输入为 64×64 灰度图，先 `autocontrast` 再归一化；四个卷积块后接全局平均池化得 256 维特征，分类头 `Linear(256, 682)`，共 1,347,914 个参数。分类与检索共用这套特征：训练集特征 L2 归一化存为 `index.npy`，查询取余弦相似度 top-8。最高置信度低于 `meta.json` 的 `threshold` 时显示「判不准」。

第一版曾用冻结 ResNet18 特征加线性头，按篇切分测试集闭集部分 top-1 为 0.1437；批次探针（15 类，0.9506）显示特征含很强的批次与扫描风格信号，故改用从零训练的小网络。两套切分协议、拒识与检索的实测数字及当前局限见设计方案第五章和第八章。

