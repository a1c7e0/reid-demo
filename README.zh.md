# reid-demo（中文说明）

自托管相册，按**人**分组——先用脸，再用**人体 Re-ID**。人体这一步正是**背影识别**的关键：一个人背对镜头、或脸被挡住，仍然能挂到正确的人身上。

这个仓库是该能力的**演示**：基础相册、人物页、人物管理（review）、照片元数据，外加查看模型工作的辅助功能——照片上的**人物框**、**绑定修改**、**相似度展示**。

英文版见 [README.md](README.md)（内容一致，英文版是主文档）。

---

## 快速开始

```bash
docker build -t reid-demo .
docker run -d --name reid-demo \
  -p 8000:8000 \
  -v "$PWD/data:/data" \
  reid-demo
```

打开 <http://localhost:8000>。

首次构建会下载 ONNX 模型并导出其中两个（`scripts/download_models.sh`、`scripts/export_reid_zoo.py`），**校验哈希不匹配会直接构建失败**。留几分钟和约 1 GB 构建缓存。

也提供了 `docker-compose.yml`。

---

## 怎么用

### 1. 导入照片

- **Settings → Upload photos**：浏览器选文件上传
- **Settings → Scan a folder**：导入服务器上已有的目录

```bash
docker run --rm -v "$PWD/data:/data" -v /path/to/photos:/import:ro \
  reid-demo python -c "from app.ingest import ingest_dir; ingest_dir('/import')"
```

每张照片会做人脸/人体检测、嵌入、排队处理。进度看 **Settings → Status**。

### 2. 照片页

按拍摄日期分组，点开进查看器。

查看器上**画出模型给出的所有框**：绿色 = 脸，蓝色虚线 = 身体。每个框都标了归属的人（没归属就写 `face` / `body`），身体框还会写**挂靠原因**（`body+time`、`body+gap` 等）。右侧 **Info** 面板显示照片元数据——拍摄时间、像素尺寸、文件大小、相对路径、sha1——以及每个检测单位一行，带**人物选择器**可以改绑。

**相似度**在人物选择器里展示，**按相似度降序**，最可能的人排最上。

### 3. 人物页

方形磁贴：头像、名字（或 `Person N`）、置顶标记。

- **点**磁贴进入该人物
- 标题栏 **Select** 按钮进入多选，**Done** 退出。选中的磁贴打勾，底部出现动作条：**Edit**（改名）、**Review**、**Pin / Unpin**、**Reset**，选两个及以上还有 **Merge**——合并面板会**显式标出方向**（"Everyone else joins person #12"），因为方向搞反在界面上没法补救
- **Sort**：按照片数多/少、名字正/倒序

### 4. 人物详情

这个人的照片、他聚出来的脸、以及身体挂靠的原因标签。操作：**Rename**、**Pin / Unpin**、**Review**，以及 **Mixed grouping?**——一键拆散：凡是两组脸互相不够像的地方就拆开（complete-link），也就是人会说"这俩不是一个人"的地方。

### 5. Review（审核）

修聚类错误的地方。一次一条：左边目标人物，右边候选簇，带脸/人体相似度数字、照片间隔天数、**明确标出方向**。**Merge into this person** / **Not this person** / **Skip**，另有 **Decided history** 可以 **Reopen**。

队列有两处设计：

- **按 MMR 排序**（Carbonell & Goldstein, SIGIR 1998）+ 期望错误削减加权，下一条是"可能有用"和"不是你刚答过的那道题"的最优折中。同一个人被拆成几个簇时，不会连着把同一个问题问三遍
- **拒绝会记住**：点了"Not this person"就记下被拒绝的样子，后面读起来是同一人的候选会直接跳过

可调参数：`REVIEW_MMR_LAMBDA`（默认 0.3）、`REVIEW_MMR_IMPACT`（默认开）、`REVIEW_DUP_T`（默认 0.72）。

---

## 模型在做什么

完整流程：**导入 → 人脸检测 → 人脸嵌入 → 人体检测 → 人体嵌入 → 聚类 → 身体挂靠**。

1. **人脸**：YuNet 检测，`align5` 做相似变换对齐到 ArcFace 模板，**LVFace-T** 出嵌入。质量门（模糊度 / 检测分 / 侧脸）决定哪些脸才参与聚类
2. **聚类**：Ente 的贪心单链接算法（移植在 `app/ente_core.py`）——新照片优先、只向前找最近邻、按质量自适应阈值（好邻居 `ENTE_T_GOOD=0.64` / 差邻居 `ENTE_T_BAD=0.72`）、一照一人的约束、**共现不变量**（同一张照片里出现的两个人永远不进同一个簇）
3. **碎块找回**：complete-link 合并，**只有所有交叉对都 ≥ `PERSON_SPLIT_SIM`** 才把两个簇折到一起——这个档位上数学上不可能混人。它正好是 **Mixed grouping?** 拆分按钮的逆运算
4. **身体挂靠**：OSNet-AIN x1.0 出嵌入；背影只有在**超过挂靠阈值**、且**比第二名高 `BODY_MARGIN`**、且**落在拍摄时间窗内**（`BODY_MAX_GAP_S`）才会挂到人身上。身体**永远不会自己立一个人**

完整设计见 [`docs/DESIGN.md`](docs/DESIGN.md)，Ente 移植记录见 [`docs/ENTE-PORT-SPEC.md`](docs/ENTE-PORT-SPEC.md)，每个阈值及其依据见 [`docs/ALGO-TUNING.md`](docs/ALGO-TUNING.md)。

**当前加载的模型、大小、所有生效阈值都能在 Settings → Model & configuration 里看到**，不用猜这个构建在干什么。

---

## 配置

环境变量（均可选，默认值见 `app/config.py`）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `DATA_DIR` | `/data` | 照片存储、缩略图、数据库 |
| `MODEL_DIR` | `/models` | ONNX 模型 |
| `ENTE_T_GOOD` | `0.64` | 好质量邻居的合并阈值 |
| `ENTE_T_BAD` | `0.72` | 差质量邻居的合并阈值 |
| `PERSON_SPLIT_SIM` | `0.53` | 拆分下限，同时也是碎块合并阈值 |
| `PERSON_EJECT_BAR` | `0` | 自动踢出阈值（默认关） |
| `BODY_JOIN_T` | `0.63` | 身体挂靠阈值 |
| `BODY_MARGIN` | `0.06` | 身体挂靠歧义余量 |
| `BODY_MAX_GAP_S` | `10800` | 身体挂靠的拍摄时间窗（3 小时） |
| `REVIEW_SIM_MIN` | `0.15` | 审核灰带下界 |
| `REVIEW_MMR_LAMBDA` | `0.3` | MMR 相关性/多样性折中 |
| `REVIEW_MMR_IMPACT` | `1` | 审核相关性按合并影响加权 |
| `REVIEW_MMR_TOP` | `200` | MMR 重排的候选条数 |
| `REVIEW_DUP_T` | `0.72` | "读作同一人"的抑制阈值 |

---

## 前端

Vue 3 + Vue Router，配合 **Tom Select**（人物选择器）、**PhotoSwipe 5**（查看器）、**Toastify**（提示），**全部本地化，不用 CDN**。交互沿用 Ente 移动端的模式（方形磁贴、底部动作条、确认弹层），但控件是基于这些组件写的，不是照搬标记。

宽度 ≥1000px 时切到桌面布局：内容栏更宽、缩略图更大、导航吸到顶部而不是浮在照片上。网页文字全部英文。

构建时会跑 **JS 语法检查**（对 `app/static/js/` 逐个 `node --check`），ES 模块坏了会直接构建失败，不会上线出白屏。

---

## 开发

```bash
docker build -t reid-demo .          # 含 JS 语法门
docker run --rm -p 8000:8000 -v "$PWD/data:/data" reid-demo
```

评测脚本在 `scripts/eval/`，**只用公开数据**：

```bash
# LFW 上测碎裂 / 碎块找回
docker run --rm --cpus=1 --memory=1200m \
  -v "$PWD:/work" -v "$PWD/testsets/lfw_flat:/lfw:ro" \
  reid-demo python /work/scripts/eval/fragment_regen.py

# review 排序对比（MMR vs 纯相关性 vs 纯多样性）
python3 scripts/eval/review_order_bench.py
```

数据集及其下载链接见 [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md)。**任何数据集和权重都不入库**，见 `.gitignore`。

---

## 研究记录

这个构建里的每个决定都是**测出来的**，不是拍脑袋。记录都在 `docs/`：

| 文件 | 内容 |
|---|---|
| [`DESIGN.md`](docs/DESIGN.md) | 端到端流水线 |
| [`ALGO-TUNING.md`](docs/ALGO-TUNING.md) | 每个阈值 + 背后的 A/B |
| [`MODEL-ZOO.md`](docs/MODEL-ZOO.md) | 人脸模型对比（6 个） |
| [`REID-MODEL-ZOO.md`](docs/REID-MODEL-ZOO.md) | 人体模型对比 |
| [`ENTE-PORT-SPEC.md`](docs/ENTE-PORT-SPEC.md) | 哪些照搬 Ente、哪些重新标定 |
| [`HARD-SETS.md`](docs/HARD-SETS.md) | 评测集及各自用途 |
| [`REID-DATA.md`](docs/REID-DATA.md) | Re-ID 调参集 |
| [`DATA-SOURCES.md`](docs/DATA-SOURCES.md) | 全部下载链接 |
| [`RESEARCH-LOG.md`](docs/RESEARCH-LOG.md) | 实验与结论的流水记录 |
| [`prototypes-survey.md`](docs/prototypes-survey.md) | UI 调研笔记 |

两条贯穿始终的规则：**一切只在公开数据上评测**；聚类目标**刻意不对称**——两个人混进一个簇重罚，一个人拆成多个簇无所谓。在相册里，混了是**对"这是谁"撒谎**，拆了只是**一键能修的乱**。
