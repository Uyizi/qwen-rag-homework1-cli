# 本地 Qwen 个人知识库

软件前沿技术作业1的独立 CLI 代码仓库

默认使用本地 Qwen 连续聊天，支持流式回答、Markdown 和多轮上下文
切换到文档模式后，支持 PDF、Markdown、TXT 以及 BM25 和 E5 向量检索
文档模式依据证据回答并显示出处，资料不足时拒答

## 目录

```text
cli.py                     命令行交互
terminal_ui.py             流式显示 Markdown 与多行输入
chat_session.py            聊天上下文 文档切换 评测隔离
rag.py                     导入 分块 检索 生成与日志
pdf_structure.py           PDF 表格恢复与数值计算
evidence_checks.py         证据与条件校验
model_worker.py            WSL 常驻模型进程
bootstrap.py               路径配置与启动检查
config.example.json        配置模板
requirements.txt           Windows 依赖
requirements-backend.txt   WSL 后端依赖
corpus/                    论文及三种格式资料
artifacts/chunks.json      标准来源编号对应的片段
eval/questions.csv         20题 标准答案 题型与出处
eval/questions.json        同一题集的 JSON 版本
eval/results/              两种方法各20条 CSV 和 JSON
eval/run-20261008-102913/   完整原始记录 评分 环境与指标
docs/                      评分依据与代码错误记录
vendor/wheels/             终端界面依赖的离线安装包及许可证
```

## 运行前提

Windows 10 或 11、WSL2 和支持 CUDA 与 BF16 的 NVIDIA GPU
Windows 和 WSL 中均需有 Python 环境

需要提前准备本地 Qwen 和 multilingual-e5-small 模型文件
原评测使用个人导出的 Qwen3.5-4B-Dense-style-v1
权重体积较大，不包含在代码仓库中
使用其他兼容权重可以运行，但不会保证复现本仓库的评分

一键启动指依赖和本地权重准备完成后双击启动
首次启动填写三个路径，之后自动复用本地配置
程序不自动下载模型、依赖或调用云端 API

## 实测环境

| 项目 | 版本 |
| --- | --- |
| Windows Python | 3.13.7 |
| NumPy | 2.1.2 |
| PyMuPDF | 1.28.0 |
| Rich | 15.0.0 |
| prompt-toolkit | 3.0.53 |
| WSL Python | 3.13.11 |
| PyTorch | 2.14.0+cu130 |
| Transformers | 5.17.0 |
| Accelerate | 1.10.1 |
| Safetensors | 0.8.0 |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| 生成配置 | BF16 贪心解码 关闭 thinking |
| 向量模型 | multilingual-e5-small 384维 |

原始环境快照位于 [eval/run-20261008-102913/environment.json](eval/run-20261008-102913/environment.json)

## 安装

克隆代码

```powershell
git clone https://github.com/Uyizi/qwen-rag-homework1-cli.git
cd qwen-rag-homework1-cli
```

创建 Windows 虚拟环境并从已经准备好的 wheel 目录安装依赖
将 D:\wheels 替换为本机实际目录

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links D:\wheels -r requirements.txt
```

WSL 使用已有兼容环境时可以跳过后端安装
新建环境时先在 WSL 中切换到克隆目录，再从本地 wheel 安装匹配的 CUDA 依赖
requirements-backend.txt 记录原评测使用的后端版本
安装前需准备对应 Python 与 CUDA 版本的全部依赖 wheel

```bash
python3 -m venv ~/venvs/qwen-rag
~/venvs/qwen-rag/bin/python -m pip install --no-index --find-links /path/to/wheels -r requirements-backend.txt
```

若需要联网取得 wheel 或模型文件，请先在不经过 VPN 的网络中准备文件
本仓库中的安装命令只读取本地文件

## 一键启动

双击 `启动.cmd`

首次启动依次填写

1. WSL Python 路径，例如 `/home/你的用户名/venvs/qwen-rag/bin/python`
2. 本地 Qwen 模型目录，例如 `D:\Models\Qwen3.5-4B-Dense-style-v1`
3. 本地 E5 目录，例如 `D:\Models\multilingual-e5-small`

Windows 模型目录会转换为 WSL 路径
配置保存在被 Git 忽略的 `config.json` 中
也可以复制 `config.example.json` 后手动填写三个路径

在终端中启动或仅检查环境

```powershell
.\start.ps1
.\start.ps1 --check
.\start.ps1 --configure --check
```

路径也可以通过 `QWEN_WSL_PYTHON`、`QWEN_MODEL_PATH`、`E5_MODEL_PATH` 设置
环境检查验证依赖、CUDA 和本地模型文件，不加载模型权重
首次完整启动需要加载 Qwen 并建立索引，耗时通常为数分钟
界面相关的纯 Python 依赖已放入 `vendor/wheels/`，从本机现有环境整理，无需下载
安装全部前端依赖时可同时指定 `--find-links vendor/wheels` 和自己的 NumPy、PyMuPDF wheel 目录

## 使用

直接输入即可普通聊天，使用模型自身知识并保留当前会话上下文
Enter 发送，Alt+Enter 换行，Tab 补全命令，生成时 Ctrl+C 停止
Ctrl+D 或 `/exit` 退出，`--plain` 使用纯文本界面
聊天上下文只保存在内存，退出后不保留

```text
/chat
/rag
/docs
/docs all
/docs 1
/method bm25
/method vector
/new
/sources
/info
/reindex
/eval
/exit
```

把新的 UTF-8 文本或带文字层的 PDF 放入 `corpus/` 后执行 `/reindex`
`/rag` 进入文档模式，`/chat` 回到普通聊天
文档模式初始范围为 `HAGNN论文.pdf`
`/docs` 列出资料，`/docs all` 检索全部，`/docs 编号` 选择某份资料
选择只影响当前会话，不改写本地配置
作者、标题与所属机构问题会补取所选资料的首页原文，仍须给出引用
首页补充证据不计入 Top-3 命中
文档问答不使用普通聊天历史，普通聊天也不会自动读取文档
需要依据资料回答时请明确切换到文档模式
`/model` 显示真实加载的权重目录，界面更换不改变 Dense 模型
扫描页会提示先进行 OCR

单次问答

```powershell
.\start.ps1 --question "你是谁"
.\start.ps1 --mode rag --method vector --question "论文使用了哪四个数据集" --json
```

## 评测数据

题集包含10道可答题、5道无答案题和5道跨片段题
20题由 AI 起草，经作者审核采用
标准答案不会传入生成模型

| 方法 | 原始回答与分数 | 完整生成记录 |
| --- | --- | --- |
| BM25 | [bm25.csv](eval/results/bm25.csv) | [bm25.json](eval/results/bm25.json) |
| E5 | [vector.csv](eval/results/vector.csv) | [vector.json](eval/results/vector.json) |

CSV 包含问题、标准答案、最终回答、状态、耗时和评分
JSON 保留检索片段、追加上下文、提示词、模型原始输出与计算记录
它们来自同一次完整运行的按方法拆分，没有重试或挑选答案
完整合并原始数据保留在 `eval/run-20261008-102913/predictions.jsonl`

| 方法 | 正确率 | 陷阱题幻觉率 | Top-3命中 | 平均完整延迟 |
| --- | --- | --- | --- | --- |
| BM25 | 16/20 80% | 0/5 | 13/15 | 4.95秒 |
| E5 | 18/20 90% | 0/5 | 13/15 | 5.70秒 |

评分依据是作者抽查意见与作者授权的 AI 逐题复核
原始回答未改写，评分来源保存在 `scoring-provenance.json`
陷阱题无最终交付编造不代表所有回答正确
格式或引用拦截不能算正确拒答
追加上下文不计入 Top-3
这些结果属于同题诊断复测，不是独立测试集成绩
已提交的40条结果来自 v4.3，当前 v4.4 增加首页元信息取证，未重跑或改写这些成绩

在已经加载模型的 CLI 中输入 `/eval` 可重新评测
评测固定使用启动配置中的文档范围，不受 `/docs` 和聊天历史影响
新结果写入新的运行目录，不覆盖已提交的评分
填写新评分后汇总

```powershell
.\.venv\Scripts\python.exe evaluate.py summarize --run run-20261008-102913
```

评分说明见 [最终评分说明](docs/最终评分说明.md)
AI 代码错误及修正记录见 [AI_ERRORS.md](docs/AI_ERRORS.md)

## 检查

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
```

测试不加载生成模型
新增会话测试覆盖多轮上下文、整轮裁剪、取消后续用、文档选择、评测范围恢复与聊天结束符
启动检查与原始问答评测分别记录
运行时日志、向量缓存和本地配置不会提交到 Git

## 资料

论文为作者本人资料
Optimizing Adaptive Context in Hierarchy Aware Graph Neural Networks

DOI https://doi.org/10.1109/CAIBDA70336.2026.11621128
