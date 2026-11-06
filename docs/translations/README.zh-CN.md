<!-- Simplified Chinese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">一个运行实验、部署最佳解决方案并从每次运行中学习的引擎</h4>
<p align="center"><sub>智能实验 · 学习循环 · 部署 · 领域适配</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <b>简体中文</b> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">了解更多</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">加入 Discord</a> ·
  <a href="https://leeroo.com">官网</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  如果你喜欢这个项目，请给它点一颗星来支持我们 ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **如果你是正在查看此仓库的 AI 智能体：** Kapso 插件带有用于完成设置、陈述目标、编写 harness 和审查运行的技能，只需一行即可安装：`claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`（Codex：`codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`）。

---

## 新闻

- **IOAI 2026 · AI 模型赛道。** [国际人工智能奥林匹克竞赛](https://ioai-official.org)是 AI 时代的 IMO：471 名选手来自 108 个国家和地区，在单 GPU 预算下完成六道由专家设计的题目。2026 年它开设了 [IOAI²](https://ioai-official.org/ai-model-track/)，让 AI 系统在两场无人值守的 6 小时考试中参加*同一场*考试。计时一旦开始，任何人都不得解题、纠正或改进任何内容。Kapso 作为 14 个创始 AI 参赛者之一参赛。
  - 🎓 **得分超过全部 471 名人类选手**：总分 536.07，高于考场内的每一位选手。
  - 🏆 **IOAI² 特级大师奖杯**：在所有参赛的 AI 系统中位列前 3。[完整结果 →](../../benchmarks/ioai2026/README.md) · [IOAI 对该赛道的回顾 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **在全部三个 [RelBench](../../benchmarks/relbench/README.md) 排行榜上均列第一**：在 Stanford 面向关系数据预测性 ML 的基准上，Kapso 在官方排行榜上领跑分类（81.2 AUROC，覆盖 12 项任务，对比 [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) 的 79.3 和 Stanford STAR 的 RT-J 的 79.0）、回归（0.2476 NMAE，覆盖 9 项任务，对比 RT-J 的 0.2711 和 KumoRFM-v2 的 0.2847，误差减少 8.7%）和推荐（18.4 MAP，覆盖 10 项任务，对比 ID-GNN 的 9.3，后者是其余条目中的最佳者）。每个数字都是[官方 RelBench 排行榜](https://star-project.stanford.edu/relbench/leaderboard/)上的一条记录。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)**：一个汇集我们在数据和 ML 领域经验教训的 wiki，约 100k 个页面通过对论文和仓库进行实验提取而来，每月处理约 6M 次请求，其中约 28% 来自 AI 智能体。Kapso 通过 MCP 与它连接：其会话在构思和实现阶段用它来搜索知识、制定计划和诊断故障。
- **[Moltbook 智能体 🦞](https://www.moltbook.com/)**：构建能够优化其他智能体并在 Moltbook 上辩论的 AI 智能体！[开始使用 →](../../moltbook_bot/README.md)
- **技术报告**：我们的技术报告现已发布！[阅读论文](https://arxiv.org/abs/2601.21526)
- **在 [MLE-Bench](../../benchmarks/mle/README.md) 上排名 #1**：KAPSO 在 Kaggle ML 竞赛（MLE Benchmark）的开源系统中取得了最高排名。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **在 [ALE-Bench](../../benchmarks/ale/README.md) 上排名 #1**：KAPSO 在长周期算法发现问题（ALE Benchmark）上取得了最高排名。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## Kapso 是什么？

Kapso 是一个运行实验、部署最佳解决方案并从每次运行中学习的引擎。给定一个目标、初始成功标准和评估指标，它会展开调研，挑选最佳想法来运行实验，有时还会给出反馈以改进评估指标和成功标准，在指定的技术栈上部署最佳解决方案，并从每次运行中学习。它维护一个经验教训库，每条经验教训都带有可信度评分和界定其适用范围的边界，因此知识会随时间产生复利，而不只是不断累积。

我们曾在 LinkedIn 和 Meta AI 等大公司内部构建 AI 系统，并将其部署到拥有数百万用户的产品中。Kapso 就是把这套工作流做成了引擎：开源，由 [Leeroo](https://leeroo.com) 出品，以 [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) 之名发布在 PyPI 上。

### 四个部分

| 部分 | 动词 | 作用 |
|------|------|--------------|
| [**智能实验**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 实验循环是一组专用模块，它们共享同一个数据结构，即实验树：每个节点记录假设、实现（git 分支和技术难点）、已用预算以及与目标的距离。构思环节借鉴此前的实验和经验教训库来提出想法，每个想法都说明它与哪些具体的早期实验有何不同、采纳了哪些经验教训；重排序器按相对于目标值的预期回报对它们排序，并结合时间和预算加以权衡。实现器在从父节点分叉出的分支上进行构建，验证器则对照安全规则和成功标准检查该节点，判定分数是否有效、目标是否达成。成功标准和测试可以在运行过程中演进：各模块提出有理有据的更新，由验证器批准或驳回。[工作原理 →](https://docs.leeroo.com/docs/evolve/overview) |
| [**学习循环**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 每条经验教训都记录一项知识、一个可信度评分、界定其适用范围的边界条件，以及支撑其历次更新的证据。每次运行之后，`learn()` 会根据所用经验教训对结果的影响来更新它们的可信度和边界，从相似的追踪记录中归纳出经验教训，或从单条证据及其机制中演绎出一条，合并重叠的经验教训，并通过在使用和不使用这些经验教训的情况下运行同一构建，对更新后的经验教训进行 A/B 测试，因此更新是测量出来的，而不是猜出来的。下一次构建的构思和实现会读取每条经验教训的可信度和边界，以决定何时依赖它，而不是简单地对知识做 RAG。[经验 →](https://docs.leeroo.com/docs/experience) 说明提供给一次构建的是什么：你的库、你的团队共享的库、Leeroopedia，或者从某个仓库或某个经过调研的问题摄取而来的语料库。 |
| [**部署**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 通过一个精简版的实验循环，把树中的最佳解决方案变成在目标技术栈上运行的软件：一个会话对节点的副本进行适配、部署，并演示健康检查和一次可用的调用。Docker、Modal、BentoCloud 和 LangGraph Platform 已随附提供，你也可以把自己的目标技术栈添加到 harness 中。 |
| [**领域适配**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harness 是 Kapso 的领域包：针对预测建模等某一领域的知识和工作流，使构建不必从零开始实验和学习；预测建模和竞技编程随 Kapso 一同提供。要引入你自己的领域知识，可通过 Kapso 定义的接口编写一个 harness：你的指令、代码、护栏和检查，在插件中用 `/kapso:harness` 编写，或在 CLI 中用 `kapso harness` 编写。 |

## 🚀 快速开始

### 安装

**0. 在编程智能体里？** 安装 Kapso 插件，让它陪你走完下面的步骤：

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

随后 `/kapso:setup` 完成安装和验证，`/kapso:build <what you want>` 启动第一个战役（在 Codex 上：`$kapso-setup`、`$kapso-build <what you want>`）。参见 [Kapso 插件](https://docs.leeroo.com/docs/coding-agent-skills)。

**1. 前置条件。** Kapso 通过编程智能体 CLI 进行推理（没有直接调用 API 的后备方案），因此在一切开始工作之前，你需要 Node.js，并且两个智能体 CLI 都已登录：

```bash
# Node.js 18+ (https://nodejs.org)，然后：
npm install -g @openai/codex            # 调研、评判、实用工具
codex login

npm install -g @anthropic-ai/claude-code  # 构思 + 实现（默认宿主）
claude auth login
```

OpenAI 密钥是可选的：只有知识图谱和对过往实验的语义搜索（配置中的 `models.embedding`）会对文本做嵌入，生成环节从不使用它。当你开启其中任意一项时：

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. 连接 Leeroopedia（推荐）。** 一个密钥就能让每个战役会话用上 [Leeroopedia](https://leeroopedia.com)，这是一个托管的 ML 和 AI 框架知识库：框架的实际行为、计划评审、有据可查的超参数范围、故障诊断。无需安装任何东西；在 [Leeroopedia 控制台](https://app.leeroopedia.com/dashboard)新注册的账户附带 $20 的免费额度。

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. 安装软件包**（Python 3.10+）：

```bash
pip install leeroo-kapso
```

软件包名为 `leeroo-kapso`。PyPI 上名为 `kapso` 的软件包是一个无关的 WhatsApp 工具，它会安装自己的 `kapso` 命令并遮蔽本项目的命令；如果已安装它，请先执行 `pip uninstall kapso`。

**4. 验证安装：**

```bash
kapso doctor
```

`doctor` 会报告**你的配置**实际需要什么，并针对任何缺失项给出确切的修复方法。要求随配置而定，因此全 codex 的配置绝不会被要求提供 `claude`。把它限定到某一个动词，即可只查看该动词的要求：

```bash
kapso doctor build             # 或 learn / deploy / experience
```

这些正是该动词自身在开始任何工作之前运行的检查：缺少某个 CLI 时，`kapso.build(...)` 会在几秒内失败，而不是在会话深处才失败。标有 `[-- ]` 的项目是可选的；它们限制的是你可能并不需要的功能（docker、Weaviate、Neo4j、各部署目标）。

**或者让你的编程智能体完成这一切：** 把[安装提示词](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)粘贴到 Claude Code、Codex 或 OpenCode 中。

**索引所需的存储（可选）**：`kapso experience add --repo` 和 `--research` 会把语料库摄取到本地的 Weaviate + Neo4j 中。在源码检出目录中执行：

```bash
bash scripts/start_infra.sh   # 通过 docker 启动两者
```

**从源码安装（用于开发）**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

旧版 aider 适配器是一个可选附加项（`pip install "leeroo-kapso[aider]"`，Python <3.13）；默认的 claude/codex 智能体不需要任何附加项。

### 基本用法

核心循环只需要上面的前置条件，别无其他：

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 起步时不需要知识图谱

# Build：通过实验得到一个解决方案。战役会在启动时打印
# `status: <path>`，可以在另一个终端中实时观察它：
#     kapso watch ./campaign
# 如果某个会话需要只有你才能提供的东西（一个 API 密钥、一个文件），
# 战役会暂停，由 `kapso inbox reply` 恢复它：
#     https://docs.leeroo.com/docs/build/inbox
# 用下面的命令把结果以 Markdown 形式分享到 pull request 或 issue 中：
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 或省略，从零开始
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 从你刚运行的战役中学习：挖掘轨迹，为经验教训
# 评级，并把按证据定价的知识卡片存入库中。库（一个本地
# git 仓库）在首次使用时自动创建，经验教训会留在你的
# 机器上，直到你分享它们：
#     kapso experience share <git-url>
# 此后每次 learn() 都会把库推送到那里。
lesson = kapso.learn(solution)
print(lesson.explain())

# 再次构建，这次会获得它刚挣得的卡片：一次构建获得的正是
# 它所加载的经验。
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

存储运行起来之后，你还可以把外部知识摄取到一个索引中，并以同样的方式提供给构建：

```python
from kapso import Kapso

kapso = Kapso()

# 只需一次：把一个仓库，或针对某个问题调研得到的网络资料，摄取到一个索引中
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# 每个加载了该索引的构建都会搜索它；Leeroopedia 则凭它的密钥接入
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

关于预算的一点说明：一次摄取会提取材料所能支撑的全部内容。一个问题就可能变成数十个相互链接的页面和长达数小时的摄取；摄取时间随可提取的实质内容而增长，因此请添加你会去搜索的来源，而不是你手头的每一个来源。

要把一个解决方案变成可运行的软件：

```python
software = kapso.deploy(solution, target="local")   # 或 docker、modal、bentoml、langgraph、auto
result = software.run(software.proof["inputs"])     # 部署已验证过的那次调用
software.stop()
```

### 如何为战役设限？

使用 `max_iterations`、`time_budget_minutes` 和 `cost_budget` 来限制迭代次数、已用时间以及尽力而为估算的模型花费（以 USD 计）。
对于平台成本，请在 `search_strategy.params.cost` 中启用[成本感知选择](https://docs.leeroo.com/docs/evolve/cost-aware-selection)，并提供你的估算器、`cap_per_candidate` 和 `campaign_budget`；它的 `weight` 还能让选择过程偏向更便宜的候选方案。

### 选择模型

Kapso 使用的每个模型都在一个配置文件中指定。随包附带的默认配置在 `claude-opus-5` 上运行构建会话，在 `claude-fable-5` 上运行学习小组，在 `gpt-5.6-sol` 上运行 codex 角色，不过模型的访问权限取决于订阅（一个套餐可能对某个模型设有上限，同时又提供另一个模型）。要在不同的模型上运行，请复制随包附带的配置，进行编辑，然后让 Kapso 指向你的配置：

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 例如，在 opus 而不是 fable 上运行学习小组：
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

纯文本推理角色可以选择启用 `openai_compatible` Chat Completions。
在你的配置中设置模型和端点；凭据留在环境变量中。
使用 `Kapso(config=...)` 或 `kapso build --config ...` 加载该配置。
编程和网络调研仍然需要具备相应工具的智能体。
参见[配置示例和限制](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)。

在长时间运行之前，请对照你的实际订阅，探测配置中指定的每个模型：你的登录账户无法提供的模型会在这里几秒内失败，而不是在运行数小时之后才失败。对于你能够使用的模型，其用量上限是单 token 探测看不到的：

```bash
kapso doctor --models                            # 随包附带的配置
kapso doctor --models --config kapso-config.yaml # 你的配置
kapso doctor learn --models                      # 仅学习小组
```

要让这项探测成为每次调用的一部分，而不是一件需要你记着去运行的事，请在配置中开启它。它为每个不同的 `{cli, model}` 组合花费一个用完即弃的 token，在一次无人值守、长达数小时的 `learn()` 之前，这是值得的：

```yaml
preflight:
  enabled: true            # 每个动词执行前的静态检查（默认）
  live_model_probe: true   # + 单 token 模型探测（默认：false）
```

更换模型也会改变*节奏*：配置中各小组的 `timeout_minutes` 上限是按默认模型校准的，换上的模型如果推理时间更长，可能需要调高这些上限。

详细的集成步骤，请参见[快速开始](https://docs.leeroo.com/docs/first-build)和[安装](https://docs.leeroo.com/docs/installation)指南。

## 示例

每个示例都是一项任务：一份简报、一个起始仓库、数据或获取数据的方法，以及一个至多十二行的驱动脚本；[examples/README.md](../../examples/README.md) 列出了所需的分钟数、那一条命令以及是否记录了结果。

| 示例 | 展示内容 |
|---------|---------------|
| [**流失预测：首次构建**](../../examples/churn-first-build/README.md) | 你的第一次构建：已有一个评估器、一条基线、一个目标值，以及记录 |
| [**欺诈检测：无评估器**](../../examples/fraud-no-evaluator/README.md) | 没有评估器：战役的第一个节点根据简报中的协议构建一个，每个候选方案都用它来衡量 |
| [**dbt 管道**](../../examples/pipeline-dbt/README.md) | 一条运行在 DuckDB 上、导出数据带有缺陷的 dbt 管道：让数据契约得以成立 |
| [**交易**](../../examples/trading/README.md) | 构建上挂载一个 harness：一个设置会话、每个会话中的页面和钩子、每次尝试的验证 |
| [**完整循环**](../../examples/full-loop/README.md) | build、审查、learn、带着该经验教训再次 build、deploy：一个端到端的 Kaggle 表格模型 |
| [**提示词 harness**](../../examples/prompt-harness/README.md) | 把提示词作为被测对象，在 AIME 题目上评分 |
| [**智能体脚手架**](../../examples/agent-scaffold/README.md) | 一个视觉语言提取器，对照真值表格评分 |
| [**torch 性能**](../../examples/torch-performance/README.md) | 在数值容差范围内，一个在 CUDA 上更快的模块 |
| [**GPU 内核**](../../examples/gpu-kernels/README.md) | CUDA 上的多头注意力：flash attention 或 Triton |

## 支持的基准测试

| 基准测试 | 说明 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI 的 ML 工程基准：涵盖表格、视觉、文本和音频的完整竞赛，从原始数据到评分提交 |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI 的算法优化基准：在长达数小时的搜索中设计、实现并迭代竞赛启发式算法 |
| [**RelBench**](../../benchmarks/relbench/README.md) | Stanford 面向关系数据预测性 ML 的基准：直接基于 Amazon、H&M、Stack Exchange、临床试验等的多表数据库进行分类、回归和推荐 |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 涵盖视觉、语言和优化的限时奥赛 ML：专家出题、竞赛硬件、零人工帮助 |

每个基准测试还有一个文档页面，介绍如何运行它、它的 CLI 选项以及输出包含什么：
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 文档与支持

- **完整文档**：[docs.leeroo.com](https://docs.leeroo.com/docs)
  - [安装](https://docs.leeroo.com/docs/installation)：编程智能体 CLI、软件包以及 `kapso doctor`
  - [快速开始](https://docs.leeroo.com/docs/first-build)：你的第一个战役
  - [CLI 参考](https://docs.leeroo.com/docs/reference/cli)：每个命令、标志和默认值
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api)：`Kapso` 上的每个公开方法
  - [配置](https://docs.leeroo.com/docs/reference/configuration)：`config.yaml` 中的每个键
  - [learn 的产出](https://docs.leeroo.com/docs/learn/what-learn-produces)：经验教训库，其中每条都带有可信度评分、边界及其证据，以及[开发模式](https://docs.leeroo.com/docs/learn/development-mode)
  - [实验循环](https://docs.leeroo.com/docs/evolve/overview)：实验树及其模块，以及[一次实验内部发生了什么](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [经验](https://docs.leeroo.com/docs/experience)：你的库、Leeroopedia，以及转化为可搜索知识的仓库和论文
  - [部署](https://docs.leeroo.com/docs/deploy/targets)：本地、Docker、Modal 及其他目标
  - [Kapso 插件](https://docs.leeroo.com/docs/coding-agent-skills)：面向 Claude Code、Codex 和 Cursor 的命令、技能和 MCP 服务器
- **从你的编程智能体中使用**：文档站在 `https://docs.leeroo.com/mcp` 运行着一个 MCP 服务器，并且每个页面都以 Markdown 形式提供，参见 [llms.txt](https://docs.leeroo.com/llms.txt) 和[在你的编程智能体中使用 Kapso](https://docs.leeroo.com/docs/coding-agent-skills)。
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **让你的编程智能体运行 Kapso**：[`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 下的插件带有十条命令、六个技能和 MCP 服务器，每个宿主只需一行安装命令（上面的第 0 步）；OpenCode 以文件夹的形式加载这些技能（`ln -s <clone>/plugin/skills .opencode/skills`）。参见 [Kapso 插件](https://docs.leeroo.com/docs/coding-agent-skills)。
- **为你的编程智能体提供 Kapso 的知识工具**：随软件包一同安装的 `kapso-mcp`，会向你所在的会话提供网络调研、仓库的记忆、战役的实验历史和经验教训库，参见 [MCP 关口](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)。
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI；然后在 ~/.codex/config.toml 的 [mcp_servers.kapso] 下设置 `startup_timeout_sec = 60`（在繁忙的机器上，Codex 的十秒太短）
  ```
- **社区**：[Discord](https://discord.gg/hqVbPNNEZM)
- **官网**：[leeroo.com](https://leeroo.com)


## 面向企业的 Kapso

我们的企业产品构建在这个引擎之上，并已部署在错综复杂的企业内部，在那里没有任何人能够调研并理解整个数据资产：我们曾帮助一家公司从旧的数据平台迁移到新的湖仓，涉及 30 多个源系统族、数千张表和转换，以及数百份 BI 报表。它处理的每一项任务都会充实关于你的系统、你的数据和你来之不易的规则的经验教训库，因此它工作得越久，就越了解你的公司。若想引入 Kapso 来应对你们富有挑战性的企业任务，请[与我们联系](https://leeroo.com/contact-us)。

## 翻译

README 提供其他语言版本：使用顶部的切换器，或浏览 [`docs/translations/`](../../docs/translations/)。每个译本都由 `scripts/translate_readme.py` 根据 README.md 生成，并由 `scripts/check_readme_translations.py` 保持同步；当某个译本与 README.md 在某个链接、某个数字或某行代码上不一致时，该脚本会报错失败。如需改进某个译本，请针对该文件夹中的对应文件提交 pull request。

## 贡献

我们欢迎贡献！有关如何上手的详细信息，请参阅我们的[贡献指南](../../CONTRIBUTING.md)。

## 引用

如果你在工作中使用了 Kapso，请引用这份技术报告：

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
