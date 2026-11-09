<!-- Simplified Chinese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">一个运行实验、部署最佳解决方案并从每次运行中学习的引擎</h4>
<p align="center"><sub>智能实验 · 学习 · 部署 · 领域适配</sub></p>

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
  如果你喜欢这个项目，请给我们一个 star ⭐ 以示支持
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **如果你是正在查看此仓库的 AI 智能体：** Kapso 插件附带用于完成设置、陈述目标、
> 编写 harness 以及审阅运行的技能，一行命令即可安装：
> `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> （Codex：`codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`）。

---

## 新闻

- **IOAI 2026 · AI 模型赛道。** [国际人工智能奥林匹克竞赛](https://ioai-official.org)是 AI 时代的 IMO：来自 108 个国家和地区的 471 名选手，在单 GPU 预算下完成六道由专家设计的任务。2026 年它开设了 [IOAI²](https://ioai-official.org/ai-model-track/)，让 AI 系统在两场无人值守的 6 小时考试中参加*同一份*试卷。计时一旦开始，任何人都不得解答、纠正或改进任何内容。Kapso 作为 14 个创始 AI 参赛者之一参赛。
  - 🎓 **超越全部 471 名人类选手**：总分 536.07，高于赛场上的每一位选手。
  - 🏆 **IOAI² 特级大师奖杯**：在所有参赛 AI 系统中位列前 3。[完整结果 →](../../benchmarks/ioai2026/README.md) · [IOAI 对该赛道的回顾 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **在 [RelBench](../../benchmarks/relbench/README.md) 全部三个排行榜上位列第一**：在 Stanford 面向关系数据预测式机器学习的基准上，Kapso 在官方排行榜的分类（12 项任务上 81.2 AUROC，相比 [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) 的 79.3 和 Stanford STAR 的 [RT-J](https://star-project.stanford.edu/rt-j) 的 79.0）、回归（9 项任务上 0.2476 NMAE，相比 RT-J 的 0.2711 和 KumoRFM-v2 的 0.2847，误差减少 8.7%）和推荐（10 项任务上 18.4 MAP，相比其他最佳条目 ID-GNN 的 9.3）三个方向均领先。每个数字都是[官方 RelBench 排行榜](https://star-project.stanford.edu/relbench/leaderboard/)上的一个条目。保留在 harness 中的学习胜过固化在模型权重中的学习，而更新经验教训库的成本远低于重新训练一个表格基础模型。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)**：一个汇集我们在数据和机器学习领域经验教训的 wiki，约 100k 个页面，通过对论文和代码仓库进行实验提取而来，每月服务约 6M 次请求，其中约 28% 来自 AI 智能体。Kapso 通过 MCP 连接到它：其会话在构思和实现阶段使用它来搜索知识、制定计划和诊断失败。
- **在 [MLE-Bench](../../benchmarks/mle/README.md) 上位列 #1**：KAPSO 在 Kaggle 机器学习竞赛（MLE Benchmark）上取得了开源系统中的最高排名。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **在 [ALE-Bench](../../benchmarks/ale/README.md) 上位列 #1**：KAPSO 在长程算法发现问题（ALE Benchmark）上取得了最高排名。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## Kapso 是什么？

在公司内部交付一个 AI 或数据成果，意味着要通过实验学习两件事。没有任何个人或 AI 智能体知道一个数据资产体系是如何连接和运作的；学习它的唯一方法是调查公司的工件（代码仓库、流水线、文档、数据源），提出假设，运行实验并从中学习，直到理解每个系统的行为及其条件。而即使有了这种理解，最佳解决方案仍然无法预先选定，无论任务是优化数据仓库还是预测客户流失，所以你需要不断实验和学习，直到找到一个适合该环境的方案。适合该环境的知识诀窍会不断积累，正如在强化学习中一样。RAG 则恰恰相反：它向智能体灌输从未在此环境中经过检验的通用知识，所有知识的权重相同，而这些知识在被实际尝试之前几乎没有价值。

Kapso 就是我们为此构建的引擎。它从一个领域 harness 和一个空的经验教训库出发；给定目标和成功标准，它运行一棵实验树，提出假设，然后逐一实现并验证，并将最佳解决方案部署到目标技术栈上。随后它从这次运行中学习，根据每个假设和实现对结果的影响为其分配功劳，并更新经验教训库，库中的每条经验教训都带有可信度分数和一个界定其适用范围的边界。随着时间推移，领域 harness 会携带一个经验教训库，其中积累了在该环境中经过实验验证的最佳实践。该系统与模型无关、与编码智能体无关：开源，由 [Leeroo](https://leeroo.com) 开发，以 [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) 的名称发布在 PyPI 上。

### 四个组成部分

| 组成部分 | 动词 | 作用 |
|------|------|--------------|
| [**智能实验**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 多个专门模块共享同一个数据结构，即实验树：每个节点记录假设、实现（git 分支和技术难点）、已用预算以及与目标的距离。对于每个实验，构思模块根据先前的实验和经验教训库提出想法，按预期回报相对于各自所需预算重新排序，并将排名靠前的想法发送给实现模块，后者在从父节点分叉出的分支上进行构建。由于经验教训库不断学习在该特定环境中什么有效，这些想法会变得针对该环境，而不是裸 LLM 或 RAG 所给出的泛泛之谈。结果由一个带指纹、任何实验都无法访问的评估器进行验证，而评估本身也在演进：实验模块可以提出对指标和标准的更新，交由验证器批准。[工作原理 →](https://docs.leeroo.com/docs/evolve/overview) |
| [**学习**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 库中的一切都来自实验：每条经验教训都是从实验运行的追踪记录中归纳或演绎而来，带有可信度分数、界定其适用范围的边界条件以及背后的证据，并且在每次运行之后，`learn()` 会根据该经验教训对最终结果的影响更新这些内容。对库的更改本身也是实验：每条更新后的经验教训都会通过在有它和没有它的情况下运行同一个构建来做 A/B 测试，因此改进是测量出来的，而不是猜测出来的。该库基于 git 构建，因此你可以与他人分享你的经验（`kapso experience share`）、列出你可用的库（`kapso experience list`）并更新它们（`kapso experience update`）。[经验 →](https://docs.leeroo.com/docs/experience) 指定一次构建所获得的经验来源：你的库、你的团队共享的库、Leeroopedia，或者从代码仓库或研究问题摄取的语料库。 |
| [**部署**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 运行一次更短的同类实验，使最佳解决方案适配目标技术栈：一个会话对节点的副本进行适配、部署它，并演示健康检查和一次可用的调用。内置 Docker、Modal、BentoCloud 和 LangGraph Platform，你也可以把自己的目标技术栈添加到 harness 中。 |
| [**领域适配**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harness 是 Kapso 的领域包：针对某个领域（如预测建模）的知识和工作流，使构建不必从零开始实验和学习；Kapso 自带预测建模和竞赛编程两个 harness。一次运行从一个领域 harness 和一个空的经验教训库开始，随着时间推移，harness 会携带积累了在该环境中行之有效内容的库。使用插件中的 `/kapso:harness` 或 CLI 中的 `kapso harness`，编写你的初始领域 harness，包括你的指令、代码、护栏和检查。 |

## 🚀 快速开始

### 安装

**0. 正在使用编码智能体？** 安装 Kapso 插件，让它陪你一起完成下面的步骤：

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

然后 `/kapso:setup` 负责安装和验证，`/kapso:build <what you want>`
启动第一个活动（在 Codex 上：`$kapso-setup`、`$kapso-build <what you want>`）。参见 [Kapso 插件](https://docs.leeroo.com/docs/coding-agent-skills)。

**1. 前置条件。** Kapso 通过编码智能体 CLI 运行其推理
（没有直接调用 API 的回退方式），因此在一切开始工作之前，
你需要安装 Node.js 并登录两个智能体 CLI：

```bash
# Node.js 18+ (https://nodejs.org)，然后：
npm install -g @openai/codex            # 研究、评判、工具
codex login

npm install -g @anthropic-ai/claude-code  # 构思 + 实现（默认宿主）
claude auth login
```

OpenAI 密钥是可选的：只有知识图谱和对过往实验的语义搜索
（配置中的 `models.embedding`）会对文本做嵌入，
生成过程从不使用它。当你启用其中任一功能时：

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. 连接 Leeroopedia（推荐）。** 一个密钥就能让每个活动会话
使用 [Leeroopedia](https://leeroopedia.com)，这是一个托管的
机器学习和 AI 框架知识库：框架的实际行为、计划评审、
有文档记录的超参数范围、失败诊断。无需安装任何东西；
在 [Leeroopedia 控制台](https://app.leeroopedia.com/dashboard)
新注册的账户附带 $20 的免费额度。

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. 安装软件包**（Python 3.10+）：

```bash
pip install leeroo-kapso
```

软件包名为 `leeroo-kapso`。PyPI 上名为 `kapso` 的软件包是一个无关的
WhatsApp 工具，它会安装自己的 `kapso` 命令并遮蔽本工具的命令；
如果已安装，请先执行 `pip uninstall kapso`。

**4. 验证设置：**

```bash
kapso doctor
```

`doctor` 会报告**你的配置**实际需要什么，并为每一项缺失给出
确切的修复方法。需求随配置而定，因此全 codex 的设置
永远不会被要求安装 `claude`。把它限定到一个动词，就只查看
该动词的需求：

```bash
kapso doctor build             # 或 learn / deploy / experience
```

这些正是该动词在开始任何工作之前自己运行的检查：
`kapso.build(...)` 会在缺少 CLI 时于几秒内失败，而不是在会话
深处才失败。标记为 `[-- ]` 的项是可选的；它们只限制
你可能不需要的功能（docker、Weaviate、Neo4j、部署目标）。

**或者让你的编码智能体完成所有这些：** 将
[设置提示词](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
粘贴到 Claude Code、Codex 或 OpenCode 中。

**索引所需的存储（可选）**：`kapso experience add --repo`
和 `--research` 会将语料库摄取到本地的 Weaviate + Neo4j 中。
在源码检出目录中：

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

旧版 aider 适配器是一个额外组件（`pip install "leeroo-kapso[aider]"`，
Python <3.13）；默认的 claude/codex 智能体不需要任何额外组件。

### 基本用法

核心动词除了上述前置条件外不需要任何其他东西：

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 起步时不需要知识图谱

# Build：通过实验得到一个解决方案。活动在启动时会打印
# `status: <path>`，可在另一个终端中实时观看：
#     kapso watch ./campaign
# 如果某个会话需要只有你能提供的东西（API 密钥、文件），
# 活动会暂停，`kapso inbox reply` 可将其恢复：
#     https://docs.leeroo.com/docs/build/inbox
# 将结果以 Markdown 形式分享到拉取请求或 issue：
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 或省略以从零开始
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 从你刚运行的活动中学习：挖掘轨迹、为经验教训评级，
# 并将按证据定价的知识卡片存入库中。库（一个本地
# git 仓库）在首次使用时自动创建；经验教训会留在你的
# 机器上，直到你分享它们：
#     kapso experience share <git-url>
# 此后每次 learn() 都会将库推送到那里。
lesson = kapso.learn(solution)
print(lesson.explain())

# 再次构建，并提供它刚刚获得的卡片：一次构建所获得的
# 经验恰好就是它所加载的经验。
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

在存储运行的情况下，你还可以将外部知识摄取到索引中，
并以同样的方式提供：

```python
from kapso import Kapso

kapso = Kapso()

# 一次性操作：将一个代码仓库，或针对某个问题进行的网络研究，摄取到索引中
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# 每个加载该索引的构建都会搜索它；Leeroopedia 依靠其密钥即可使用
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

关于预算的说明：一次摄取会提取材料所能支撑的全部内容。
单个问题可能变成几十个相互链接的页面和长达数小时的
摄取；摄取时间随可提取内容的多少而增长，所以请添加
你将要搜索的来源，而不是你拥有的每一个来源。

以及，将解决方案变成运行中的软件：

```python
software = kapso.deploy(solution, target="local")   # 或 docker、modal、bentoml、langgraph、auto
result = software.run(software.proof["inputs"])     # 部署已证明可用的那次调用
software.stop()
```

### 如何限定一次活动？

使用 `max_iterations`、`time_budget_minutes` 和 `cost_budget` 来限定迭代次数、耗时以及以美元计的尽力而为的模型开销。
对于平台成本，在 `search_strategy.params.cost` 中启用[成本感知选择](https://docs.leeroo.com/docs/evolve/cost-aware-selection)，配置你的估算器、`cap_per_candidate` 和 `campaign_budget`；其 `weight` 还能让选择过程偏好更便宜的候选方案。

### 选择模型

Kapso 使用的每个模型都在一个配置文件中指定。打包的默认配置
在 `claude-opus-5` 上运行构建会话，在 `claude-fable-5` 上运行
学习团队，在 `gpt-5.6-sol` 上运行 codex 角色，但模型访问权限
取决于订阅（一个套餐可能限制某个模型而正常提供另一个）。
要在不同的模型上运行，请复制打包的配置、编辑它，并让
Kapso 指向你的配置：

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 例如，让学习团队在 opus 而不是 fable 上运行：
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

纯文本推理角色可以选择使用 `openai_compatible` 的 Chat Completions。
在配置中设置模型和端点；凭据保留在环境变量中。
使用 `Kapso(config=...)` 或 `kapso build --config ...` 加载配置。
编码和网络研究仍然需要具备相应工具的智能体。
参见[配置示例和限制](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)。

在长时间运行之前，请针对你实际的订阅探测配置中指定的
每个模型：你的登录无法提供的模型会在这里于几秒内失败，
而不是在运行数小时后才失败。对于你可以使用的模型，
其用量上限是单 token 探测无法看到的：

```bash
kapso doctor --models                            # 打包的配置
kapso doctor --models --config kapso-config.yaml # 你的配置
kapso doctor learn --models                      # 仅学习团队
```

要让该探测成为每次调用的一部分，而不是需要你记得去运行的
事情，请在配置中启用它。每个不同的 `{cli, model}` 组合会
消耗一个丢弃的 token，在无人值守的数小时 `learn()` 之前
这是值得的：

```yaml
preflight:
  enabled: true            # 每个动词之前的静态检查（默认）
  live_model_probe: true   # + 单 token 模型探测（默认：false）
```

更换模型也会改变*节奏*：配置中团队的 `timeout_minutes` 上限
是按默认模型校准的，换成推理时间更长的模型后可能需要
调高这些上限。

有关详细的集成步骤，请参阅[快速开始](https://docs.leeroo.com/docs/first-build)和[安装](https://docs.leeroo.com/docs/installation)指南。

## 支持的基准测试

| 基准测试 | 描述 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI 的机器学习工程基准：涵盖表格、视觉、文本和音频的完整竞赛，从原始数据到评分提交 |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI 的算法优化基准：在长达数小时的搜索中设计、实现并迭代竞赛启发式算法 |
| [**RelBench**](../../benchmarks/relbench/README.md) | Stanford 面向关系数据预测式机器学习的基准：直接基于 Amazon、H&M、Stack Exchange、临床试验等多表数据库进行分类、回归和推荐 |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 限时的奥林匹克机器学习，涵盖视觉、语言和优化：专家出题、竞赛硬件、零人工帮助 |

每个基准测试还有一个文档页面，介绍如何运行它、其 CLI 选项以及输出包含的内容：
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 示例

每个示例都是一个任务：一份任务简报、一个起始代码仓库、数据或获取数据的方法，以及一个至多十二行的驱动脚本；[examples/README.md](../../examples/README.md) 中列出了所需分钟数、那一条命令以及是否记录了结果。

| 示例 | 展示内容 |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | 你的第一次构建：已有评估器、一个基线、一个目标值、一份记录 |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | 没有评估器：活动的第一个节点根据任务简报中的协议构建一个评估器，随后每个候选方案都依据它来衡量 |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | 一个在 DuckDB 上运行、导出结果带有缺陷的 dbt 流水线：让数据契约得以成立 |
| [**trading**](../../examples/trading/README.md) | 构建上的 harness：一个设置会话、每个会话中的页面和钩子、每次尝试的验证 |
| [**end to end**](../../examples/end-to-end/README.md) | 构建、审阅、学习、带着经验教训再次构建、部署：一个 Kaggle 表格模型的端到端流程 |
| [**prompt harness**](../../examples/prompt-harness/README.md) | 以提示词作为被测对象，在 AIME 题目上评分 |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | 一个视觉语言提取器，对照真值表格评分 |
| [**torch performance**](../../examples/torch-performance/README.md) | 在数值容差范围内，CUDA 上一个更快的模块 |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | CUDA 上的多头注意力：flash attention 或 Triton |

## 📚 文档与支持

- **完整文档**：[docs.leeroo.com](https://docs.leeroo.com/docs)
  - [安装](https://docs.leeroo.com/docs/installation)：编码智能体 CLI、软件包以及 `kapso doctor`
  - [快速开始](https://docs.leeroo.com/docs/first-build)：你的第一个活动
  - [CLI 参考](https://docs.leeroo.com/docs/reference/cli)：每个命令、标志和默认值
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api)：`Kapso` 上的每个公开方法
  - [配置](https://docs.leeroo.com/docs/reference/configuration)：`config.yaml` 中的每个键
  - [learn 产出什么](https://docs.leeroo.com/docs/learn/what-learn-produces)：经验教训库，每条都带有可信度分数、边界及其证据，以及[开发模式](https://docs.leeroo.com/docs/learn/development-mode)
  - [智能实验](https://docs.leeroo.com/docs/evolve/overview)：实验树及其模块，以及[一个实验内部发生了什么](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [经验](https://docs.leeroo.com/docs/experience)：你的库、Leeroopedia，以及转化为可搜索知识的代码仓库和论文
  - [部署](https://docs.leeroo.com/docs/deploy/targets)：local、Docker、Modal 及其他目标
  - [Kapso 插件](https://docs.leeroo.com/docs/coding-agent-skills)：面向 Claude Code、Codex 和 Cursor 的命令、技能和 MCP 服务器
- **从你的编码智能体访问**：文档在 `https://docs.leeroo.com/mcp` 运行着一个 MCP 服务器，并且每个页面都以 Markdown 形式提供，参见 [llms.txt](https://docs.leeroo.com/llms.txt) 和[在你的编码智能体中使用 Kapso](https://docs.leeroo.com/docs/coding-agent-skills)。
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **让你的编码智能体运行 Kapso**：[`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 下的插件包含十个命令、六个技能和 MCP 服务器，每个宿主一行安装命令（见上文第 0 步）；OpenCode 以文件夹形式加载这些技能（`ln -s <clone>/plugin/skills .opencode/skills`）。参见 [Kapso 插件](https://docs.leeroo.com/docs/coding-agent-skills)。
- **为你的编码智能体提供 Kapso 的知识工具**：随软件包一起安装的 `kapso-mcp` 会向你工作的会话提供网络研究、代码仓库的记忆、活动的实验历史以及经验教训库，参见 [MCP 门控](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)。
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI；然后在 ~/.codex/config.toml 的 [mcp_servers.kapso] 下设置 `startup_timeout_sec = 60`（Codex 的十秒在繁忙的机器上不够用）
  ```
- **社区**：[Discord](https://discord.gg/hqVbPNNEZM)
- **官网**：[leeroo.com](https://leeroo.com)


## 企业版 Kapso

在企业中，Kapso 智能体自带初始的数据和 AI 工程 harness 以及一个经验教训库，能够自主地将业务目标转化为可运行的 AI 和数据系统，调查公司的工件、代码仓库、流水线、数据源等，并在此过程中逐步建立企业数据资产体系。要为你的企业任务引入 Kapso，请[联系我们](https://leeroo.com/contact-us)。

## 贡献

我们欢迎贡献！请参阅我们的[贡献指南](../../CONTRIBUTING.md)，了解如何开始。

## 贡献者

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

由 [contrib.rocks](https://contrib.rocks) 生成。
