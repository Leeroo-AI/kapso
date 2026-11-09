<!-- Traditional Chinese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">一個執行實驗、部署最佳解決方案，並從每次執行中學習的引擎</h4>
<p align="center"><sub>智慧實驗 · 學習 · 部署 · 領域適應</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <b>繁體中文</b>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">了解更多</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">加入 Discord</a> ·
  <a href="https://leeroo.com">官方網站</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  如果你喜歡這個專案，請給我們一顆星 ⭐ 以示支持
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **如果你是正在查看此儲存庫的 AI 代理：** Kapso 外掛包含用於
> 設定、陳述目標、撰寫 harness 以及審閱執行結果的技能，只需一行即可
> 安裝：`claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> （Codex：`codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`）。

---

## 最新消息

- **IOAI 2026 · AI 模型賽道。** [國際人工智慧奧林匹亞](https://ioai-official.org)是 AI 時代的 IMO：471 位參賽者、六項由專家設計的任務、單一 GPU 的預算。其 2026 年的 [IOAI²](https://ioai-official.org/ai-model-track/) 賽道讓 AI 系統在一次自主進行的賽程中接受同一份試題。Kapso 作為 14 個創始 AI 參賽者之一，依主辦單位評分取得 536.07 分，高於所有人類參賽者，並奪得 IOAI² 特級大師獎盃，在所有參賽的 AI 系統中名列前 3。[完整結果 →](../../benchmarks/ioai2026/README.md) · [IOAI 對本賽道的回顧 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **在 [RelBench](../../benchmarks/relbench/README.md) 三個排行榜上全部位居第一。** 在 Stanford 針對關聯式資料預測式機器學習的基準測試上，Kapso 在[官方排行榜](https://star-project.stanford.edu/relbench/leaderboard/)的分類（81.2 AUROC，對比 [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) 的 79.3 與 Stanford STAR 的 [RT-J](https://star-project.stanford.edu/rt-j) 的 79.0）、迴歸（0.2476 NMAE，對比 0.2711 與 0.2847）以及推薦（18.4 MAP，對比其他最佳參賽者 ID-GNN 的 9.3）三項中均領先。

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0 in classification; 0.2476 NMAE against 0.2711 and 0.2847 in regression, lower being better; 18.4 MAP against ID-GNN's 9.3 in recommendations. The first two panels use a truncated axis, the third starts at zero" width="574"></p>

- **[Leeroopedia](https://leeroopedia.com)**：我們在資料與機器學習領域的公開教訓庫，約有 100k 個頁面，透過對論文與儲存庫進行實驗所提取而成，每月服務約 6M 次請求，其中約 28% 來自 AI 代理。Kapso 的工作階段在構思與實作期間會透過 MCP 查詢它。
- **在 [MLE-Bench](../../benchmarks/mle/README.md) 上排名 #1**：KAPSO 在 Kaggle 機器學習競賽（MLE 基準測試）上，於開源系統中取得最高排名。

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574"></p>

- **在 [ALE-Bench](../../benchmarks/ale/README.md) 上排名 #1**：KAPSO 在長程演算法發現問題（ALE 基準測試）上取得最高排名。

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574"></p>

## Kapso 是什麼？

Kapso 是一個開源引擎，它執行實驗、部署最佳解決方案，並從每次執行中學習。它從一個領域 harness 與一個空的教訓庫出發，在給定目標與成功標準後，它會：

- **執行一棵實驗樹**：提出假設，然後逐一實作並驗證；
- **在目標技術堆疊上部署最佳解決方案**；
- **從執行中學習**：依據每個假設與實作對結果的影響給予其功勞，並更新教訓庫，每條教訓都帶有可信度分數以及其適用範圍的邊界。

隨著時間推移，領域 harness 會承載一個教訓庫：該環境中經過實驗驗證的最佳實務。Kapso 不依賴特定模型，也不依賴特定程式編寫代理，由 [Leeroo](https://leeroo.com) 開發，以 [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) 之名發佈於 PyPI。

### 四個部分

| 部分 | 動詞 | 作用 |
|------|------|--------------|
| [**智慧實驗**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 每個實驗都是一個在自己的分支上實作並評分的假設，組成一棵樹。構思階段借鑑先前的實驗與教訓庫，並依預期回報對照預算重新排序想法，使其貼合此環境，而非僅是單純 LLM 或 RAG 的泛用想法；任何實驗都無法碰觸的帶指紋評估器負責驗證結果，模組則可提出評估更新，交由驗證者核准。[運作方式 →](https://docs.leeroo.com/docs/evolve/overview) |
| [**學習**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 每條教訓都是從實驗追蹤紀錄中歸納或演繹而來，並帶有可信度分數、其適用範圍的邊界及其證據；`learn()` 會依每次執行的結果更新它們，並透過分別在有無該變更的情況下重新 build 來對每項變更進行 A/B 測試，因此改進是量測出來的，而非猜測。教訓庫建立在 git 之上：`kapso experience share`、`list` 與 `update`。[Experience →](https://docs.leeroo.com/docs/experience) 說明一次 build 會被提供哪些內容。 |
| [**部署**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 一段同類型但較短的實驗，將最佳解決方案適配到目標技術堆疊，加以部署，並證明健康檢查與一次實際呼叫皆可運作。內建 Docker、Modal、BentoCloud 與 LangGraph Platform；harness 可以新增目標。 |
| [**領域適應**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harness 是 Kapso 的領域套件：某個領域（例如預測式建模）的知識與工作流程，讓 build 永遠不必從零開始，也是教訓庫累積之處。內建預測式建模與競技程式設計；可在外掛中以 `/kapso:harness` 或在 CLI 中以 `kapso harness` 撰寫你自己的 harness。 |

## 🚀 快速入門

### 安裝

**0. 正在使用程式編寫代理嗎？** 安裝 Kapso 外掛，讓它陪你走完
下列步驟：

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

接著 `/kapso:setup` 會進行安裝與驗證，而 `/kapso:build <what you want>`
會啟動第一場戰役（在 Codex 上：`$kapso-setup`、`$kapso-build <what you want>`）。參見 [Kapso 外掛](https://docs.leeroo.com/docs/coding-agent-skills)。

**1. 先決條件。** Kapso 透過程式編寫代理的 CLI 執行推論
（沒有直接 API 的備援），因此在一切能運作之前，你需要 Node.js，
並且兩個代理 CLI 都已登入：

```bash
# Node.js 18+（https://nodejs.org），然後：
npm install -g @openai/codex            # 研究、評判、工具
codex login

npm install -g @anthropic-ai/claude-code  # 構思 + 實作（預設主機）
claude auth login
```

OpenAI 金鑰為選用：只有知識圖譜與對過往實驗的語意搜尋
（設定中的 `models.embedding`）會嵌入文字，生成則從不使用它。
當你啟用其中任一項時：

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. 連接 Leeroopedia（建議）。** 一把金鑰就能讓每個工作階段
使用 [Leeroopedia](https://leeroopedia.com)，我們託管的機器學習與
AI 框架教訓庫。無需安裝任何東西；在
[Leeroopedia 儀表板](https://app.leeroopedia.com/dashboard)新建帳號即附贈
$20 的免費額度。

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. 安裝套件**（Python 3.10+）：

```bash
pip install leeroo-kapso
```

**4. 驗證設定：**

```bash
kapso doctor
```

`doctor` 會回報**你的設定**實際需要什麼，並對任何缺失項目給出
確切的修復方法。需求跟隨設定而定，因此全 codex 的
設定永遠不會被要求安裝 `claude`。將範圍縮小到單一動詞，即可只檢視該
動詞的需求：

```bash
kapso doctor build             # 或 learn / deploy / experience
```

這些正是該動詞在開始任何工作前自己會執行的檢查：
`kapso.build(...)` 會在數秒內因缺少 CLI 而失敗，而不是在工作階段
深處才失敗。標記為 `[-- ]` 的項目為選用；它們只會限制你可能
用不到的功能（docker、Weaviate、Neo4j、部署目標）。

**或者讓你的程式編寫代理代勞：** 將
[設定提示詞](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
貼到 Claude Code、Codex 或 OpenCode 中。

**索引所需的資料儲存（選用）**：`kapso experience add --repo`
與 `--research` 會將語料庫匯入本機的 Weaviate + Neo4j。從
原始碼簽出目錄執行：

```bash
bash scripts/start_infra.sh   # 透過 docker 啟動兩者
```

**從原始碼安裝（供開發用）**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

舊版的 aider 轉接器屬於額外套件（`pip install "leeroo-kapso[aider]"`，
Python <3.13）；預設的 claude/codex 代理不需要任何額外套件。

### 基本用法

核心動詞除了上述先決條件外不需要其他東西：

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 起步不需要知識圖譜

# Build：透過實驗得到一個解決方案。戰役啟動時會印出
# `status: <path>`，可從另一個終端機即時觀看：
#     kapso watch ./campaign
# 若某個工作階段需要只有你能提供的東西（API 金鑰、檔案），
# 戰役會暫停，而 `kapso inbox reply` 會讓它繼續：
#     https://docs.leeroo.com/docs/build/inbox
# 將結果以 Markdown 分享到 pull request 或 issue，使用：
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 或省略以從零開始
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 從你剛執行的戰役中學習：挖掘軌跡、為教訓評分，
# 並將以證據定價的知識卡存入教訓庫。教訓庫（一個本機
# git 儲存庫）會在首次使用時自動建立，教訓在你分享之前
# 都留在你的機器上：
#     kapso experience share <git-url>
# 之後每次 learn() 都會將教訓庫推送到那裡。
lesson = kapso.learn(solution)
print(lesson.explain())

# 再次 build，這次會提供它剛贏得的卡片：一次 build 得到的
# 正是它所載入的經驗。
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

資料儲存運行後，你也可以將外部知識匯入
索引，並以同樣方式提供：

```python
from kapso import Kapso

kapso = Kapso()

# 一次性：將一個儲存庫，或針對某個問題研究的網路內容，匯入索引
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# 每個載入該索引的 build 都會搜尋它；Leeroopedia 則憑其金鑰接入
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

關於預算的一點說明：一次匯入會提取素材所能支撐的所有內容。
單一個問題可能變成數十個相互連結的頁面以及長達數小時的
匯入；匯入時間隨可提取的實質內容而增長，因此請加入
你會搜尋的來源，而非你擁有的所有來源。

接著，將解決方案變成運行中的軟體：

```python
software = kapso.deploy(solution, target="local")   # 或 docker、modal、bentoml、langgraph、auto
result = software.run(software.proof["inputs"])     # 部署已證明可行的那次呼叫
software.stop()
```

### 如何限制一場戰役？

使用 `max_iterations`、`time_budget_minutes` 與 `cost_budget` 來限制迭代次數、經過時間，以及以美元計的盡力而為模型花費。
對於平台成本，請在 `search_strategy.params.cost` 中啟用[成本感知選擇](https://docs.leeroo.com/docs/evolve/cost-aware-selection)，搭配你的估算器、`cap_per_candidate` 與 `campaign_budget`；其 `weight` 也能讓選擇偏好較便宜的候選方案。

### 選擇模型

Kapso 使用的每個模型都在單一設定檔中指名。套件內建的預設值
在 `claude-opus-5` 上執行 build 工作階段，在
`claude-fable-5` 上執行學習團隊，codex 角色則使用 `gpt-5.6-sol`，但模型存取
取決於訂閱方案（一個方案可能限制某個模型卻提供另一個）。
若要改用不同的模型，請複製套件內建設定、編輯，然後將
Kapso 指向你的設定：

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 例如：改在 opus 而非 fable 上執行學習團隊：
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

純文字推論角色可以選擇使用 `openai_compatible` 的 Chat Completions。
在設定中指定模型與端點；憑證則保留在環境變數中。
以 `Kapso(config=...)` 或 `kapso build --config ...` 載入設定。
程式編寫與網路研究仍需要具備相應工具的代理。
參見[設定範例與限制](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)。

在長時間執行之前，請針對你實際的訂閱方案探測設定中指名的每個模型：
你的登入身分無法提供的模型會在此數秒內失敗，而不是在
執行數小時後才失敗。對於你能使用的模型，其用量上限
無法透過單一 token 的探測得知：

```bash
kapso doctor --models                            # 套件內建設定
kapso doctor --models --config kapso-config.yaml # 你的設定
kapso doctor learn --models                      # 僅限學習團隊
```

若要讓這項探測成為每次呼叫的一部分，而不是得靠你記得去執行，
請在設定中啟用它。每個不同的 `{cli, model}` 組合會花費一個
用後即棄的 token，在無人看管的多小時 `learn()` 之前這是值得的：

```yaml
preflight:
  enabled: true            # 每個動詞執行前的靜態檢查（預設）
  live_model_probe: true   # + 單一 token 的模型探測（預設：false）
```

更換模型也會改變*節奏*：設定中的團隊 `timeout_minutes` 上限
是依預設模型校準的，換成推理時間更長的模型
可能需要調高這些上限。

詳細的整合步驟請參見[快速入門](https://docs.leeroo.com/docs/first-build)與[安裝](https://docs.leeroo.com/docs/installation)指南。

## 支援的基準測試

| 基準測試 | 說明 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI 的機器學習工程基準測試：涵蓋表格、視覺、文字與音訊的完整競賽，從原始資料到評分後的提交 |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI 的演算法最佳化基準測試：在長達數小時的搜尋中設計、實作並迭代競賽啟發式演算法 |
| [**RelBench**](../../benchmarks/relbench/README.md) | Stanford 針對關聯式資料預測式機器學習的基準測試：直接從 Amazon、H&M、Stack Exchange、臨床試驗等多表格資料庫進行分類、迴歸與推薦 |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 橫跨視覺、語言與最佳化的限時奧林匹亞機器學習：專家出題、競賽硬體、零人工協助 |

每個基準測試也都有一個文件頁面，說明如何執行、其 CLI 選項以及輸出包含的內容：
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 範例

每個範例都是一項任務：一份簡報、一個起始儲存庫、資料或取得資料的方式，以及一支至多十二行的驅動程式；[examples/README.md](../../examples/README.md) 列有所需分鐘數、那一行指令，以及是否有記錄結果。

| 範例 | 任務 |
|---------|----------|
| [**預測式建模（客戶流失）**](../../examples/churn-first-build/README.md) | 預測哪些客戶將會離開：在戰役無法編輯的評估器下，將流失模型的樣本外 ROC AUC 提升至目標。 |
| [**詐欺偵測**](../../examples/fraud-no-evaluator/README.md) | 在儲存庫沒有評估器的情況下為交易評定詐欺分數：簡報陳述協定，戰役先建立評估器，再改進評分器。 |
| [**資料工程（dbt 模型）**](../../examples/pipeline-dbt/README.md) | 一個建立在 dbt 與 DuckDB 上的零售資料倉儲，其每夜匯出的資料帶有缺陷。 |
| [**量化金融**](../../examples/trading/README.md) | 針對合成金融商品的每日交易策略：在交易 harness 上提升樣本外 Sharpe 值，每次嘗試都經過其檢查。這是研究範例，並非財務建議。 |
| [**表格式機器學習**](../../examples/end-to-end/README.md) | 一場 Kaggle 競賽從頭到尾：Spaceship Titanic 從虛擬分類器到真正的模型，然後審閱、學習、帶著教訓再次 build，並部署。 |
| [**提示詞工程**](../../examples/prompt-harness/README.md) | 改寫 LLM 提示詞範本，直到在 AIME 2024 數學題上的準確率提升。 |
| [**PyTorch 最佳化**](../../examples/torch-performance/README.md) | 在數值容差內讓 PyTorch 模組在 CUDA 上更快：融合、編譯、更少的記憶體流量。 |
| [**GPU 核心最佳化**](../../examples/gpu-kernels/README.md) | 以 flash attention 或 Triton 核心在 CUDA 上加速多頭自注意力，並保持在數值容差內。 |

## 📚 文件與支援

- **完整文件**：[docs.leeroo.com](https://docs.leeroo.com/docs)
  - [安裝](https://docs.leeroo.com/docs/installation)：程式編寫代理的 CLI、套件，以及 `kapso doctor`
  - [快速入門](https://docs.leeroo.com/docs/first-build)：你的第一場戰役
  - [CLI 參考](https://docs.leeroo.com/docs/reference/cli)：每個指令、旗標與預設值
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api)：`Kapso` 上的每個公開方法
  - [設定](https://docs.leeroo.com/docs/reference/configuration)：`config.yaml` 中的每個鍵
  - [learn 產出什麼](https://docs.leeroo.com/docs/learn/what-learn-produces)：教訓庫，每條教訓都帶有可信度分數、邊界及其證據，以及[開發模式](https://docs.leeroo.com/docs/learn/development-mode)
  - [智慧實驗](https://docs.leeroo.com/docs/evolve/overview)：實驗樹及其模組，以及[單一實驗內部發生的事](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience)：你的教訓庫、Leeroopedia、儲存庫與論文化為可搜尋的知識
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets)：本機、Docker、Modal 與其他目標
  - [Kapso 外掛](https://docs.leeroo.com/docs/coding-agent-skills)：適用於 Claude Code、Codex 與 Cursor 的指令、技能與 MCP 伺服器
- **從你的程式編寫代理使用**：文件站在 `https://docs.leeroo.com/mcp` 提供 MCP 伺服器，且每個頁面都以 Markdown 形式提供，參見 [llms.txt](https://docs.leeroo.com/llms.txt) 與[在你的程式編寫代理中使用 Kapso](https://docs.leeroo.com/docs/coding-agent-skills)。
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **讓你的程式編寫代理執行 Kapso**：[`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 下的外掛包含十個指令、六項技能與 MCP 伺服器，每個主機一行安裝指令（見上方步驟 0）；OpenCode 以資料夾形式載入技能（`ln -s <clone>/plugin/skills .opencode/skills`）。參見 [Kapso 外掛](https://docs.leeroo.com/docs/coding-agent-skills)。
- **讓你的程式編寫代理取得 Kapso 的知識工具**：隨套件安裝的 `kapso-mcp`，會將網路研究、儲存庫的記憶、戰役的實驗歷史以及教訓庫提供給你正在工作的工作階段，參見 [MCP 閘道](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)。
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI；接著在 ~/.codex/config.toml 的 [mcp_servers.kapso] 下加上 `startup_timeout_sec = 60`（Codex 的十秒在忙碌的機器上太短）
  ```
- **社群**：[Discord](https://discord.gg/hqVbPNNEZM)
- **官方網站**：[leeroo.com](https://leeroo.com)


## 企業版 Kapso

在企業中，Kapso 代理隨附初始的資料與 AI 工程 harness 以及一個教訓庫，能自行將業務目標轉化為可運作的 AI 與資料系統，調查公司的工件、儲存庫、管線、資料來源等，並在過程中逐步建立企業的資料資產。若要為你的企業任務導入 Kapso，請[與我們聯繫](https://leeroo.com/contact-us)。

## 貢獻

我們歡迎貢獻！請參閱我們的[貢獻指南](../../CONTRIBUTING.md)，了解如何開始。

## 貢獻者

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

以 [contrib.rocks](https://contrib.rocks) 製作。
