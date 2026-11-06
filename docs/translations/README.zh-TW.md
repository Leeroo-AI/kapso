<!-- Traditional Chinese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">一個執行實驗、部署最佳解決方案，並從每次執行中學習的引擎</h4>
<p align="center"><sub>智慧實驗 · 學習迴圈 · 部署 · 領域適應</sub></p>

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
  <a href="https://leeroo.com">網站</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  如果你喜歡這個專案，請給它一顆星來支持我們 ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **如果你是正在查看此儲存庫的 AI 代理：** Kapso 外掛內含設定、陳述目標、撰寫 harness 與檢視執行結果的技能，只需一行即可安裝：
> `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> （Codex：`codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`）。

---

## 最新消息

- **IOAI 2026 · AI 模型賽道。** [國際人工智慧奧林匹亞競賽](https://ioai-official.org)是 AI 時代的 IMO：471 名參賽者來自 108 個國家與地區，在單一 GPU 的預算下挑戰六道由專家設計的題目。2026 年，它開設了 [IOAI²](https://ioai-official.org/ai-model-track/)，讓 AI 系統在兩場各 6 小時、無人值守的賽程中應考*同一份*試題。計時一旦開始，任何人都不得代為解題、修正或改進任何內容。Kapso 以 14 個創始 AI 參與者之一的身分參賽。
  - 🎓 **得分超越全部 471 名人類參賽者**：總分 536.07，高於賽場上的每一位參賽者。
  - 🏆 **IOAI² Grand Master 獎盃**：在所有參賽的 AI 系統中名列前 3 名。[完整結果 →](../../benchmarks/ioai2026/README.md) · [IOAI 對此賽道的回顧 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **在 [RelBench](../../benchmarks/relbench/README.md) 全部三個排行榜上皆名列第一**：在 Stanford 針對關聯式資料預測型 ML 的基準測試上，Kapso 在官方排行榜的分類（81.2 AUROC，涵蓋 12 項任務，對比 [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) 的 79.3 與 Stanford STAR 的 RT-J 的 79.0）、迴歸（0.2476 NMAE，涵蓋 9 項任務，對比 RT-J 的 0.2711 與 KumoRFM-v2 的 0.2847，誤差減少 8.7%）與推薦（18.4 MAP，涵蓋 10 項任務，對比其他參賽項目中表現最佳的 ID-GNN 的 9.3）上均居領先。每個數字都是[官方 RelBench 排行榜](https://star-project.stanford.edu/relbench/leaderboard/)上的一筆登錄成績。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)**：一部彙整我們在資料與 ML 領域所得教訓的 wiki，約 100k 頁，內容萃取自對論文與儲存庫所做的實驗，每月處理約 6M 次請求，其中約 28% 來自 AI 代理。Kapso 透過 MCP 與它連線：其工作階段在構思與實作期間用它來搜尋知識、擬定計畫並診斷失敗。
- **[Moltbook Agents 🦞](https://www.moltbook.com/)**：打造能最佳化其他代理並在 Moltbook 上辯論的 AI 代理！[開始使用 →](../../moltbook_bot/README.md)
- **技術報告**：我們的技術報告現已發布！[閱讀論文](https://arxiv.org/abs/2601.21526)
- **[MLE-Bench](../../benchmarks/mle/README.md) 排名 #1**：KAPSO 在 Kaggle ML 競賽（MLE Benchmark）的開源系統中取得最高排名。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **[ALE-Bench](../../benchmarks/ale/README.md) 排名 #1**：KAPSO 在長時程演算法探索問題（ALE Benchmark）上取得最高排名。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## Kapso 是什麼？

Kapso 是一個執行實驗、部署最佳解決方案，並從每次執行中學習的引擎。給定目標、初始的成功標準與評估指標後，它會進行調查、挑選最好的想法來執行實驗、有時提供回饋以改進評估指標與成功標準、將最佳解決方案部署到所需的技術堆疊上，並從每次執行中學習。它維護一個教訓庫，其中每則教訓都帶有可信度分數，以及界定其適用範圍的邊界，因此知識會隨時間產生複利效應，而不只是不斷累積。

我們曾在包括 LinkedIn 與 Meta AI 在內的大型公司內部打造 AI 系統，並將它們部署到擁有數百萬使用者的產品中。Kapso 就是把那套工作流程做成引擎：開源，由 [Leeroo](https://leeroo.com) 開發，並以 [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) 之名發布於 PyPI。

### 四個部分

| 部分 | 動詞 | 功能 |
|------|------|--------------|
| [**智慧實驗**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 實驗迴圈是一組共用同一個資料結構（即實驗樹）的專門模組：每個節點記錄假設、實作（git 分支與技術困難）、已使用的預算，以及與目標的距離。構思階段會參考先前的實驗與教訓庫來提出想法，每個想法都會說明它與特定的先前實驗有何不同、採用了哪些教訓；重排序器會依據相對於目標值的預期報酬，並權衡時間與預算，為這些想法排序。實作器在從父節點分叉出來的分支上進行建構，驗證器則依安全規則與成功標準檢查該節點，判定分數是否有效、目標是否達成。成功標準與測試可以在執行過程中演進：各模組提出有理據的更新，由驗證器核准或駁回。[運作方式 →](https://docs.leeroo.com/docs/evolve/overview) |
| [**學習迴圈**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 每則教訓記錄一項知識、一個可信度分數、界定其適用範圍的邊界條件，以及其歷次更新背後的證據。每次執行之後，`learn()` 會根據所用教訓對結果的影響來更新它們的可信度與邊界，從相似的追蹤紀錄中歸納出教訓，或從單一證據及其機制演繹出一則教訓，合併重疊的教訓，並以帶有與不帶有這些教訓的方式執行同一個 build，對更新後的教訓進行 A/B 測試，因此更新是經過測量而非憑空猜測。下一次 build 的構思與實作會讀取每則教訓的可信度與邊界，以決定何時該依賴它，而不是單純對知識做 RAG。[Experience →](https://docs.leeroo.com/docs/experience) 指明提供給 build 的內容：你的教訓庫、你團隊共用的教訓庫、Leeroopedia，或是從儲存庫或某個研究問題匯入的語料庫。 |
| [**部署**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 透過較短版本的實驗迴圈，將樹中的最佳解決方案轉化為在目標技術堆疊上執行的軟體：一個工作階段會調整該節點的副本、加以部署，並展示健康檢查與一次可運作的呼叫。內建支援 Docker、Modal、BentoCloud 與 LangGraph Platform，你也可以把自己的目標技術堆疊加入 harness。 |
| [**領域適應**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harness 是 Kapso 的領域套件：針對某個領域（例如預測建模）的知識與工作流程，讓 build 不必從零開始實驗與學習；預測建模與競技程式設計的 harness 隨 Kapso 一併提供。若要帶入你自己的領域知識，可透過 Kapso 定義的介面撰寫 harness：你的指示、程式碼、護欄與檢查，可用外掛中的 `/kapso:harness` 或 CLI 中的 `kapso harness` 撰寫。 |

## 🚀 快速入門

### 安裝

**0. 正在使用程式設計代理？** 安裝 Kapso 外掛，讓它陪你完成以下步驟：

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

接著 `/kapso:setup` 會完成安裝與驗證，`/kapso:build <what you want>` 則啟動第一個活動（在 Codex 上：`$kapso-setup`、`$kapso-build <what you want>`）。請參閱 [Kapso 外掛](https://docs.leeroo.com/docs/coding-agent-skills)。

**1. 先決條件。** Kapso 透過程式設計代理的 CLI 執行推論（沒有直接呼叫 API 的備援方案），因此在一切開始運作之前，你需要 Node.js，並且兩個代理 CLI 都已登入：

```bash
# Node.js 18+ (https://nodejs.org)，接著：
npm install -g @openai/codex            # 研究、評審、工具程式
codex login

npm install -g @anthropic-ai/claude-code  # 構思 + 實作（預設宿主）
claude auth login
```

OpenAI 金鑰為選用：只有知識圖譜與針對過往實驗的語意搜尋（你設定中的 `models.embedding`）會對文字做嵌入，生成則完全不會用到它。當你啟用其中任一項時：

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. 連接 Leeroopedia（建議）。** 一把金鑰就能讓每個活動工作階段使用 [Leeroopedia](https://leeroopedia.com)，這是一個託管的 ML 與 AI 框架知識庫：框架實際的行為方式、計畫審查、有文件記載的超參數範圍、失敗診斷。無須安裝任何東西；在 [Leeroopedia 儀表板](https://app.leeroopedia.com/dashboard)註冊新帳號即可獲得 $20 的免費額度。

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. 安裝套件**（Python 3.10+）：

```bash
pip install leeroo-kapso
```

套件名稱是 `leeroo-kapso`。PyPI 上名為 `kapso` 的套件是一個不相干的 WhatsApp 工具，它會安裝自己的 `kapso` 指令並遮蔽本套件的指令；如果已安裝該套件，請先執行 `pip uninstall kapso`。

**4. 驗證設定：**

```bash
kapso doctor
```

`doctor` 會回報**你的設定**實際需要什麼，並針對任何缺少的項目指出確切的修正方式。需求隨設定而定，因此全部使用 codex 的設定絕不會被要求提供 `claude`。將範圍縮小到單一動詞，即可只看該動詞的需求：

```bash
kapso doctor build             # 或 learn / deploy / experience
```

這些正是該動詞本身在開始任何工作之前所執行的檢查：缺少 CLI 時，`kapso.build(...)` 會在幾秒內失敗，而不是等到深入某個工作階段後才失敗。標示為 `[-- ]` 的項目為選用；它們只會限制你可能用不到的功能（docker、Weaviate、Neo4j、各個部署目標）。

**或者讓你的程式設計代理完成這一切：** 將[設定提示詞](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)貼到 Claude Code、Codex 或 OpenCode 中。

**索引所需的資料儲存區（選用）**：`kapso experience add --repo` 與 `--research` 會將語料庫匯入本機的 Weaviate + Neo4j。從原始碼簽出的目錄執行：

```bash
bash scripts/start_infra.sh   # 透過 docker 啟動兩者
```

**從原始碼安裝（供開發使用）**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

舊版的 aider 轉接器是一個額外相依項目（`pip install "leeroo-kapso[aider]"`，Python <3.13）；預設的 claude/codex 代理不需要任何額外相依項目。

### 基本用法

核心迴圈除了上述先決條件之外，不需要其他任何東西：

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 一開始不需要知識圖譜

# Build：透過實驗得出解決方案。活動啟動時會印出
# `status: <path>`，可從另一個終端機即時觀看：
#     kapso watch ./campaign
# 如果某個工作階段需要只有你能提供的東西（API 金鑰、檔案），
# 活動會暫停，`kapso inbox reply` 可讓它繼續：
#     https://docs.leeroo.com/docs/build/inbox
# 將結果以 Markdown 形式分享到 pull request 或 issue：
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 或省略以從零開始
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 從你剛執行完的活動中學習：挖掘軌跡、為教訓評分，
# 並把依證據定價的知識卡片存入教訓庫。教訓庫（一個本機
# git repo）會在首次使用時自動建立；在你分享之前，教訓都會留在
# 你的機器上：
#     kapso experience share <git-url>
# 之後每次 learn() 都會把教訓庫推送到那裡。
lesson = kapso.learn(solution)
print(lesson.explain())

# 再次 build，並提供它剛獲得的卡片：build 得到的正是
# 它所載入的經驗。
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

在資料儲存區運作的情況下，你也可以將外部知識匯入索引，並以相同方式提供給 build：

```python
from kapso import Kapso

kapso = Kapso()

# 只需一次：將一個儲存庫，或針對某個問題所做的網路研究，匯入索引
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# 每個載入該索引的 build 都會搜尋它；Leeroopedia 則憑自己的金鑰運作
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

關於預算的提醒：匯入會萃取出素材所能支撐的一切內容。單單一個問題就可能變成數十個相互連結的頁面，以及長達數小時的匯入；匯入時間會隨可萃取的實質內容量而增加，因此請加入你會搜尋的來源，而不是你手上的每一個來源。

若要將解決方案轉化為可執行的軟體：

```python
software = kapso.deploy(solution, target="local")   # 或 docker、modal、bentoml、langgraph、auto
result = software.run(software.proof["inputs"])     # 部署已驗證過的那次呼叫
software.stop()
```

### 如何為活動設限？

使用 `max_iterations`、`time_budget_minutes` 與 `cost_budget` 來限制迭代次數、經過時間，以及以 USD 計的模型花費（盡力而為）。
若要控管平台成本，請在 `search_strategy.params.cost` 中啟用[成本感知選擇](https://docs.leeroo.com/docs/evolve/cost-aware-selection)，並設定你的估算器、`cap_per_candidate` 與 `campaign_budget`；其 `weight` 還能讓選擇過程偏好較便宜的候選方案。

### 選擇模型

Kapso 使用的每個模型都在同一個設定檔中指定。套件內建的預設值以 `claude-opus-5` 執行 build 工作階段、以 `claude-fable-5` 執行學習小組、以 `gpt-5.6-sol` 執行 codex 角色，但模型的存取權取決於訂閱方案（某個方案可能限制某個模型的用量，同時仍提供另一個模型）。若要改用不同的模型，請複製套件內建的設定、加以編輯，再讓 Kapso 指向你的設定：

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 例如：讓學習小組改用 opus 而非 fable 執行：
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

純文字推論角色可以選擇改用 `openai_compatible` Chat Completions。
在你的設定中指定模型與端點；憑證則留在環境變數中。
以 `Kapso(config=...)` 或 `kapso build --config ...` 載入設定。
程式設計與網路研究仍需要具備相應工具的代理。
請參閱[設定範例與限制](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)。

在長時間執行之前，請依你實際的訂閱方案探測設定中指定的每個模型：你的登入帳號無法使用的模型會在這裡於幾秒內失敗，而不是在執行了數小時之後才失敗。不過，對於你可以使用的模型，其用量上限是單一 token 的探測看不出來的：

```bash
kapso doctor --models                            # 套件內建的設定
kapso doctor --models --config kapso-config.yaml # 你自己的設定
kapso doctor learn --models                      # 僅學習小組
```

若要讓這項探測成為每次呼叫的一部分，而不是一件得靠你記得去執行的事，請在設定中將它開啟。每個不同的 `{cli, model}` 組合會花費一個用完即丟的 token，在無人值守、長達數小時的 `learn()` 之前，這是值得的：

```yaml
preflight:
  enabled: true            # 每個動詞執行前的靜態檢查（預設）
  live_model_probe: true   # + 單一 token 的模型探測（預設：false）
```

更換模型也會改變*步調*：設定中各小組的 `timeout_minutes` 上限是依預設模型校準的，換上推理時間較長的模型後，可能需要調高這些上限。

如需詳細的整合步驟，請參閱[快速入門](https://docs.leeroo.com/docs/first-build)與[安裝](https://docs.leeroo.com/docs/installation)指南。

## 範例

每個範例都是一項任務：一份任務簡述、一個起始儲存庫、資料或其取得方式，以及一支至多十二行的驅動程式；[examples/README.md](../../examples/README.md) 列出所需分鐘數、那一行指令，以及是否有記錄結果。

| 範例 | 展示內容 |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | 你的第一次 build：已有評估器、一個基準線、一個目標值、執行紀錄 |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | 沒有評估器：活動的第一個節點會依任務簡述中的協定建立一個，之後每個候選方案都以它來衡量 |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | 一條建構於 DuckDB 上、匯出資料帶有缺陷的 dbt 管線：讓資料契約得以成立 |
| [**trading**](../../examples/trading/README.md) | 在 build 上使用 harness：一個設定工作階段、每個工作階段中的頁面與掛鉤、每次嘗試的驗證 |
| [**full loop**](../../examples/full-loop/README.md) | Build、檢視、learn、帶著所得教訓再次 build、deploy：一個 Kaggle 表格資料模型的端到端流程 |
| [**prompt harness**](../../examples/prompt-harness/README.md) | 以提示詞作為受測對象，以 AIME 題目評分 |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | 一個視覺語言擷取器，對照真實標註表格評分 |
| [**torch performance**](../../examples/torch-performance/README.md) | 在數值容差範圍內，於 CUDA 上執行得更快的模組 |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | CUDA 上的多頭注意力：flash attention 或 Triton |

## 支援的基準測試

| 基準測試 | 說明 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI 的 ML 工程基準測試：涵蓋表格、視覺、文字與音訊的完整競賽，從原始資料到評分後的提交 |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI 的演算法最佳化基準測試：在長達數小時的搜尋中設計、實作並反覆改進競賽啟發式演算法 |
| [**RelBench**](../../benchmarks/relbench/README.md) | Stanford 針對關聯式資料預測型 ML 的基準測試：直接從 Amazon、H&M、Stack Exchange、臨床試驗等的多資料表資料庫進行分類、迴歸與推薦 |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 涵蓋視覺、語言與最佳化的限時奧林匹亞 ML 競賽：專家出題、競賽指定硬體、零人為協助 |

每個基準測試也都有一個文件頁面，說明如何執行、其 CLI 選項，以及輸出包含哪些內容：
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 文件與支援

- **完整文件**：[docs.leeroo.com](https://docs.leeroo.com/docs)
  - [安裝](https://docs.leeroo.com/docs/installation)：程式設計代理的 CLI、套件，以及 `kapso doctor`
  - [快速入門](https://docs.leeroo.com/docs/first-build)：你的第一個活動
  - [CLI 參考](https://docs.leeroo.com/docs/reference/cli)：每個指令、旗標與預設值
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api)：`Kapso` 上的每個公開方法
  - [設定](https://docs.leeroo.com/docs/reference/configuration)：`config.yaml` 中的每個鍵
  - [learn 產出什麼](https://docs.leeroo.com/docs/learn/what-learn-produces)：教訓庫，其中每則教訓都帶有可信度分數、邊界與其證據，以及[開發模式](https://docs.leeroo.com/docs/learn/development-mode)
  - [實驗迴圈](https://docs.leeroo.com/docs/evolve/overview)：實驗樹及其模組，以及[單一實驗內部發生了什麼](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience)：你的教訓庫、Leeroopedia、轉化為可搜尋知識的儲存庫與論文
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets)：本機、Docker、Modal 與其他部署目標
  - [Kapso 外掛](https://docs.leeroo.com/docs/coding-agent-skills)：適用於 Claude Code、Codex 與 Cursor 的指令、技能與 MCP 伺服器
- **從你的程式設計代理存取**：文件站在 `https://docs.leeroo.com/mcp` 提供 MCP 伺服器，且每個頁面都以 Markdown 形式提供；請參閱 [llms.txt](https://docs.leeroo.com/llms.txt) 與[在你的程式設計代理中使用 Kapso](https://docs.leeroo.com/docs/coding-agent-skills)。
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **讓你的程式設計代理執行 Kapso**：[`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 底下的外掛包含十個指令、六項技能與 MCP 伺服器，每個宿主只需一行安裝指令（見上方的步驟 0）；OpenCode 則以資料夾形式載入這些技能（`ln -s <clone>/plugin/skills .opencode/skills`）。請參閱 [Kapso 外掛](https://docs.leeroo.com/docs/coding-agent-skills)。
- **為你的程式設計代理提供 Kapso 的知識工具**：隨套件一併安裝的 `kapso-mcp`，會向你所在的工作階段提供網路研究、儲存庫的記憶、活動的實驗歷史與教訓庫；請參閱 [MCP 閘門](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)。
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI；接著在 ~/.codex/config.toml 的 [mcp_servers.kapso] 底下設定 `startup_timeout_sec = 60`（Codex 預設的十秒在忙碌的機器上太短）
  ```
- **社群**：[Discord](https://discord.gg/hqVbPNNEZM)
- **網站**：[leeroo.com](https://leeroo.com)


## Kapso 企業版

我們的企業產品建構於此引擎之上，並已部署在錯綜複雜的企業環境中，在那裡沒有任何人能夠調查並理解整個資料資產：我們曾協助一家公司從舊的資料平台遷移到新的資料湖倉，橫跨超過 30 個來源系統家族、數千張資料表與轉換，以及數百份 BI 報表。它處理的每一項任務都會充實教訓庫，累積關於你的系統、你的資料以及你得來不易的規則的知識，因此它運作得越久，就越了解你的公司。若想導入 Kapso 來處理你具挑戰性的企業任務，請[與我們聯絡](https://leeroo.com/contact-us)。

## 翻譯

README 提供其他語言版本：請使用頂端的語言切換器，或瀏覽 [`docs/translations/`](../../docs/translations/)。每份翻譯皆由 `scripts/translate_readme.py` 根據 README.md 產生，並由 `scripts/check_readme_translations.py` 保持同步；只要某份翻譯在連結、數字或程式碼行上與 README.md 不一致，該指令稿就會失敗。若要改進某份翻譯，請針對該資料夾中的檔案提出 pull request。

## 貢獻

我們歡迎各種貢獻！請參閱我們的[貢獻指南](../../CONTRIBUTING.md)，了解如何開始。

## 引用

如果你在研究或工作中使用了 Kapso，請引用這份技術報告：

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
