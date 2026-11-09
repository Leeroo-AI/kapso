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
  <a href="https://leeroo.com">網站</a>
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

> **如果你是正在查看此儲存庫的 AI 代理：**Kapso 外掛程式包含用於設定、
> 陳述目標、撰寫 harness 與審閱執行的技能，只需一行即可安裝：
> `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> （Codex：`codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`）。

---

## 最新消息

- **IOAI 2026 · AI 模型賽道。**[國際人工智慧奧林匹亞](https://ioai-official.org)是 AI 時代的 IMO：來自 108 個國家與地區的 471 位參賽者，在單一 GPU 預算下完成六項由專家設計的任務。2026 年它開設了 [IOAI²](https://ioai-official.org/ai-model-track/)，讓 AI 系統在兩場無人看管的 6 小時賽程中參加*同一份*考試。計時一旦開始，任何人都不得解題、修正或改進任何內容。Kapso 以 14 個創始 AI 參與者之一的身分參賽。
  - 🎓 **超越全部 471 位人類參賽者**：總分 536.07，高於會場中的每一位參賽者。
  - 🏆 **IOAI² 大師獎盃**：在所有參賽的 AI 系統中名列前 3。[完整結果 →](../../benchmarks/ioai2026/README.md) · [IOAI 對該賽道的回顧 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **在 [RelBench](../../benchmarks/relbench/README.md) 全部三個排行榜上名列第一**：在 Stanford 針對關聯式資料預測 ML 的基準測試上，Kapso 在官方排行榜的分類（12 項任務的 AUROC 為 81.2，對比 [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) 的 79.3 與 Stanford STAR 的 [RT-J](https://star-project.stanford.edu/rt-j) 的 79.0）、迴歸（9 項任務的 NMAE 為 0.2476，對比 RT-J 的 0.2711 與 KumoRFM-v2 的 0.2847，誤差少 8.7%）與推薦（10 項任務的 MAP 為 18.4，對比其他最佳參賽項目 ID-GNN 的 9.3）三項中皆領先。每個數字都是[官方 RelBench 排行榜](https://star-project.stanford.edu/relbench/leaderboard/)上的一筆紀錄。保存在 harness 中的學習勝過烘焙進模型權重中的學習，而更新教訓庫的成本遠低於重新訓練表格式基礎模型。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)**：一部收錄我們在資料與 ML 領域教訓的維基，約 100k 個頁面是透過對論文與儲存庫進行實驗而擷取出來的，每月提供約 6M 次請求，其中約 28% 來自 AI 代理。Kapso 透過 MCP 連接到它：其工作階段在構思與實作期間用它來搜尋知識、建立計畫與診斷失敗。
- **[MLE-Bench](../../benchmarks/mle/README.md) 第 1 名**：KAPSO 在 Kaggle ML 競賽（MLE Benchmark）中於開源系統裡取得最高排名。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **[ALE-Bench](../../benchmarks/ale/README.md) 第 1 名**：KAPSO 在長程演算法探索問題（ALE Benchmark）上取得最高排名。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## Kapso 是什麼？

在公司內部交付 AI 或資料成果，意味著要透過實驗學會兩件事。沒有任何個人或 AI 代理知道一個資料資產是如何連接與運作的；學會它的唯一方法是調查公司的產出物（儲存庫、管線、文件、資料來源）、提出假設、執行實驗並從中學習，直到理解每個系統的行為及其條件為止。而即使有了這種理解，最佳解決方案仍然無法事先挑選，無論工作是最佳化資料倉儲還是預測客戶流失，所以你得不斷實驗與學習，直到找到適合該環境的方案。適合該環境的訣竅會逐漸累積，就如同 RL 一般。RAG 則恰好相反：它餵給代理的是從未在此環境中測試過的通用知識，全部帶有相同的權重，而這些知識在該環境中被嘗試之前幾乎沒有價值。

Kapso 就是我們為此打造的引擎。它從一個領域 harness 與一個空的教訓庫出發；給定目標與成功標準後，它會執行一棵實驗樹，提出假設，然後逐一實作並驗證，並將最佳解決方案部署到目標技術堆疊上。接著它從這次執行中學習，依據每個假設與實作對結果的影響為其分配功勞，並更新教訓庫，其中每條教訓都帶有可信度分數以及說明其適用範圍的邊界。隨著時間推移，領域 harness 會承載一個累積了該環境中經過實驗驗證之最佳實務的教訓庫。這套系統不綁定特定模型，也不綁定特定編碼代理：它是開源的，由 [Leeroo](https://leeroo.com) 開發，並以 [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) 之名發佈於 PyPI。

### 四大組成部分

| 組成部分 | 動詞 | 功能 |
|------|------|--------------|
| [**智慧實驗**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 各個專門模組共用同一個資料結構，也就是實驗樹：每個節點記錄假設、實作（git 分支與技術難點）、已使用的預算，以及與目標的距離。對於每個實驗，構思模組會依據先前的實驗與教訓庫提出想法，按預期回報相對於各自所需預算重新排序，並將排名最高的想法送交實作器，實作器則在從父節點分叉出來的分支上進行建置。由於教訓庫持續學習在那個確切環境中什麼有效，這些想法會變得專屬於該環境，而不是單純的 LLM 或 RAG 所能提供的通用想法。結果由任何實驗都無法存取、帶有指紋的評估器進行驗證，而評估本身也會演進：實驗模組可以提出對指標與標準的更新，交由驗證器核准。[運作方式 →](https://docs.leeroo.com/docs/evolve/overview) |
| [**學習**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 教訓庫中的一切都來自實驗：每條教訓都是從實驗執行的追蹤記錄中歸納或演繹而來，帶有可信度分數、說明其適用範圍的邊界條件及其背後的證據，而在每次執行之後，`learn()` 會根據該教訓對最終結果的影響更新這些內容。對教訓庫的變更本身也是實驗：每條更新過的教訓都會透過在有它與沒有它的情況下執行同一個 build 進行 A/B 測試，因此改進是量測出來的，而不是猜測的。教訓庫建立在 git 之上，所以你可以與他人分享你的經驗（`kapso experience share`）、列出你可用的庫（`kapso experience list`）並更新它們（`kapso experience update`）。[經驗 →](https://docs.leeroo.com/docs/experience) 指的是提供給一次 build 的內容：你的庫、你的團隊共享的庫、Leeroopedia，或是從儲存庫或研究過的問題中匯入的語料庫。 |
| [**部署**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 執行一次同類型但較短的實驗，讓最佳解決方案適應目標技術堆疊：一個工作階段會調整該節點的副本、部署它，並展示健康檢查與一次成功的呼叫。內建支援 Docker、Modal、BentoCloud 與 LangGraph Platform，你也可以將自己的目標技術堆疊加入 harness。 |
| [**領域適應**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harness 是 Kapso 的領域套件：針對某個領域（例如預測建模）的知識與工作流程，讓 build 不必從零開始實驗與學習；Kapso 內建預測建模與競技程式設計兩種 harness。一次執行從一個領域 harness 與一個空的教訓庫開始，隨著時間推移，harness 會承載那個累積了該環境中有效做法的教訓庫。使用外掛程式中的 `/kapso:harness` 或 CLI 中的 `kapso harness`，撰寫你最初的領域 harness，包括你的指示、程式碼、防護欄與檢查。 |

## 🚀 快速入門

### 安裝

**0. 正在使用編碼代理？** 安裝 Kapso 外掛程式，讓它帶著你
走完以下步驟：

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

接著，`/kapso:setup` 會安裝並驗證，而 `/kapso:build <what you want>`
會啟動第一個活動（在 Codex 上：`$kapso-setup`、`$kapso-build <what you want>`）。請參閱 [Kapso 外掛程式](https://docs.leeroo.com/docs/coding-agent-skills)。

**1. 先決條件。** Kapso 透過編碼代理 CLI 執行推論
（沒有直接 API 的備援），所以在一切能運作之前，你需要 Node.js
以及兩個已登入的代理 CLI：

```bash
# Node.js 18+ (https://nodejs.org)，接著：
npm install -g @openai/codex            # 研究、評判、公用程式
codex login

npm install -g @anthropic-ai/claude-code  # 構思 + 實作（預設主機）
claude auth login
```

OpenAI 金鑰是選用的：只有知識圖譜與針對過去實驗的語意搜尋
（設定中的 `models.embedding`）會嵌入文字，生成過程從不使用它。
當你開啟其中任一項時：

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. 連接 Leeroopedia（建議）。** 一把金鑰就能讓每個活動
工作階段使用 [Leeroopedia](https://leeroopedia.com)，這是一個託管的
ML 與 AI 框架知識庫：框架實際上如何運作、計畫審查、
有文件記載的超參數範圍、失敗診斷。無需安裝任何東西；
在 [Leeroopedia 儀表板](https://app.leeroopedia.com/dashboard)註冊的新帳號
附帶 $20 的免費額度。

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. 安裝套件**（Python 3.10+）：

```bash
pip install leeroo-kapso
```

套件名稱是 `leeroo-kapso`。PyPI 上名為 `kapso` 的套件是一個不相關的
WhatsApp 工具，它會安裝自己的 `kapso` 命令並遮蔽本套件的命令；
如果它已存在，請先執行 `pip uninstall kapso`。

**4. 驗證設定：**

```bash
kapso doctor
```

`doctor` 會回報**你的設定**實際需要什麼，並對任何缺少的項目
指出確切的修復方式。需求隨設定而定，所以全 codex 的
設定永遠不會被要求安裝 `claude`。縮小到單一動詞即可只查看
該動詞的需求：

```bash
kapso doctor build             # 或 learn / deploy / experience
```

這些正是該動詞在開始任何工作之前自己會執行的檢查：
缺少 CLI 時，`kapso.build(...)` 會在幾秒內失敗，而不是在工作階段
深處才失敗。標記為 `[-- ]` 的項目是選用的；它們只會限制
你可能不需要的功能（docker、Weaviate、Neo4j、部署目標）。

**或者讓你的編碼代理完成這一切：**將
[設定提示詞](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
貼入 Claude Code、Codex 或 OpenCode。

**索引所需的資料儲存區（選用）**：`kapso experience add --repo`
與 `--research` 會將語料庫匯入本機的 Weaviate + Neo4j。從
原始碼檢出目錄中執行：

```bash
bash scripts/start_infra.sh   # 透過 docker 啟動兩者
```

**從原始碼安裝（用於開發）**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

舊版的 aider 轉接器是額外套件（`pip install "leeroo-kapso[aider]"`，
Python <3.13）；預設的 claude/codex 代理不需要任何額外套件。

### 基本用法

核心動詞除了上述先決條件之外不需要其他任何東西：

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 開始時不需要知識圖譜

# Build：透過實驗得出解決方案。活動在啟動時會印出
# `status: <path>`，可從另一個終端機即時觀看它：
#     kapso watch ./campaign
# 如果某個工作階段需要只有你能提供的東西（API 金鑰、檔案），
# 活動會暫停，而 `kapso inbox reply` 會讓它繼續：
#     https://docs.leeroo.com/docs/build/inbox
# 以 Markdown 分享結果，用於 pull request 或 issue：
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 或省略以從零開始
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 從你剛執行的活動中學習：挖掘軌跡、為教訓評分，
# 並將以證據定價的知識卡片存入庫中。該庫（一個本機
# git 儲存庫）會在首次使用時自動建立；教訓會留在你的
# 機器上，直到你分享它們為止：
#     kapso experience share <git-url>
# 之後每次 learn() 都會將庫推送到那裡。
lesson = kapso.learn(solution)
print(lesson.explain())

# 再次 build，這次提供它剛獲得的卡片：一次 build 所獲得的
# 正是它所載入的經驗。
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

在資料儲存區執行的情況下，你也可以將外部知識匯入
索引，並以同樣的方式提供：

```python
from kapso import Kapso

kapso = Kapso()

# 一次性操作：將一個儲存庫，或針對某個問題研究過的網路內容，匯入索引
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# 每次載入該索引的 build 都會搜尋它；Leeroopedia 則隨其金鑰一同啟用
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

關於預算的說明：匯入會擷取素材所能支撐的一切。
單一問題可能變成數十個相互連結的頁面以及長達數小時的
匯入；匯入時間隨可擷取的內容量而增長，所以請加入
你將會搜尋的來源，而不是你擁有的每一個來源。

而要將解決方案變成執行中的軟體：

```python
software = kapso.deploy(solution, target="local")   # 或 docker、modal、bentoml、langgraph、auto
result = software.run(software.proof["inputs"])     # 部署所證明可行的那次呼叫
software.stop()
```

### 如何限制活動的範圍？

使用 `max_iterations`、`time_budget_minutes` 與 `cost_budget` 來限制迭代次數、經過時間，以及以美元計的盡力估算模型花費。
對於平台成本，請在 `search_strategy.params.cost` 中啟用[成本感知選擇](https://docs.leeroo.com/docs/evolve/cost-aware-selection)，並提供你的估算器、`cap_per_candidate` 與 `campaign_budget`；其 `weight` 也能讓選擇偏好較便宜的候選方案。

### 選擇模型

Kapso 使用的每個模型都在同一個設定檔中指定。套件內建的預設值
以 `claude-opus-5` 執行 build 工作階段、以 `claude-fable-5` 執行
學習團隊、以 `gpt-5.6-sol` 執行 codex 角色，但模型的存取
取決於訂閱方案（某個方案可能對一個模型設限，同時提供另一個模型）。
要以不同的模型執行，請複製內建設定、編輯，然後讓
Kapso 指向你的設定：

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 例如，以 opus 而非 fable 執行學習團隊：
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

純文字推論角色可以選擇使用 `openai_compatible` 的 Chat Completions。
在設定中指定模型與端點；憑證則保留在環境變數中。
以 `Kapso(config=...)` 或 `kapso build --config ...` 載入設定。
程式編寫與網路研究仍需要具備對應工具的代理。
請參閱[設定範例與限制](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)。

在長時間執行之前，請針對你實際的訂閱方案探測設定中指定的
每個模型：你的登入無法提供的模型會在這裡於幾秒內失敗，
而不是在執行數小時後才失敗。對於你能使用的模型，其用量上限
無法透過單一 token 的探測看出：

```bash
kapso doctor --models                            # 內建設定
kapso doctor --models --config kapso-config.yaml # 你的設定
kapso doctor learn --models                      # 只探測學習團隊
```

要讓這項探測成為每次呼叫的一部分，而不是需要你記得執行的
事情，請在設定中開啟它。每個不同的 `{cli, model}` 組合只花費
一個可拋棄的 token，在無人看管、長達數小時的 `learn()` 之前
這是值得的：

```yaml
preflight:
  enabled: true            # 每個動詞之前的靜態檢查（預設）
  live_model_probe: true   # + 單一 token 的模型探測（預設：false）
```

更換模型也會改變*節奏*：設定中團隊的 `timeout_minutes` 上限
是根據預設模型校準的，若更換的模型推理時間較長，
可能需要調高這些上限。

如需詳細的整合步驟，請參閱[快速入門](https://docs.leeroo.com/docs/first-build)與[安裝](https://docs.leeroo.com/docs/installation)指南。

## 支援的基準測試

| 基準測試 | 說明 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI 的 ML 工程基準測試：涵蓋表格、視覺、文字與音訊的完整競賽，從原始資料到評分提交 |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI 的演算法最佳化基準測試：在長達數小時的搜尋中設計、實作並迭代競賽啟發式演算法 |
| [**RelBench**](../../benchmarks/relbench/README.md) | Stanford 針對關聯式資料預測 ML 的基準測試：直接從 Amazon、H&M、Stack Exchange、臨床試驗等多表格資料庫進行分類、迴歸與推薦 |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 限時的奧林匹亞 ML，涵蓋視覺、語言與最佳化：專家出題、競賽硬體、零人工協助 |

每個基準測試也都有一個文件頁面，說明如何執行、其 CLI 選項以及輸出包含的內容：
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 範例

每個範例都是一項任務：一份任務簡述、一個起始儲存庫、資料或取得資料的方法，以及最多十二行的驅動程式；[examples/README.md](../../examples/README.md) 列出所需分鐘數、那一行命令，以及是否有記錄結果。

| 範例 | 展示內容 |
|---------|---------------|
| [**客戶流失首次 build**](../../examples/churn-first-build/README.md) | 你的第一次 build：已有評估器、基準線、目標值與記錄 |
| [**詐欺偵測無評估器**](../../examples/fraud-no-evaluator/README.md) | 沒有評估器：活動的第一個節點依據任務簡述中的協定建立一個評估器，每個候選方案都以它衡量 |
| [**dbt 管線**](../../examples/pipeline-dbt/README.md) | 在 DuckDB 上、匯出結果帶有缺陷的 dbt 管線：讓資料契約得以成立 |
| [**交易**](../../examples/trading/README.md) | build 上的 harness：一個設定工作階段、每個工作階段中的頁面與掛鉤、每次嘗試的驗證 |
| [**端到端**](../../examples/end-to-end/README.md) | Build、審閱、learn、帶著教訓再次 build、deploy：端到端的 Kaggle 表格式模型 |
| [**提示詞 harness**](../../examples/prompt-harness/README.md) | 以提示詞作為受測對象，在 AIME 題目上計分 |
| [**代理鷹架**](../../examples/agent-scaffold/README.md) | 視覺語言擷取器，對照真實標註表格計分 |
| [**torch 效能**](../../examples/torch-performance/README.md) | 在數值容差內更快的 CUDA 模組 |
| [**GPU 核心**](../../examples/gpu-kernels/README.md) | CUDA 上的多頭注意力：flash attention 或 Triton |

## 📚 文件與支援

- **完整文件**：[docs.leeroo.com](https://docs.leeroo.com/docs)
  - [安裝](https://docs.leeroo.com/docs/installation)：編碼代理 CLI、套件，以及 `kapso doctor`
  - [快速入門](https://docs.leeroo.com/docs/first-build)：你的第一個活動
  - [CLI 參考](https://docs.leeroo.com/docs/reference/cli)：每個命令、旗標與預設值
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api)：`Kapso` 上的每個公開方法
  - [設定](https://docs.leeroo.com/docs/reference/configuration)：`config.yaml` 中的每個鍵
  - [learn 產出什麼](https://docs.leeroo.com/docs/learn/what-learn-produces)：教訓庫，每條教訓都帶有可信度分數、邊界及其證據，以及[開發模式](https://docs.leeroo.com/docs/learn/development-mode)
  - [智慧實驗](https://docs.leeroo.com/docs/evolve/overview)：實驗樹及其模組，以及[單一實驗內部發生的事](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [經驗](https://docs.leeroo.com/docs/experience)：你的庫、Leeroopedia，以及轉化為可搜尋知識的儲存庫與論文
  - [部署](https://docs.leeroo.com/docs/deploy/targets)：本機、Docker、Modal 與其他目標
  - [Kapso 外掛程式](https://docs.leeroo.com/docs/coding-agent-skills)：適用於 Claude Code、Codex 與 Cursor 的命令、技能與 MCP 伺服器
- **從你的編碼代理**：文件站在 `https://docs.leeroo.com/mcp` 執行一個 MCP 伺服器，且每個頁面都以 Markdown 提供，請參閱 [llms.txt](https://docs.leeroo.com/llms.txt) 與[在你的編碼代理中使用 Kapso](https://docs.leeroo.com/docs/coding-agent-skills)。
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **讓你的編碼代理執行 Kapso**：[`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 下的外掛程式包含十個命令、六項技能與 MCP 伺服器，每個主機只需一行安裝命令（上方步驟 0）；OpenCode 以資料夾的形式載入技能（`ln -s <clone>/plugin/skills .opencode/skills`）。請參閱 [Kapso 外掛程式](https://docs.leeroo.com/docs/coding-agent-skills)。
- **為你的編碼代理提供 Kapso 的知識工具**：隨套件安裝的 `kapso-mcp` 會向你工作的工作階段提供網路研究、儲存庫的記憶、活動的實驗歷史與教訓庫，請參閱 [MCP 閘道](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)。
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI；接著在 ~/.codex/config.toml 的 [mcp_servers.kapso] 下加入 `startup_timeout_sec = 60`（Codex 的十秒在忙碌的機器上太短了）
  ```
- **社群**：[Discord](https://discord.gg/hqVbPNNEZM)
- **網站**：[leeroo.com](https://leeroo.com)


## 企業版 Kapso

在企業中，Kapso 代理隨附初始的資料與 AI 工程 harness 以及教訓庫，能自行將業務目標轉化為可運作的 AI 與資料系統，一路調查公司的產出物、儲存庫、管線、資料來源等，並在過程中逐步建立企業的資料資產。要為你的企業任務導入 Kapso，請[與我們聯繫](https://leeroo.com/contact-us)。

## 貢獻

我們歡迎各種貢獻！請參閱我們的[貢獻指南](../../CONTRIBUTING.md)，了解如何開始。

## 貢獻者

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

使用 [contrib.rocks](https://contrib.rocks) 製作。
