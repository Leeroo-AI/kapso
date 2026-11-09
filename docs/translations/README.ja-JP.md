<!-- Japanese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">実験を実行し、最良のソリューションをデプロイし、すべての実行から学習するエンジン</h4>
<p align="center"><sub>スマート実験 · 学習ループ · デプロイ · ドメイン適応</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <b>日本語</b> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">詳細はこちら</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Discord に参加</a> ·
  <a href="https://leeroo.com">ウェブサイト</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  このプロジェクトが気に入ったら、スター ⭐ を付けて応援してください
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **このリポジトリを見ている AI エージェントの方へ:** セットアップ、目的の記述、harness の作成、実行のレビューのためのスキルを備えた Kapso プラグインは、一行でインストールできます: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso` （Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`）。

---

## ニュース

- **IOAI 2026 · AI モデルトラック。** [国際人工知能オリンピック](https://ioai-official.org)は AI 時代の IMO です。471 人の出場者が 108 の国と地域から集まり、単一 GPU の予算のもとで、専門家が設計した六つの課題に挑みます。2026 年には [IOAI²](https://ioai-official.org/ai-model-track/) が始まり、AI システムが*同じ*試験を、無人の 6 時間セッション二回で受けます。時計が動き出したら、人間は何かを解くことも、修正することも、改善することも許されません。Kapso は 14 の Founding AI Participants の一つとして参加しました。
  - 🎓 **471 人の人間全員を上回る得点**: 合計 536.07 で、会場のすべての出場者を上回りました。
  - 🏆 **IOAI² Grand Master Trophy**: 参加したすべての AI システムの中で上位 3 位以内。[全結果 →](../../benchmarks/ioai2026/README.md) · [IOAI によるトラックの振り返り →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **三つすべての [RelBench](../../benchmarks/relbench/README.md) リーダーボードで首位**: リレーショナルデータ上の予測 ML を対象とする Stanford のベンチマークで、Kapso は公式リーダーボードのトップに立っています。分類では 81.2 AUROC（12 タスク。[KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) は 79.3、Stanford STAR の [RT-J](https://star-project.stanford.edu/rt-j) は 79.0）、回帰では 0.2476 NMAE（9 タスク。RT-J は 0.2711、KumoRFM-v2 は 0.2847 で、誤差は 8.7% 少ない）、レコメンデーションでは 18.4 MAP（10 タスク。他の最良エントリーである ID-GNN は 9.3）です。すべての数値は[公式 RelBench リーダーボード](https://star-project.stanford.edu/relbench/leaderboard/)に掲載されたエントリーです。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)**: データと ML のドメインにおける私たちの教訓をまとめた wiki です。論文やリポジトリを対象に実験して抽出した約 100k ページを収め、月に約 6M 件のリクエストに応えており、その約 28% は AI エージェントからのものです。Kapso は MCP 経由でこれに接続します。Kapso のセッションは、アイデア創出と実装の際にこれを使って知識を検索し、計画を立て、失敗を診断します。
- **[Moltbook Agents 🦞](https://www.moltbook.com/)**: 他のエージェントを最適化し、Moltbook で議論する AI エージェントを作りましょう！ [はじめる →](../../moltbook_bot/README.md)
- **テクニカルレポート**: テクニカルレポートを公開しました！ [論文を読む](https://arxiv.org/abs/2601.21526)
- **[MLE-Bench](../../benchmarks/mle/README.md) で 1 位**: KAPSO は、Kaggle の ML コンペティション（MLE Benchmark）において、オープンソースシステムの中でトップの順位を獲得しました。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **[ALE-Bench](../../benchmarks/ale/README.md) で 1 位**: KAPSO は、長期にわたるアルゴリズム発見問題（ALE Benchmark）でトップの順位を獲得しました。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## Kapso とは？

Kapso は、実験を実行し、最良のソリューションをデプロイし、すべての実行から学習するエンジンです。目的、初期の成功基準、評価指標が与えられると、調査を行い、最良のアイデアを選んで実験を実行し、ときには評価指標と成功基準を改善するためのフィードバックを返し、最良のソリューションを望みのスタックにデプロイし、各実行から学習します。Kapso は教訓バンクを保持しており、各教訓には確信度スコアと、それが適用される範囲を示す境界が付いています。そのため知識は、時間とともに単に蓄積されるのではなく、複利のように積み上がっていきます。

私たちは LinkedIn や Meta AI を含む大企業の中で AI システムを構築し、数百万人のユーザーを抱えるプロダクト向けにデプロイしてきました。Kapso はそのワークフローをエンジンにしたものです。[Leeroo](https://leeroo.com) によるオープンソースで、PyPI に [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) として公開されています。

### 四つのパート

| パート | 動詞 | 内容 |
|------|------|--------------|
| [**スマート実験**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 実験ループは、実験ツリーという一つのデータ構造を共有する専門モジュール群です。各ノードには、仮説、実装（git ブランチと技術的な困難）、使用した予算、目的までの距離が記録されます。アイデア創出は、過去の実験と教訓バンクを踏まえてアイデアを提案し、各アイデアは、特定の過去の実験とどう違うか、どの教訓を採用するかを明示します。リランカーは、時間と予算を勘案しつつ、ターゲットに対する期待リターンでそれらを順位付けします。実装モジュールは親ノードからフォークしたブランチ上で構築し、検証モジュールはノードを安全ルールと成功基準に照らして確認し、スコアが有効かどうか、目的が達成されたかどうかを判定します。成功基準とテストは実行中に進化させることができます。モジュールが根拠を添えた更新を提案し、検証モジュールがそれを承認または却下します。[仕組み →](https://docs.leeroo.com/docs/evolve/overview) |
| [**学習ループ**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 各教訓には、一片の知識、確信度スコア、それが適用される範囲を示す境界条件、そして更新の根拠となる証拠が記録されます。各実行の後、`learn()` は、使用された教訓が結果に与えた効果からその確信度と境界を更新し、類似したトレースから教訓を帰納するか、単一の証拠とそのメカニズムから教訓を演繹し、重複する教訓を統合し、更新した教訓を、同じビルドを教訓ありとなしで実行して A/B テストします。そのため、更新は推測ではなく測定に基づきます。次のビルドのアイデア創出と実装は、知識を単に RAG するのではなく、各教訓の確信度と境界を読んで、いつそれに頼るかを判断します。[Experience →](https://docs.leeroo.com/docs/experience) は、ビルドに何が供給されるかを指定します。あなたのバンク、チームで共有するバンク、Leeroopedia、あるいはリポジトリや調査した問いから取り込んだコーパスです。 |
| [**デプロイ**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | ツリー内の最良のソリューションを、実験ループの短縮版によって、ターゲットスタック上で動作するソフトウェアに変えます。セッションがノードのコピーを適応させ、デプロイし、ヘルスチェックと実際に動く呼び出しを実演します。Docker、Modal、BentoCloud、LangGraph Platform が同梱されており、独自のターゲットスタックを harness に追加することもできます。 |
| [**ドメイン適応**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harness は Kapso のドメインパッケージです。予測モデリングのような分野の知識とワークフローをまとめたもので、これによりビルドはゼロから実験して学習せずに済みます。予測モデリングと競技プログラミングの harness は Kapso に同梱されています。独自のドメイン知識を持ち込むには、Kapso が定義するインターフェースを通じて harness を書きます。あなたの指示、コード、ガードレール、チェックを、プラグインの `/kapso:harness` または CLI の `kapso harness` で作成します。 |

## 🚀 クイックスタート

### インストール

**0. コーディングエージェントの中にいますか？** Kapso プラグインをインストールして、以下の手順を一緒に進めてもらいましょう:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

その後、`/kapso:setup` がインストールと検証を行い、`/kapso:build <what you want>` が最初のキャンペーンを起動します（Codex では `$kapso-setup`、`$kapso-build <what you want>`）。[Kapso プラグイン](https://docs.leeroo.com/docs/coding-agent-skills)を参照してください。

**1. 前提条件。** Kapso は推論をコーディングエージェントの CLI を通じて実行します（直接 API へのフォールバックはありません）。そのため、何かを動かす前に、Node.js と、ログイン済みの両方のエージェント CLI が必要です:

```bash
# Node.js 18+ (https://nodejs.org) を用意してから:
npm install -g @openai/codex            # リサーチ、判定、ユーティリティ
codex login

npm install -g @anthropic-ai/claude-code  # アイデア創出 + 実装 (デフォルトのホスト)
claude auth login
```

OpenAI のキーは任意です。テキストを埋め込むのは、ナレッジグラフと過去の実験に対するセマンティック検索（設定の `models.embedding`）だけで、生成には一切使われません。どちらかを有効にする場合:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia を接続します（推奨）。** キー一つで、すべてのキャンペーンセッションが [Leeroopedia](https://leeroopedia.com) を使えるようになります。これは ML と AI のフレームワークに関するホスト型のナレッジベースで、フレームワークが実際にどう振る舞うか、計画のレビュー、文書化されたハイパーパラメータの範囲、失敗の診断を提供します。インストールするものはありません。[Leeroopedia ダッシュボード](https://app.leeroopedia.com/dashboard)で新規アカウントを作ると、$20 分の無料クレジットが付いてきます。

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. パッケージをインストールします**（Python 3.10+）:

```bash
pip install leeroo-kapso
```

パッケージ名は `leeroo-kapso` です。PyPI にある `kapso` という名前のパッケージは無関係な WhatsApp ツールで、独自の `kapso` コマンドをインストールしてこちらを覆い隠してしまいます。それが入っている場合は、先に `pip uninstall kapso` を実行してください。

**4. セットアップを検証します:**

```bash
kapso doctor
```

`doctor` は、**あなたの設定**が実際に必要とするものを報告し、不足しているものがあれば、その正確な修正方法を示します。要件は設定に従うので、すべて codex の構成で `claude` を求められることはありません。一つの動詞に絞ると、その動詞の要件だけを確認できます:

```bash
kapso doctor build             # または learn / deploy / experience
```

これらは、その動詞自体が作業を始める前に実行するのと同じチェックです。`kapso.build(...)` は、CLI が欠けていれば、セッションの奥深くではなく数秒で失敗します。`[-- ]` と記された項目は任意で、必要としないかもしれない機能（docker、Weaviate、Neo4j、デプロイのターゲット）を制限するだけです。

**あるいは、これらすべてをコーディングエージェントに任せることもできます:** [セットアップ用プロンプト](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)を Claude Code、Codex、または OpenCode に貼り付けてください。

**インデックスに必要なストア（任意）**: `kapso experience add --repo` と `--research` は、コーパスをローカルの Weaviate + Neo4j に取り込みます。ソースのチェックアウトから:

```bash
bash scripts/start_infra.sh   # docker で両方を起動
```

**ソースから（開発用）**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

レガシーの aider アダプターは extra です（`pip install "leeroo-kapso[aider]"`、Python <3.13）。デフォルトの claude/codex エージェントに extra は不要です。

### 基本的な使い方

コアループには、上記の前提条件以外に何も必要ありません:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 開始にナレッジグラフは不要

# Build: 実験を通じてソリューションを作ります。キャンペーンは起動時に
# `status: <path>` を出力します。別のターミナルから次のコマンドでライブで見られます:
#     kapso watch ./campaign
# セッションが、あなたにしか用意できないもの (API キー、ファイル) を必要とすると、
# キャンペーンは一時停止し、`kapso inbox reply` で再開します:
#     https://docs.leeroo.com/docs/build/inbox
# プルリクエストや issue 用に結果を Markdown で共有するには:
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 省略するとゼロから開始
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Learn: 実行したばかりのキャンペーンから学習します。軌跡を掘り起こし、
# 教訓を採点し、証拠で値付けされたナレッジカードをバンクに蓄えます。バンク (ローカルの
# git リポジトリ) は初回利用時に自動で作成されます。共有するまで、教訓は
# あなたのマシンに留まります:
#     kapso experience share <git-url>
# 共有後は、learn() のたびにバンクがそこへ push されます。
lesson = kapso.learn(solution)
print(lesson.explain())

# もう一度 Build。今回は獲得したばかりのカードが供給されます: ビルドに供給されるのは、
# ロードした experience だけです。
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

ストアが稼働していれば、外部の知識をインデックスに取り込み、同じ方法で供給することもできます:

```python
from kapso import Kapso

kapso = Kapso()

# 一度だけ: リポジトリ、またはある問いについて調査したウェブを、インデックスに取り込む
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# インデックスをロードしたビルドはすべてそれを検索します。Leeroopedia はそのキーで利用できます
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

予算についての注意: 取り込みは、素材が裏付けるものをすべて抽出します。たった一つの問いが、相互にリンクされた数十のページと、数時間に及ぶ取り込みになることがあります。取り込み時間は抽出可能な内容の量に応じて増えるため、手持ちのソースすべてではなく、検索するソースを追加してください。

そして、ソリューションを動作するソフトウェアに変えるには:

```python
software = kapso.deploy(solution, target="local")   # または docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # デプロイが実証した呼び出し
software.stop()
```

### キャンペーンに上限を設けるには？

`max_iterations`、`time_budget_minutes`、`cost_budget` を使って、反復回数、経過時間、ベストエフォートでのモデル費用（USD）に上限を設けます。プラットフォームのコストについては、`search_strategy.params.cost` で[コストを考慮した選択](https://docs.leeroo.com/docs/evolve/cost-aware-selection)を有効にし、独自の estimator、`cap_per_candidate`、`campaign_budget` を指定します。その `weight` により、選択がより安価な候補を優先するようにもできます。

### モデルの選択

Kapso が使うすべてのモデルは、一つの設定ファイルに記載されています。同梱のデフォルトでは、ビルドセッションを `claude-opus-5`、学習クルーを `claude-fable-5`、codex のロールを `gpt-5.6-sol` で実行します。ただし、モデルへのアクセスはサブスクリプションに依存します（プランによっては、あるモデルには上限をかけ、別のモデルは提供することがあります）。別のモデルで実行するには、同梱の設定をコピーして編集し、Kapso に自分の設定を指定します:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 例: 学習クルーを fable ではなく opus で実行する:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

テキストのみの推論ロールは、`openai_compatible` の Chat Completions にオプトインできます。モデルとエンドポイントは設定で指定し、認証情報は環境に置いたままにします。設定は `Kapso(config=...)` または `kapso build --config ...` で読み込みます。コーディングとウェブリサーチには、引き続き、対応するツールを備えたエージェントが必要です。[設定例と制限](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)を参照してください。

長時間の実行の前に、設定に記載されたすべてのモデルを、実際のサブスクリプションに対してプローブしてください。あなたのログインで提供できないモデルは、実行開始から数時間後ではなく、ここで数秒のうちに失敗します。提供可能なモデルにかかっている使用量の上限は、一トークンのプローブでは見えません:

```bash
kapso doctor --models                            # 同梱の設定
kapso doctor --models --config kapso-config.yaml # あなたの設定
kapso doctor learn --models                      # 学習クルーのみ
```

このプローブを、忘れずに実行するものではなく、すべての呼び出しの一部にするには、設定で有効にします。異なる `{cli, model}` の組ごとに使い捨てのトークンを一つ消費しますが、無人で数時間に及ぶ `learn()` の前なら、その価値があります:

```yaml
preflight:
  enabled: true            # すべての動詞の前に静的チェック (デフォルト)
  live_model_probe: true   # + 一トークンのモデルプローブ (デフォルト: false)
```

モデルの入れ替えは*ペース*も変えます。設定内のクルーの `timeout_minutes` の上限はデフォルトのモデルで調整されたものであり、より長く推論するモデルに入れ替えた場合は、引き上げが必要になることがあります。

詳細な統合手順については、[クイックスタート](https://docs.leeroo.com/docs/first-build)と[インストール](https://docs.leeroo.com/docs/installation)のガイドを参照してください。

## 例

すべての例は一つのタスクです。ブリーフ、出発点となるリポジトリ、データまたはその入手方法、そして十二行以内のドライバーで構成されます。[examples/README.md](../../examples/README.md) には、所要時間（分）、実行するコマンド一つ、結果が記録されているかどうかが載っています。

| 例 | 示す内容 |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | はじめてのビルド: 用意済みの評価器、ベースライン、ターゲット、記録 |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | 評価器なし: キャンペーンの最初のノードがブリーフのプロトコルから評価器を構築し、すべての候補がそれに照らして測定されます |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | エクスポートに欠陥を含む、DuckDB 上の dbt パイプライン: データコントラクトが守られるようにします |
| [**trading**](../../examples/trading/README.md) | ビルドに harness を載せる: セットアップセッション、各セッションでのページとフック、試行ごとの検証 |
| [**full loop**](../../examples/full-loop/README.md) | build、レビュー、learn、教訓を供給して再び build、deploy: Kaggle の表形式モデルをエンドツーエンドで |
| [**prompt harness**](../../examples/prompt-harness/README.md) | テスト対象としてのプロンプト。AIME の問題で採点します |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | 視覚言語の抽出器。正解の表に照らして採点します |
| [**torch performance**](../../examples/torch-performance/README.md) | 数値許容誤差の範囲内で、CUDA 上のより高速なモジュール |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | CUDA 上のマルチヘッドアテンション: flash attention または Triton |

## 対応ベンチマーク

| ベンチマーク | 説明 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI の ML エンジニアリングベンチマーク。表形式、ビジョン、テキスト、音声にわたる本格的なコンペティションを、生データから採点される提出物まで扱います |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI のアルゴリズム最適化ベンチマーク。数時間に及ぶ探索の中で、コンテスト用ヒューリスティクスを設計、実装、反復します |
| [**RelBench**](../../benchmarks/relbench/README.md) | リレーショナルデータ上の予測 ML を対象とする Stanford のベンチマーク。Amazon、H&M、Stack Exchange、臨床試験などのマルチテーブルデータベースから直接、分類、回帰、レコメンデーションを行います |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | ビジョン、言語、最適化にわたる、時間制限付きのオリンピック形式 ML。専門家が作成した課題、コンテスト用ハードウェア、人間の助けはゼロです |

各ベンチマークには、実行方法、CLI オプション、出力の内容を説明するドキュメントページもあります:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 ドキュメントとサポート

- **完全なドキュメント**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [インストール](https://docs.leeroo.com/docs/installation): コーディングエージェントの CLI、パッケージ、そして `kapso doctor`
  - [クイックスタート](https://docs.leeroo.com/docs/first-build): 最初のキャンペーン
  - [CLI リファレンス](https://docs.leeroo.com/docs/reference/cli): すべてのコマンド、フラグ、デフォルト
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso` のすべての公開メソッド
  - [設定](https://docs.leeroo.com/docs/reference/configuration): `config.yaml` のすべてのキー
  - [learn が生成するもの](https://docs.leeroo.com/docs/learn/what-learn-produces): 教訓バンク。各教訓には確信度スコア、境界、証拠が付きます。あわせて[開発モード](https://docs.leeroo.com/docs/learn/development-mode)
  - [実験ループ](https://docs.leeroo.com/docs/evolve/overview): 実験ツリーとそのモジュール、および[一つの実験の中で起きること](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience): あなたのバンク、Leeroopedia、検索可能な知識に変えられたリポジトリと論文
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local、Docker、Modal、その他のターゲット
  - [Kapso プラグイン](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code、Codex、Cursor 向けのコマンド、スキル、MCP サーバー
- **コーディングエージェントから**: ドキュメントは `https://docs.leeroo.com/mcp` で MCP サーバーを動かしており、すべてのページが Markdown として提供されます。[llms.txt](https://docs.leeroo.com/llms.txt) と[コーディングエージェントで使う Kapso](https://docs.leeroo.com/docs/coding-agent-skills) を参照してください。
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **コーディングエージェントに Kapso を実行させる**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 以下のプラグインは、十個のコマンド、六つのスキル、MCP サーバーを備え、ホストごとに一行でインストールできます（上記のステップ 0）。OpenCode はスキルをフォルダーとして読み込みます（`ln -s <clone>/plugin/skills .opencode/skills`）。[Kapso プラグイン](https://docs.leeroo.com/docs/coding-agent-skills)を参照してください。
- **コーディングエージェントに Kapso のナレッジツールを与える**: パッケージと一緒にインストールされる `kapso-mcp` は、ウェブリサーチ、リポジトリのメモリ、キャンペーンの実験履歴、教訓バンクを、あなたが作業しているセッションに提供します。[MCP ゲート](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)を参照してください。
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI。その後 ~/.codex/config.toml の [mcp_servers.kapso] の下に `startup_timeout_sec = 60` を設定 (Codex の十秒は、負荷の高いマシンでは短い)
  ```
- **コミュニティ**: [Discord](https://discord.gg/hqVbPNNEZM)
- **ウェブサイト**: [leeroo.com](https://leeroo.com)


## エンタープライズ向け Kapso

私たちのエンタープライズ製品はこのエンジンの上に構築されており、どんな人間もデータ資産全体を調査して理解することはできないような、複雑に入り組んだ企業の中にデプロイされてきました。私たちはある企業を、古いデータプラットフォームから新しいレイクハウスへ移行させました。その対象は、30 を超えるソースシステムファミリー、数千のテーブルと変換、数百の BI レポートに及びます。Kapso が取り組むすべてのタスクは、あなたのシステム、データ、苦労して得たルールについての教訓バンクに蓄積されるため、長く働くほどあなたの会社に精通していきます。難度の高いエンタープライズのタスクに Kapso を導入するには、[お問い合わせください](https://leeroo.com/contact-us)。

## 翻訳

この README は他の言語でも読めます。上部のスイッチャーを使うか、[`docs/translations/`](../../docs/translations/) を参照してください。各翻訳は `scripts/translate_readme.py` によって README.md から生成され、`scripts/check_readme_translations.py` によって同期が保たれます。このスクリプトは、翻訳がリンク、数値、コードの行のいずれかで README.md と食い違うと失敗します。翻訳を改善するには、そのフォルダー内のファイルに対してプルリクエストを開いてください。

## コントリビューション

コントリビューションを歓迎します！ はじめ方の詳細は[コントリビューションガイド](../../CONTRIBUTING.md)をご覧ください。

## 引用

Kapso をあなたの仕事で使用する場合は、テクニカルレポートを引用してください:

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
