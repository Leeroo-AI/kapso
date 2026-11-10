<!-- Japanese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">実験を実行し、最良のソリューションをデプロイし、あらゆる実行から学習するエンジン</h4>
<p align="center"><sub>スマートな実験 · 学習 · デプロイ · ドメイン適応</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <b>日本語</b> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">詳しく見る</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Discordに参加</a> ·
  <a href="https://leeroo.com">ウェブサイト</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  このプロジェクトが気に入ったら、スターをつけて応援してください ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **このリポジトリを見ているAIエージェントへ:** セットアップ、目的の記述、harnessの作成、
> 実行のレビューのためのスキルを備えたKapsoプラグインは、一行でインストールできます:
> `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`)。

---

## ニュース

- **IOAI 2026 · AIモデルトラック。** [国際AIオリンピック](https://ioai-official.org)はAI時代のIMOです: 471人の参加者、専門家が設計した六つの課題、GPU一枚の予算。その2026年の[IOAI²](https://ioai-official.org/ai-model-track/)トラックでは、AIシステムに同じ試験を一回の自律セッションで課しました。14のFounding AI Participantsの一つであるKapsoは、主催者の採点で536.07点を獲得し、すべての人間の参加者を上回って、IOAI² Grand Master Trophy、すなわち参加AIシステム中のトップ3を獲得しました。[全結果 →](../../benchmarks/ioai2026/README.md) · [IOAIによるトラックの総括 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **[RelBench](../../benchmarks/relbench/README.md)の三つのリーダーボードすべてで首位。** リレーショナルデータに対する予測MLのStanfordのベンチマークにおいて、Kapsoは[公式リーダーボード](https://star-project.stanford.edu/relbench/leaderboard/)の分類(AUROC 81.2、[KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview)の79.3、Stanford STARの[RT-J](https://star-project.stanford.edu/rt-j)の79.0に対して)、回帰(NMAE 0.2476、0.2711と0.2847に対して)、レコメンデーション(MAP 18.4、他のエントリで最良のID-GNNの9.3に対して)で首位に立っています。

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0 in classification; 0.2476 NMAE against 0.2711 and 0.2847 in regression, lower being better; 18.4 MAP against ID-GNN's 9.3 in recommendations. The first two panels use a truncated axis, the third starts at zero" width="574"></p>

- **[Leeroopedia](https://leeroopedia.com)**: データとMLの領域における私たちの公開レッスンバンクで、論文やリポジトリでの実験から抽出された約100kページを収め、月に約6Mのリクエストを処理しており、そのうち約28%はAIエージェントからのものです。Kapsoのセッションは、アイデア出しと実装の間にMCP経由でこれを照会します。
- **[MLE-Bench](../../benchmarks/mle/README.md)で#1**: KAPSOはKaggleのMLコンペティション(MLE Benchmark)において、オープンソースシステム中で最高位を達成しました。

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574"></p>

- **[ALE-Bench](../../benchmarks/ale/README.md)で#1**: KAPSOは長期的なアルゴリズム発見問題(ALE Benchmark)において最高位を達成しました。

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574"></p>

## Kapsoとは?

Kapsoは、実験を実行し、最良のソリューションをデプロイし、あらゆる実行から学習するオープンソースのエンジンです。ドメインのharnessと空のレッスンバンクから始まり、目的と成功基準を与えられると、次のことを行います:

- **実験ツリーを実行する**: 仮説を提案し、それぞれを実装して検証します。
- **最良のソリューションをデプロイする**: ターゲットスタック上にデプロイします。
- **実行から学習する**: 各仮説と実装には結果への影響度に応じてクレジットが与えられ、バンクが更新されます。各レッスンは信頼度スコアと、それが適用される範囲を示す境界を持ちます。

時間とともに、ドメインのharnessはレッスンバンク、すなわちその環境で実験によって裏付けられたベストプラクティスを蓄えていきます。Kapsoはモデルにもコーディングエージェントにも依存せず、[Leeroo](https://leeroo.com)が開発し、PyPIで[`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/)として公開されています。

### 四つの要素

| 要素 | 動詞 | 何をするか |
|------|------|--------------|
| [**スマートな実験**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 各実験は、ツリーの中で独自のブランチ上に実装されスコア付けされる一つの仮説です。アイデア出しは過去の実験とレッスンバンクを参照し、予算に対する期待リターンでアイデアを再ランク付けするため、素のLLMやRAGの汎用的なアイデアではなく、この環境に適合したものになります。どの実験も触れることのできないフィンガープリント付きの評価器が結果を検証し、モジュールは検証者が承認するための評価の更新を提案できます。[仕組み →](https://docs.leeroo.com/docs/evolve/overview) |
| [**学習**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | すべてのレッスンは実験のトレースから帰納または演繹され、信頼度スコア、適用範囲を示す境界、そしてそのエビデンスを持ちます。`learn()`は各実行の結果からそれらを更新し、あらゆる変更をそれあり・なしで再ビルドしてA/Bテストするため、改善は推測ではなく測定されます。バンクはgitの上に構築されています: `kapso experience share`、`list`、`update`。[Experience →](https://docs.leeroo.com/docs/experience)は、ビルドに何が提供されるかを定めます。 |
| [**デプロイ**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 同じ種類のより短い実験が、最良のソリューションをターゲットスタックに適応させてデプロイし、ヘルスチェックと動作する呼び出しを実証します。Docker、Modal、BentoCloud、LangGraph Platformが同梱されており、harnessはターゲットを追加できます。 |
| [**ドメイン適応**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harnessはKapsoのドメインパッケージです。予測モデリングのような領域の知識とワークフローを収めているため、ビルドがゼロから始まることはなく、レッスンバンクが蓄積される場所でもあります。予測モデリングと競技プログラミングが同梱されています。自分のharnessは、プラグインの`/kapso:harness`またはCLIの`kapso harness`で書けます。 |

## 🚀 クイックスタート

### インストール

**0. コーディングエージェントの中にいますか?** Kapsoプラグインをインストールして、
以下の手順を一緒に進めてもらいましょう:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

その後、`/kapso:setup`がインストールと検証を行い、`/kapso:build <what you want>`が
最初のキャンペーンを起動します(Codexでは: `$kapso-setup`、`$kapso-build <what you want>`)。[Kapsoプラグイン](https://docs.leeroo.com/docs/coding-agent-skills)を参照してください。

**1. 前提条件。** Kapsoは推論をコーディングエージェントのCLI経由で実行します
(直接APIへのフォールバックはありません)。そのため、何かが動く前にNode.jsと
両方のエージェントCLIにログインしておく必要があります:

```bash
# Node.js 18+ (https://nodejs.org)、その後:
npm install -g @openai/codex            # リサーチ、判定、ユーティリティ
codex login

npm install -g @anthropic-ai/claude-code  # アイデア出し + 実装(デフォルトのホスト)
claude auth login
```

OpenAIのキーは任意です。テキストを埋め込むのはナレッジグラフと過去の実験に対する
セマンティック検索(設定の`models.embedding`)だけで、生成では決して使われません。
どちらかを有効にする場合:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopediaに接続する(推奨)。** 一つのキーで、すべてのセッションが
MLとAIフレームワークに関する私たちのホスト型レッスンバンクである
[Leeroopedia](https://leeroopedia.com)を利用できます。インストールするものはありません。
[Leeroopediaダッシュボード](https://app.leeroopedia.com/dashboard)で新規アカウントを作ると
$20の無料クレジットが付いてきます。

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. パッケージをインストールする** (Python 3.10+):

```bash
pip install leeroo-kapso
```

**4. セットアップを検証する:**

```bash
kapso doctor
```

`doctor`は**あなたの設定**が実際に必要とするものを報告し、足りないものがあれば
その正確な修正方法を示します。要件は設定に従うため、すべてcodexの構成で
`claude`を求められることはありません。一つの動詞に絞れば、その動詞の
要件だけを確認できます:

```bash
kapso doctor build             # または learn / deploy / experience
```

これらは、動詞自体が作業を始める前に実行するのと同じチェックです。
`kapso.build(...)`は、CLIが欠けていればセッションの奥深くではなく
数秒で失敗します。`[-- ]`と印の付いた項目は任意で、必要ないかもしれない
機能(docker、Weaviate、Neo4j、デプロイターゲット)に関わるものです。

**あるいは、これらすべてをコーディングエージェントに任せる:**
[セットアッププロンプト](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
をClaude Code、Codex、またはOpenCodeに貼り付けてください。

**インデックスに必要なストア(任意)。** `kapso experience add --repo`
と`--research`は、コーパスをローカルのWeaviate + Neo4jに取り込みます。
ソースのチェックアウトから:

```bash
bash scripts/start_infra.sh   # docker経由で両方を起動
```

**ソースから(開発用)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

レガシーのaiderアダプタはエクストラです(`pip install "leeroo-kapso[aider]"`、
Python <3.13)。デフォルトのclaude/codexエージェントにはエクストラは不要です。

### 基本的な使い方

コアとなる動詞には、上記の前提条件以外は何も必要ありません:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 始めるのにナレッジグラフは不要

# Build: 実験を通じてソリューションを得る。キャンペーンは起動時に
# `status: <path>` を出力する。別のターミナルから次のコマンドでライブで見守れる
#     kapso watch ./campaign
# セッションがあなたにしか提供できないもの(APIキー、ファイル)を必要とすると、
# キャンペーンは一時停止し、`kapso inbox reply` が再開させる:
#     https://docs.leeroo.com/docs/build/inbox
# 結果をプルリクエストやissue用のMarkdownとして共有するには
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # または省略してゼロから始める
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 今実行したキャンペーンから学習する: トラジェクトリを掘り起こし、レッスンを
# 採点し、エビデンスで価値付けされた知識カードをバンクに収める。バンク(ローカルの
# gitリポジトリ)は初回使用時に自動で作成される。レッスンは共有するまで
# あなたのマシンに留まる:
#     kapso experience share <git-url>
# 共有後は、learn() のたびにバンクがそこへプッシュされる。
lesson = kapso.learn(solution)
print(lesson.explain())

# 獲得したばかりのカードを与えて再びビルドする: ビルドには、ロードした
# experienceがそのまま提供される。
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

ストアが稼働していれば、外部の知識をインデックスに取り込み、
同じ方法で提供することもできます:

```python
from kapso import Kapso

kapso = Kapso()

# 一度だけ: リポジトリ、またはある問いについてウェブをリサーチした結果をインデックスへ
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# インデックスをロードしたすべてのビルドがそれを検索する。Leeroopediaはそのキーだけで使える
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

予算についての注意: 取り込みは、素材が裏付けるすべてを抽出します。
一つの質問が数十のリンクされたページと数時間に及ぶ取り込みになることも
あります。取り込み時間は抽出できる内容の量に比例するため、手元のすべての
ソースではなく、実際に検索するソースを追加してください。

そして、ソリューションを動作するソフトウェアにするには:

```python
software = kapso.deploy(solution, target="local")   # または docker、modal、bentoml、langgraph、auto
result = software.run(software.proof["inputs"])     # デプロイが実証した呼び出し
software.stop()
```

### キャンペーンを制限するには?

`max_iterations`、`time_budget_minutes`、`cost_budget`を使って、イテレーション数、経過時間、ベストエフォートでのモデル費用(USD)を制限します。
プラットフォームのコストについては、`search_strategy.params.cost`で、あなたの推定器、`cap_per_candidate`、`campaign_budget`とともに[コストを考慮した選択](https://docs.leeroo.com/docs/evolve/cost-aware-selection)を有効にしてください。その`weight`により、選択がより安価な候補を優先するようにもできます。

### モデルの選択

Kapsoが使うすべてのモデルは、一つの設定ファイルで指定されています。同梱の
デフォルトでは、ビルドセッションを`claude-opus-5`で、学習クルーを
`claude-fable-5`で、codexのロールを`gpt-5.6-sol`で実行します。ただし、モデルへの
アクセスはサブスクリプション次第です(あるプランでは、あるモデルを提供しつつ別のモデルに上限を設けることがあります)。
別のモデルで実行するには、同梱の設定をコピーして編集し、
Kapsoにそれを指すようにします:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 例: 学習クルーをfableではなくopusで実行する:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

テキストのみの推論ロールは、`openai_compatible`のChat Completionsを選択できます。
設定でモデルとエンドポイントを指定してください。認証情報は環境変数に置いたままにします。
設定は`Kapso(config=...)`または`kapso build --config ...`で読み込みます。
コーディングとウェブリサーチには、引き続き対応するツールを持つエージェントが必要です。
[設定例と制限](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)を参照してください。

長時間の実行の前に、設定で指定したすべてのモデルを実際のサブスクリプションに
対してプローブしてください。あなたのログインでは提供できないモデルは、実行開始から
数時間後ではなく、ここで数秒のうちに失敗します。提供できるモデルの使用量上限は、
一トークンのプローブでは見えません:

```bash
kapso doctor --models                            # 同梱の設定
kapso doctor --models --config kapso-config.yaml # あなたの設定
kapso doctor learn --models                      # 学習クルーのみ
```

このプローブを、思い出して実行するものではなく、すべての呼び出しの一部にするには、
設定で有効にしてください。コストは異なる`{cli, model}`の組ごとに使い捨てトークン
一つ分で、無人で数時間走る`learn()`の前には十分に価値があります:

```yaml
preflight:
  enabled: true            # すべての動詞の前の静的チェック(デフォルト)
  live_model_probe: true   # + 一トークンのモデルプローブ(デフォルト: false)
```

モデルの差し替えは*ペース*も変えます。設定内のクルーの`timeout_minutes`上限は
デフォルトのモデルで調整されたもので、より長く推論する差し替えモデルでは
引き上げが必要になることがあります。

詳細な統合手順については、[クイックスタート](https://docs.leeroo.com/docs/first-build)と[インストール](https://docs.leeroo.com/docs/installation)のガイドを参照してください。

## 対応ベンチマーク

| ベンチマーク | 説明 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAIのMLエンジニアリングベンチマーク。表形式、画像、テキスト、音声にわたる完全なコンペティションを、生データから採点される提出物まで行う |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AIのアルゴリズム最適化ベンチマーク。数時間に及ぶ探索の中で、コンテストのヒューリスティクスを設計、実装、反復する |
| [**RelBench**](../../benchmarks/relbench/README.md) | リレーショナルデータに対する予測MLのStanfordのベンチマーク。Amazon、H&M、Stack Exchange、臨床試験などのマルチテーブルデータベースから直接、分類、回帰、レコメンデーションを行う |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 画像、言語、最適化にわたる時間制限付きのオリンピアードML。専門家が設定した課題、コンテストのハードウェア、人間の助けはゼロ |

各ベンチマークには、実行方法、CLIオプション、出力に含まれる内容を説明するドキュメントページもあります:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 例

どの例も一つのタスクです。ブリーフ、開始時のリポジトリ、データまたはその入手方法、そして最大十二行のドライバから成ります。[examples/README.md](../../examples/README.md)に、所要時間(分)、実行する一つのコマンド、結果が記録されているかどうかが載っています。

| 例 | タスク |
|---------|----------|
| [**データエンジニアリング(Snowflake上のdbtモデル)**](../../examples/dbt-snowflake/README.md) | Snowflake上の小売データウェアハウスで、データ品質に問題のあるdbtモデルを修正し、十七件のチケットからなるバックログが求める新しいモデルを構築して、すべてのチケットが本番テストに合格するまで進める。その後、チケットごとにプルリクエストを届け、本番環境へリリースする。 |
| [**harness最適化(カスタマーサポートエージェント)**](../../examples/support-agent-harness/README.md) | カスタマーサポートエージェントのharnessを最適化し、顧客との会話の解決率を落とさずに、基盤のLLMをFable 5.1からローカルのQwen 27Bへ切り替える。その後Dockerでデプロイし、その実行から学習する。 |
| [**予測モデリング(顧客チャーン)**](../../examples/churn-first-build/README.md) | どの顧客が離脱するかを予測する: キャンペーンが編集できない評価器のもとで、チャーンモデルのアウトオブサンプルのROC AUCを目標に向けて引き上げる。 |
| [**不正検知**](../../examples/fraud-no-evaluator/README.md) | リポジトリに評価器がない状態で取引の不正をスコアリングする: ブリーフがプロトコルを定め、キャンペーンはまず評価器を構築し、その後スコアラーを改善する。 |
| [**クオンツファイナンス**](../../examples/trading/README.md) | 合成銘柄に対する日次トレーディング戦略: トレーディングharness上でアウトオブサンプルのシャープレシオを引き上げる。すべての試行にそのチェックが適用される。研究用の例であり、投資助言ではない。 |
| [**表形式データのML**](../../examples/end-to-end/README.md) | Kaggleコンペティションをエンドツーエンドで: Spaceship Titanicをダミー分類器から本物のモデルへ、その後レビューし、学習し、レッスンを使って再びビルドし、デプロイする。 |
| [**PyTorchの最適化**](../../examples/torch-performance/README.md) | 数値許容誤差の範囲内でPyTorchモジュールをCUDA上で高速化する: 融合、コンパイル、メモリトラフィックの削減。 |
| [**GPUカーネルの最適化**](../../examples/gpu-kernels/README.md) | 数値許容誤差の範囲内で、flash attentionまたはTritonカーネルを使ってCUDA上のマルチヘッドセルフアテンションを高速化する。 |

## 📚 ドキュメントとサポート

- **完全なドキュメント**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [インストール](https://docs.leeroo.com/docs/installation): コーディングエージェントのCLI、パッケージ、`kapso doctor`
  - [クイックスタート](https://docs.leeroo.com/docs/first-build): 最初のキャンペーン
  - [CLIリファレンス](https://docs.leeroo.com/docs/reference/cli): すべてのコマンド、フラグ、デフォルト値
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso`のすべての公開メソッド
  - [設定](https://docs.leeroo.com/docs/reference/configuration): `config.yaml`のすべてのキー
  - [learnが生み出すもの](https://docs.leeroo.com/docs/learn/what-learn-produces): レッスンバンク(各レッスンは信頼度スコア、境界、エビデンスを持つ)と[開発モード](https://docs.leeroo.com/docs/learn/development-mode)
  - [スマートな実験](https://docs.leeroo.com/docs/evolve/overview): 実験ツリーとそのモジュール、そして[一つの実験の内部で起きること](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience): あなたのバンク、Leeroopedia、検索可能な知識に変換されたリポジトリと論文
  - [デプロイ](https://docs.leeroo.com/docs/deploy/targets): local、Docker、Modal、その他のターゲット
  - [Kapsoプラグイン](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code、Codex、Cursor向けのコマンド、スキル、MCPサーバー
- **コーディングエージェントから**: ドキュメントは`https://docs.leeroo.com/mcp`でMCPサーバーを提供しており、すべてのページはMarkdownとして配信されます。[llms.txt](https://docs.leeroo.com/llms.txt)と[コーディングエージェントの中のKapso](https://docs.leeroo.com/docs/coding-agent-skills)を参照してください。
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **コーディングエージェントにKapsoを実行させる**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin)配下のプラグインには、十のコマンド、六つのスキル、MCPサーバーが含まれており、ホストごとにインストールは一行です(上記のステップ0)。OpenCodeはスキルをフォルダとして読み込みます(`ln -s <clone>/plugin/skills .opencode/skills`)。[Kapsoプラグイン](https://docs.leeroo.com/docs/coding-agent-skills)を参照してください。
- **コーディングエージェントにKapsoの知識ツールを与える**: パッケージとともにインストールされる`kapso-mcp`は、ウェブリサーチ、リポジトリのメモリ、キャンペーンの実験履歴、レッスンバンクを、あなたが作業しているセッションに提供します。[MCPゲート](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)を参照してください。
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI。その後 ~/.codex/config.toml の [mcp_servers.kapso] に `startup_timeout_sec = 60` を設定する(Codexの十秒は、混み合ったマシンでは短い)
  ```
- **コミュニティ**: [Discord](https://discord.gg/hqVbPNNEZM)
- **ウェブサイト**: [leeroo.com](https://leeroo.com)


## エンタープライズ向けKapso

エンタープライズでは、Kapsoのエージェントは初期のデータおよびAIエンジニアリングのharnessとレッスンバンクを備えて提供され、企業のアーティファクト、リポジトリ、パイプライン、データソースなどを調査しながら、ビジネス上の目的を自力で動作するAIおよびデータシステムへと変え、その過程でエンタープライズのデータ資産を築き上げていきます。エンタープライズのタスクにKapsoを導入するには、[お問い合わせください](https://leeroo.com/contact-us)。

## コントリビューション

コントリビューションを歓迎します! 始め方の詳細は[コントリビューションガイド](../../CONTRIBUTING.md)をご覧ください。

## コントリビューター

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

[contrib.rocks](https://contrib.rocks)で作成。
