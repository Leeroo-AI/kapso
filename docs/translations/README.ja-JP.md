<!-- Japanese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">実験を実行し、最良のソリューションをデプロイし、すべての実行から学習するエンジン</h4>
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
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **このリポジトリを見ている AI エージェントへ:** セットアップ、目標の記述、harness の作成、
> 実行のレビューのためのスキルを備えた Kapso プラグインは、一行でインストールできます:
> `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`)。

---

## ニュース

- **IOAI 2026 · AI モデルトラック。** [International Olympiad in AI](https://ioai-official.org) は AI 時代の IMO です。108 の国と地域から 471 名の出場者が集まり、専門家が設計した六つのタスクに単一 GPU の予算で挑みます。2026 年には [IOAI²](https://ioai-official.org/ai-model-track/) が開設され、AI システムが人間と*同じ*試験を、無人の 6 時間セッション二回で受験します。時計が動き出した後は、人間が何かを解いたり、修正したり、改善したりすることは一切許されません。Kapso は 14 の Founding AI Participants の一つとして参加しました。
  - 🎓 **471 名の人間全員を上回る得点**: 合計 536.07 で、会場のすべての出場者を上回りました。
  - 🏆 **IOAI² Grand Master Trophy**: 参加した全 AI システムの中でトップ 3。[全結果 →](../../benchmarks/ioai2026/README.md) · [IOAI によるトラックのレビュー →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **[RelBench](../../benchmarks/relbench/README.md) の三つのリーダーボードすべてで一位**: リレーショナルデータ上の予測 ML に関する Stanford のベンチマークで、Kapso は公式リーダーボードの分類 (12 タスクで 81.2 AUROC。[KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) は 79.3、Stanford STAR の [RT-J](https://star-project.stanford.edu/rt-j) は 79.0)、回帰 (9 タスクで 0.2476 NMAE。RT-J は 0.2711、KumoRFM-v2 は 0.2847 で、誤差は 8.7% 少ない)、推薦 (10 タスクで 18.4 MAP。他のエントリで最良の ID-GNN は 9.3) を首位で率いています。すべての数値は[公式 RelBench リーダーボード](https://star-project.stanford.edu/relbench/leaderboard/)上のエントリです。harness の中に保持された学習が、モデルの重みに焼き込まれた学習を上回り、教訓バンクの更新は表形式基盤モデルの再学習よりはるかに低コストです。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)**: データと ML のドメインにおける私たちの教訓の wiki で、論文やリポジトリで実験して抽出した約 100k ページを収め、月に約 6M リクエストを処理しており、そのうち約 28% は AI エージェントからのものです。Kapso は MCP 経由でこれに接続します。セッションはアイデア出しと実装の間にこれを使い、知識を検索し、計画を立て、失敗を診断します。
- **[MLE-Bench](../../benchmarks/mle/README.md) で #1**: KAPSO は Kaggle ML コンペティション (MLE Benchmark) においてオープンソースシステムの中で最高ランクを達成しました。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **[ALE-Bench](../../benchmarks/ale/README.md) で #1**: KAPSO は長期的なアルゴリズム発見問題 (ALE Benchmark) において最高ランクを達成しました。

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## Kapso とは?

企業の中で AI やデータの成果を届けるということは、二つのことを実験によって学ぶということです。データ資産がどのように接続され運用されているかを知っている個人や AI エージェントは存在しません。それを知る唯一の方法は、企業のアーティファクト (リポジトリ、パイプライン、ドキュメント、データソース) を調査し、仮説を立て、実験を実行し、そこから学ぶことを、各システムの振る舞いとその条件が理解されるまで続けることです。そしてその理解が得られた後でも、ウェアハウスの最適化であれ解約予測であれ、最良のソリューションを事前に選ぶことはできないため、環境に適合するものが見つかるまで実験と学習を繰り返します。環境に適合するノウハウは、RL とよく似た形で蓄積されていきます。RAG はその逆を行います。この環境で一度も検証されたことのない一般的な知識を、すべて同じ重みでエージェントに与えるのです。そうした知識は、その環境で試されるまではほとんど価値を持ちません。

Kapso は、私たちがこのために構築したエンジンです。ドメイン harness と空の教訓バンクから出発し、目標と成功基準を与えられると、仮説を提案し、それぞれを実装して検証しながら実験ツリーを実行し、最良のソリューションをターゲットスタックにデプロイします。その後、実行から学習し、各仮説と実装に結果への影響に応じたクレジットを割り当ててバンクを更新します。バンク内のすべての教訓は、信頼度スコアと、それが適用される範囲を示す境界を持ちます。時間とともに、ドメイン harness はその環境で実験済みのベストプラクティスを蓄積した教訓バンクを備えるようになります。このシステムはモデル非依存かつコーディングエージェント非依存です。[Leeroo](https://leeroo.com) によるオープンソースで、PyPI に [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) として公開されています。

### 四つの構成要素

| 構成要素 | 動詞 | 機能 |
|------|------|--------------|
| [**スマートな実験**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 専門化されたモジュール群が、実験ツリーという一つのデータ構造を共有します。各ノードは、仮説、実装 (git ブランチと技術的な困難)、使用した予算、目標までの距離を記録します。各実験ごとに、アイデア出しモジュールが過去の実験と教訓バンクからアイデアを提案し、それぞれが使う予算に対する期待リターンで再ランク付けし、上位のものを実装モジュールに送ります。実装モジュールは親ノードからフォークしたブランチの上で構築します。バンクはまさにその環境で何が機能するかを学び続けるため、アイデアは素の LLM や RAG が出すような一般的なものではなく、その環境に特化したものになります。結果は、どの実験もアクセスできないフィンガープリント付きの評価器によって検証され、評価自体も進化します。実験モジュールは、検証者が承認するためのメトリクスと基準の更新を提案できます。[仕組み →](https://docs.leeroo.com/docs/evolve/overview) |
| [**学習**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | バンクの中身はすべて実験に由来します。各教訓は実験実行のトレースから帰納または演繹され、信頼度スコア、それが適用される範囲の境界条件、そしてその背後にある証拠を持ち、毎回の実行の後に `learn()` が最終結果に対する教訓の効果からそれらを更新します。バンクへの変更もまた実験です。更新されたすべての教訓は、同じ build をその教訓あり・なしで実行して A/B テストされるため、改善は推測ではなく測定されます。バンクは git 上に構築されているため、自分の経験を他者と共有し (`kapso experience share`)、利用可能なバンクを一覧し (`kapso experience list`)、更新する (`kapso experience update`) ことができます。[Experience →](https://docs.leeroo.com/docs/experience) は、build に何が提供されるかを定めます。自分のバンク、チームで共有するバンク、Leeroopedia、あるいはリポジトリや調査した質問から取り込んだコーパスです。 |
| [**デプロイ**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 最良のソリューションをターゲットスタックに適合させるため、同種のより短い実験を実行します。セッションがノードのコピーを適合させ、デプロイし、ヘルスチェックと動作する呼び出しを実証します。Docker、Modal、BentoCloud、LangGraph Platform が同梱されており、独自のターゲットスタックを harness に追加することもできます。 |
| [**ドメイン適応**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harness は Kapso のドメインパッケージです。予測モデリングのような領域に関する知識とワークフローを収めており、build がゼロから実験して学ぶ必要をなくします。予測モデリングと競技プログラミングの harness が Kapso に同梱されています。実行はドメイン harness と空の教訓バンクから始まり、時間とともに harness はその環境で機能したものを蓄積したバンクを備えるようになります。最初のドメイン harness、つまり指示、コード、ガードレール、チェックは、プラグインの `/kapso:harness` または CLI の `kapso harness` で作成します。 |

## 🚀 クイックスタート

### インストール

**0. コーディングエージェントの中にいますか?** Kapso プラグインをインストールし、
以下のステップを一緒に進めてもらいましょう:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

その後、`/kapso:setup` がインストールと検証を行い、`/kapso:build <what you want>` が
最初のキャンペーンを起動します (Codex では `$kapso-setup`、`$kapso-build <what you want>`)。[Kapso プラグイン](https://docs.leeroo.com/docs/coding-agent-skills)を参照してください。

**1. 前提条件。** Kapso は推論をコーディングエージェントの CLI 経由で実行します
(直接 API へのフォールバックはありません)。そのため、何かが動く前に Node.js と
両方のエージェント CLI にログインしておく必要があります:

```bash
# Node.js 18+ (https://nodejs.org)、その後:
npm install -g @openai/codex            # リサーチ、判定、ユーティリティ
codex login

npm install -g @anthropic-ai/claude-code  # アイデア出し + 実装 (デフォルトのホスト)
claude auth login
```

OpenAI のキーは任意です。テキストを埋め込むのは、ナレッジグラフと過去の実験に対する
セマンティック検索 (設定の `models.embedding`) だけで、生成には一切使われません。
どちらかを有効にする場合:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia に接続する (推奨)。** 一つのキーで、すべてのキャンペーンセッションが
[Leeroopedia](https://leeroopedia.com) を使えるようになります。これは ML と AI フレームワークに
関するホスト型のナレッジベースで、フレームワークの実際の振る舞い、計画のレビュー、
文書化されたハイパーパラメータの範囲、失敗の診断を提供します。インストールするものはありません。
[Leeroopedia ダッシュボード](https://app.leeroopedia.com/dashboard)で新しいアカウントを作成すると、
$20 の無料クレジットが付いてきます。

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. パッケージをインストールする** (Python 3.10+):

```bash
pip install leeroo-kapso
```

パッケージ名は `leeroo-kapso` です。PyPI の `kapso` という名前のパッケージは無関係な
WhatsApp ツールで、独自の `kapso` コマンドをインストールしてこちらを覆い隠してしまいます。
それが入っている場合は、先に `pip uninstall kapso` を実行してください。

**4. セットアップを検証する:**

```bash
kapso doctor
```

`doctor` は**あなたの設定**が実際に必要とするものを報告し、不足しているものに対する
正確な修正方法を示します。要件は設定に従うため、すべて codex で構成されたセットアップで
`claude` を求められることはありません。一つの動詞に絞れば、その動詞の要件だけを
確認できます:

```bash
kapso doctor build             # または learn / deploy / experience
```

これらは、各動詞自身が作業を始める前に実行するのと同じチェックです。
`kapso.build(...)` は、CLI が欠けていればセッションの奥深くではなく数秒で失敗します。
`[-- ]` と表示された項目は任意で、必要ないかもしれない機能
(docker、Weaviate、Neo4j、deploy のターゲット) を制限するだけです。

**あるいは、これらすべてをコーディングエージェントに任せることもできます:**
[セットアッププロンプト](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)を
Claude Code、Codex、または OpenCode に貼り付けてください。

**インデックスに必要なストア (任意)**: `kapso experience add --repo` と
`--research` はコーパスをローカルの Weaviate + Neo4j に取り込みます。
ソースのチェックアウトから:

```bash
bash scripts/start_infra.sh   # docker 経由で両方を起動
```

**ソースから (開発用)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

レガシーの aider アダプタはエクストラとして提供されます (`pip install "leeroo-kapso[aider]"`、
Python <3.13)。デフォルトの claude/codex エージェントにはエクストラは不要です。

### 基本的な使い方

コアとなる動詞は、上記の前提条件以外に何も必要としません:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 開始にナレッジグラフは不要

# Build: 実験を通じてソリューションを得る。キャンペーンは起動時に
# `status: <path>` を出力する。別のターミナルから次のコマンドでライブ観察できる:
#     kapso watch ./campaign
# セッションがあなたにしか提供できないもの (API キー、ファイル) を必要とする場合、
# キャンペーンは一時停止し、`kapso inbox reply` で再開する:
#     https://docs.leeroo.com/docs/build/inbox
# 結果をプルリクエストや issue 向けの Markdown として共有するには
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # または省略してゼロから開始
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 実行したばかりのキャンペーンから学ぶ: トラジェクトリを掘り起こし、教訓を
# 採点し、証拠で価格付けされたナレッジカードをバンクに収める。バンク (ローカルの
# git リポジトリ) は初回使用時に自動で作成される。教訓は共有するまで
# あなたのマシン上に留まる:
#     kapso experience share <git-url>
# 共有後は、毎回の learn() がバンクをそこにプッシュする。
lesson = kapso.learn(solution)
print(lesson.explain())

# 獲得したばかりのカードを提供されて再び build する: build には、
# ロードした経験がそのまま提供される。
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

ストアが起動していれば、外部の知識をインデックスに取り込み、同じ方法で
提供することもできます:

```python
from kapso import Kapso

kapso = Kapso()

# 一度だけ: リポジトリ、または質問についてウェブを調査した結果を、インデックスへ
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# インデックスをロードしたすべての build がそれを検索する。Leeroopedia はそのキーがあれば利用できる
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

予算についての注意: 取り込みは、素材が裏付けるすべてを抽出します。
たった一つの質問が数十のリンクされたページと数時間に及ぶ取り込みになることもあります。
取り込み時間は抽出可能な実質的内容に比例するため、手元にあるすべてのソースではなく、
検索するつもりのソースを追加してください。

そして、ソリューションを動作するソフトウェアにするには:

```python
software = kapso.deploy(solution, target="local")   # または docker、modal、bentoml、langgraph、auto
result = software.run(software.proof["inputs"])     # デプロイが実証した呼び出し
software.stop()
```

### キャンペーンをどのように制限しますか?

`max_iterations`、`time_budget_minutes`、`cost_budget` を使って、反復回数、経過時間、ベストエフォートでのモデル支出 (USD) を制限します。
プラットフォームのコストについては、`search_strategy.params.cost` で[コストを考慮した選択](https://docs.leeroo.com/docs/evolve/cost-aware-selection)を有効にし、独自の見積もり器、`cap_per_candidate`、`campaign_budget` を設定します。その `weight` により、選択がより安価な候補を優先するようにもできます。

### モデルの選択

Kapso が使うすべてのモデルは一つの設定ファイルで指定されます。同梱のデフォルトでは、
build セッションを `claude-opus-5` で、学習クルーを `claude-fable-5` で、
codex のロールを `gpt-5.6-sol` で実行します。ただし、モデルへのアクセスは
サブスクリプションに依存します (あるプランでは、あるモデルを提供しながら別のモデルに上限を設けることがあります)。
別のモデルで実行するには、同梱の設定をコピーして編集し、Kapso にそれを指定します:

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

テキストのみの推論ロールは、`openai_compatible` の Chat Completions をオプトインで使えます。
モデルとエンドポイントは設定で指定し、認証情報は環境変数に置いたままにします。
設定は `Kapso(config=...)` または `kapso build --config ...` でロードします。
コーディングとウェブリサーチには、引き続き対応するツールを持つエージェントが必要です。
[設定例と制限](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)を参照してください。

長時間の実行の前に、設定で指定したすべてのモデルを実際のサブスクリプションに対して
プローブしてください。ログインでは提供できないモデルは、実行開始から数時間後ではなく、
ここで数秒のうちに失敗します。提供できるモデルの使用量上限は、一トークンのプローブでは
見えません:

```bash
kapso doctor --models                            # 同梱の設定
kapso doctor --models --config kapso-config.yaml # あなたの設定
kapso doctor learn --models                      # 学習クルーのみ
```

このプローブを、思い出して実行するものではなく毎回の呼び出しの一部にするには、
設定で有効にします。コストは、異なる `{cli, model}` のペアごとに使い捨てトークン一つだけで、
無人で数時間続く `learn()` の前には十分見合うものです:

```yaml
preflight:
  enabled: true            # すべての動詞の前の静的チェック (デフォルト)
  live_model_probe: true   # + 一トークンのモデルプローブ (デフォルト: false)
```

モデルの入れ替えは*ペース*も変えます。設定内のクルーの `timeout_minutes` 上限は
デフォルトのモデルで調整されたもので、より長く推論する別のモデルに替えた場合は、
上限を引き上げる必要があるかもしれません。

詳細な統合手順については、[クイックスタート](https://docs.leeroo.com/docs/first-build)と[インストール](https://docs.leeroo.com/docs/installation)のガイドを参照してください。

## 対応ベンチマーク

| ベンチマーク | 説明 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI の ML エンジニアリングベンチマーク。表形式、画像、テキスト、音声にわたる本格的なコンペティションを、生データから採点される提出物まで扱う |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI のアルゴリズム最適化ベンチマーク。数時間に及ぶ探索の中で、コンテスト用ヒューリスティクスを設計、実装、反復する |
| [**RelBench**](../../benchmarks/relbench/README.md) | リレーショナルデータ上の予測 ML に関する Stanford のベンチマーク。Amazon、H&M、Stack Exchange、臨床試験などの複数テーブルのデータベースから直接、分類、回帰、推薦を行う |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 画像、言語、最適化にわたる時間制限付きのオリンピアド ML。専門家が設定したタスク、コンテスト用ハードウェア、人間の助けは一切なし |

各ベンチマークには、実行方法、CLI オプション、出力の内容を説明するドキュメントページもあります:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 例

各例は一つのタスクです。ブリーフ、開始時のリポジトリ、データまたはその入手方法、そして最大十二行のドライバで構成されます。[examples/README.md](../../examples/README.md) には、所要時間 (分)、唯一のコマンド、結果が記録されているかどうかが載っています。

| 例 | 示すもの |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | 初めての build: 評価器があり、ベースライン、ターゲット、記録がそろっている |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | 評価器なし: キャンペーンの最初のノードがブリーフのプロトコルから評価器を構築し、すべての候補がそれに対して測定される |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | エクスポートに欠陥を抱えた DuckDB 上の dbt パイプライン: データコントラクトを守らせる |
| [**trading**](../../examples/trading/README.md) | build に harness を適用: セットアップセッション、すべてのセッションでのページとフック、試行ごとの検証 |
| [**end to end**](../../examples/end-to-end/README.md) | build、レビュー、learn、教訓を提供されての再 build、deploy: Kaggle の表形式モデルをエンドツーエンドで |
| [**prompt harness**](../../examples/prompt-harness/README.md) | プロンプトをテスト対象として、AIME の問題でスコアリング |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | 画像言語抽出器を、正解テーブルに対してスコアリング |
| [**torch performance**](../../examples/torch-performance/README.md) | 数値許容誤差の範囲内で、CUDA 上のより高速なモジュールを |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | CUDA 上のマルチヘッドアテンション: flash attention または Triton |

## 📚 ドキュメントとサポート

- **完全なドキュメント**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [インストール](https://docs.leeroo.com/docs/installation): コーディングエージェントの CLI、パッケージ、そして `kapso doctor`
  - [クイックスタート](https://docs.leeroo.com/docs/first-build): 初めてのキャンペーン
  - [CLI リファレンス](https://docs.leeroo.com/docs/reference/cli): すべてのコマンド、フラグ、デフォルト値
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso` のすべての公開メソッド
  - [設定](https://docs.leeroo.com/docs/reference/configuration): `config.yaml` のすべてのキー
  - [learn が生み出すもの](https://docs.leeroo.com/docs/learn/what-learn-produces): 教訓バンク (各教訓は信頼度スコア、境界、証拠を持つ) と[開発モード](https://docs.leeroo.com/docs/learn/development-mode)
  - [スマートな実験](https://docs.leeroo.com/docs/evolve/overview): 実験ツリーとそのモジュール、そして[一つの実験の中で何が起きるか](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience): 自分のバンク、Leeroopedia、検索可能な知識に変換されたリポジトリと論文
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local、Docker、Modal、その他のターゲット
  - [Kapso プラグイン](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code、Codex、Cursor 向けのコマンド、スキル、MCP サーバー
- **コーディングエージェントから**: ドキュメントは `https://docs.leeroo.com/mcp` で MCP サーバーを稼働させており、すべてのページは Markdown として提供されます。[llms.txt](https://docs.leeroo.com/llms.txt) と[コーディングエージェントの中の Kapso](https://docs.leeroo.com/docs/coding-agent-skills)を参照してください。
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **コーディングエージェントに Kapso を実行させる**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 配下のプラグインには、十個のコマンド、六つのスキル、MCP サーバーが含まれ、ホストごとに一行でインストールできます (上記のステップ 0)。OpenCode はスキルをフォルダとしてロードします (`ln -s <clone>/plugin/skills .opencode/skills`)。[Kapso プラグイン](https://docs.leeroo.com/docs/coding-agent-skills)を参照してください。
- **コーディングエージェントに Kapso の知識ツールを与える**: パッケージと一緒にインストールされる `kapso-mcp` は、ウェブリサーチ、リポジトリのメモリ、キャンペーンの実験履歴、教訓バンクを、あなたが作業しているセッションに提供します。[MCP ゲート](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)を参照してください。
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI。その後、~/.codex/config.toml の [mcp_servers.kapso] の下に `startup_timeout_sec = 60` を設定 (Codex の十秒は、負荷の高いマシンでは短すぎる)
  ```
- **コミュニティ**: [Discord](https://discord.gg/hqVbPNNEZM)
- **ウェブサイト**: [leeroo.com](https://leeroo.com)


## エンタープライズ向け Kapso

エンタープライズでは、Kapso エージェントは初期のデータおよび AI エンジニアリング harness と教訓バンクを備えて提供され、ビジネス上の目標を自律的に動作する AI とデータシステムに変えます。その過程で、企業のアーティファクト、リポジトリ、パイプライン、データソースなどを調査し、企業のデータ資産を築き上げていきます。エンタープライズのタスクに Kapso を導入するには、[お問い合わせください](https://leeroo.com/contact-us)。

## コントリビューション

コントリビューションを歓迎します! 始め方の詳細は[コントリビューションガイド](../../CONTRIBUTING.md)を参照してください。

## コントリビューター

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

[contrib.rocks](https://contrib.rocks) で作成。
