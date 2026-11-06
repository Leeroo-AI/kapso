<!-- Korean translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">실험을 실행하고, 최고의 솔루션을 배포하며, 모든 실행에서 학습하는 엔진</h4>
<p align="center"><sub>스마트 실험 · 학습 루프 · 배포 · 도메인 적응</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <b>한국어</b> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">더 알아보기</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Discord 참여하기</a> ·
  <a href="https://leeroo.com">웹사이트</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  이 프로젝트가 마음에 드신다면 스타를 눌러 응원해 주세요 ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **이 저장소를 보고 있는 AI 에이전트라면:** 설정, 목표 명시, harness 작성, 실행 검토를 위한
> 스킬을 갖춘 Kapso 플러그인은
> 한 줄로 설치됩니다: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## 소식

- **IOAI 2026 · AI 모델 트랙.** [국제 인공지능 올림피아드](https://ioai-official.org)는 AI 시대의 IMO입니다. 참가자 471명이 108개 국가 및 지역에서 모여, 단일 GPU 예산 안에서 전문가가 설계한 여섯 개의 과제를 풉니다. 2026년에는 [IOAI²](https://ioai-official.org/ai-model-track/)가 열렸고, 여기서 AI 시스템은 사람이 지켜보지 않는 두 차례의 6시간 세션 동안 *동일한* 시험을 치릅니다. 시계가 돌아가기 시작하면 어떤 사람도 무엇 하나 풀거나, 고치거나, 개선할 수 없습니다. Kapso는 14개의 창립 AI 참가자 중 하나로 출전했습니다.
  - 🎓 **인간 참가자 471명 전원보다 높은 점수**: 총점 536.07로, 대회장의 모든 참가자를 앞섰습니다.
  - 🏆 **IOAI² Grand Master Trophy**: 출전한 모든 AI 시스템 가운데 상위 3위 안에 들었습니다. [전체 결과 →](../../benchmarks/ioai2026/README.md) · [IOAI의 트랙 리뷰 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **[RelBench](../../benchmarks/relbench/README.md) 리더보드 세 개 모두에서 선두**: 관계형 데이터에 대한 예측 ML을 다루는 Stanford의 벤치마크에서 Kapso는 공식 리더보드의 분류(81.2 AUROC, 12개 과제 기준, [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview)는 79.3, Stanford STAR의 RT-J는 79.0), 회귀(0.2476 NMAE, 9개 과제 기준, RT-J는 0.2711, KumoRFM-v2는 0.2847이며 오차가 8.7% 적음), 추천(18.4 MAP, 10개 과제 기준, 그다음으로 좋은 항목인 ID-GNN은 9.3) 부문을 모두 이끌고 있습니다. 모든 수치는 [공식 RelBench 리더보드](https://star-project.stanford.edu/relbench/leaderboard/)에 등재된 항목입니다.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)**: 데이터 및 ML 도메인에서 저희가 얻은 교훈을 담은 위키입니다. 논문과 저장소를 대상으로 실험하여 추출한 약 100k 페이지로 이루어져 있으며, 한 달에 약 6M건의 요청을 처리하고 그중 약 28%는 AI 에이전트에서 옵니다. Kapso는 MCP를 통해 여기에 연결됩니다. Kapso의 세션은 아이디어 구상과 구현 과정에서 이를 사용해 지식을 검색하고, 계획을 세우고, 실패를 진단합니다.
- **[Moltbook 에이전트 🦞](https://www.moltbook.com/)**: 다른 에이전트를 최적화하고 Moltbook에서 토론하는 AI 에이전트를 만들어 보세요! [시작하기 →](../../moltbook_bot/README.md)
- **기술 보고서**: 저희 기술 보고서가 공개되었습니다! [논문 읽기](https://arxiv.org/abs/2601.21526)
- **[MLE-Bench](../../benchmarks/mle/README.md) 1위**: KAPSO는 Kaggle ML 대회(MLE Benchmark)에서 오픈 소스 시스템 가운데 최고 순위를 달성했습니다.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **[ALE-Bench](../../benchmarks/ale/README.md) 1위**: KAPSO는 장기적인 알고리즘 발견 문제(ALE Benchmark)에서 최고 순위를 달성했습니다.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## Kapso란?

Kapso는 실험을 실행하고, 최고의 솔루션을 배포하며, 모든 실행에서 학습하는 엔진입니다. 목표, 초기 성공 기준, 평가 지표가 주어지면 Kapso는 조사를 수행하고, 가장 좋은 아이디어를 골라 실험을 실행하며, 때로는 평가 지표와 성공 기준을 개선하기 위한 피드백을 제공하고, 최고의 솔루션을 원하는 스택에 배포하며, 각 실행에서 학습합니다. Kapso는 교훈 뱅크를 유지하며, 각 교훈에는 신뢰도 점수와 그 교훈이 적용되는 범위를 나타내는 경계가 있어, 지식이 시간이 지나며 단순히 쌓이기만 하는 것이 아니라 복리로 불어납니다.

저희는 LinkedIn과 Meta AI를 비롯한 대기업 안에서 AI 시스템을 구축했고, 수백만 명의 사용자를 가진 제품에 이를 배포했습니다. Kapso는 그 워크플로를 엔진으로 만든 것입니다. [Leeroo](https://leeroo.com)가 만든 오픈 소스이며, PyPI에 [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/)로 게시되어 있습니다.

### 네 가지 구성 요소

| 구성 요소 | 동사 | 하는 일 |
|------|------|--------------|
| [**스마트 실험**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 실험 루프는 실험 트리라는 하나의 데이터 구조를 공유하는 전문 모듈들의 집합입니다. 각 노드는 가설, 구현(git 브랜치와 기술적 어려움), 사용한 예산, 목표까지의 거리를 기록합니다. 아이디어 구상은 이전 실험과 교훈 뱅크를 바탕으로 아이디어를 제안하며, 각 아이디어는 특정한 이전 실험과 어떻게 다른지, 어떤 교훈을 채택하는지를 밝힙니다. 리랭커는 목표치 대비 기대 수익을 시간과 예산에 견주어 아이디어의 순위를 매깁니다. 구현기는 부모 노드에서 분기한 브랜치 위에서 구축하고, 검증기는 노드를 안전 규칙과 성공 기준에 비추어 점검하여 점수가 유효한지, 목표가 달성되었는지를 판단합니다. 성공 기준과 테스트는 실행 중에 발전할 수 있습니다. 모듈이 근거를 갖춘 업데이트를 제안하면 검증기가 이를 승인하거나 거부합니다. [작동 방식 →](https://docs.leeroo.com/docs/evolve/overview) |
| [**학습 루프**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 각 교훈은 하나의 지식, 신뢰도 점수, 적용 범위를 나타내는 경계 조건, 그리고 업데이트를 뒷받침하는 증거를 기록합니다. 각 실행이 끝나면 `learn()`은 사용된 교훈이 결과에 미친 영향을 바탕으로 그 신뢰도와 경계를 업데이트하고, 유사한 트레이스에서 교훈을 귀납하거나 하나의 증거와 그 메커니즘에서 교훈을 연역하며, 겹치는 교훈을 병합하고, 업데이트된 교훈을 넣은 경우와 뺀 경우로 동일한 빌드를 실행해 A/B 테스트합니다. 따라서 업데이트는 추측되는 것이 아니라 측정됩니다. 다음 빌드의 아이디어 구상과 구현은 지식을 단순히 RAG로 가져오는 대신, 각 교훈의 신뢰도와 경계를 읽어 언제 그 교훈에 의존할지 결정합니다. [Experience →](https://docs.leeroo.com/docs/experience)는 빌드에 무엇이 제공되는지를 지정합니다. 여러분의 뱅크, 팀이 공유하는 뱅크, Leeroopedia, 또는 저장소나 조사한 질문에서 수집한 코퍼스입니다. |
| [**배포**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 실험 루프의 더 짧은 버전을 통해 트리의 최고 솔루션을 대상 스택에서 실행되는 소프트웨어로 바꿉니다. 세션이 노드의 사본을 조정하고, 배포한 뒤, 헬스 체크와 실제로 동작하는 호출을 시연합니다. Docker, Modal, BentoCloud, LangGraph Platform이 기본 제공되며, 여러분만의 대상 스택을 harness에 추가할 수 있습니다. |
| [**도메인 적응**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harness는 Kapso의 도메인 패키지입니다. 예측 모델링과 같은 영역을 위한 지식과 워크플로를 담고 있어, 빌드가 맨바닥에서 실험하고 학습하지 않아도 됩니다. 예측 모델링과 경쟁 프로그래밍 harness는 Kapso에 기본 포함되어 있습니다. 여러분만의 도메인 지식을 가져오려면 Kapso가 정의한 인터페이스를 통해 harness를 작성하세요. harness는 여러분의 지침, 코드, 가드레일, 검사로 이루어지며, 플러그인에서는 `/kapso:harness`로, CLI에서는 `kapso harness`로 작성합니다. |

## 🚀 빠른 시작

### 설치

**0. 코딩 에이전트 안에서 작업 중인가요?** Kapso 플러그인을 설치하고, 플러그인이
아래 단계를 함께 진행하도록 하세요:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

그런 다음 `/kapso:setup`이 설치와 검증을 수행하고, `/kapso:build <what you want>`가
첫 캠페인을 시작합니다(Codex에서는 `$kapso-setup`, `$kapso-build <what you want>`). [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills)을 참고하세요.

**1. 사전 요구 사항.** Kapso는 코딩 에이전트 CLI를 통해 추론을 실행하므로
(직접 API를 호출하는 대체 경로는 없습니다), 무엇이든 동작하려면 먼저 Node.js가 있어야 하고
두 에이전트 CLI 모두 로그인되어 있어야 합니다:

```bash
# Node.js 18+ (https://nodejs.org), 그다음:
npm install -g @openai/codex            # 리서치, 판정, 유틸리티
codex login

npm install -g @anthropic-ai/claude-code  # 아이디어 구상 + 구현(기본 호스트)
claude auth login
```

OpenAI 키는 선택 사항입니다. 지식 그래프와 과거 실험에 대한 시맨틱 검색
(설정의 `models.embedding`)만 텍스트를 임베딩하며,
생성에는 이 키가 전혀 쓰이지 않습니다. 둘 중 하나를 켤 때:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia 연결(권장).** 키 하나로 모든 캠페인
세션이 [Leeroopedia](https://leeroopedia.com)를 사용할 수 있습니다. Leeroopedia는
ML 및 AI 프레임워크에 대한 호스팅형 지식 베이스로, 프레임워크의 실제 동작 방식, 계획 리뷰,
문서화된 하이퍼파라미터 범위, 실패 진단을 제공합니다. 설치할 것은 없으며,
[Leeroopedia 대시보드](https://app.leeroopedia.com/dashboard)에서 새 계정을 만들면
$20의 무료 크레딧이 제공됩니다.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. 패키지 설치** (Python 3.10+):

```bash
pip install leeroo-kapso
```

패키지 이름은 `leeroo-kapso`입니다. PyPI에서 `kapso`라는 이름의 패키지는 이와 무관한
WhatsApp 도구로, 자체 `kapso` 명령을 설치하여 이 명령을 가립니다.
해당 패키지가 설치되어 있다면 먼저 `pip uninstall kapso`를 실행하세요.

**4. 설정 검증:**

```bash
kapso doctor
```

`doctor`는 **여러분의 설정**에 실제로 필요한 것을 보고하고, 빠진 것이 있으면 정확한
해결 방법을 알려 줍니다. 요구 사항은 설정을 따르므로, 전부 codex로 구성된
설정에서는 `claude`를 요구하지 않습니다. 동사 하나로 범위를 좁히면 그
동사의 요구 사항만 볼 수 있습니다:

```bash
kapso doctor build             # 또는 learn / deploy / experience
```

이는 동사 자체가 작업을 시작하기 전에 실행하는 검사와 동일합니다.
`kapso.build(...)`는 CLI가 없으면 세션 깊숙한 곳이 아니라 몇 초 안에
실패합니다. `[-- ]`로 표시된 항목은 선택 사항이며, 필요하지 않을 수도 있는
기능(docker, Weaviate, Neo4j, 배포 대상)을 제한합니다.

**또는 코딩 에이전트에게 이 모든 것을 맡기세요:**
[설정 프롬프트](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)를
Claude Code, Codex 또는 OpenCode에 붙여 넣으세요.

**인덱스에 필요한 스토어(선택 사항)**: `kapso experience add --repo`와
`--research`는 코퍼스를 로컬 Weaviate + Neo4j에 수집합니다. 소스
체크아웃에서:

```bash
bash scripts/start_infra.sh   # docker로 둘 다 시작
```

**소스에서 설치(개발용)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

레거시 aider 어댑터는 extra입니다(`pip install "leeroo-kapso[aider]"`,
Python <3.13). 기본 claude/codex 에이전트에는 extra가 필요하지 않습니다.

### 기본 사용법

핵심 루프에는 위의 사전 요구 사항 외에 아무것도 필요하지 않습니다:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 시작하는 데 지식 그래프는 필요하지 않습니다

# Build: 실험을 통해 솔루션을 만듭니다. 캠페인은 시작할 때
# `status: <path>`를 출력합니다. 다른 터미널에서 다음 명령으로 실시간으로 지켜보세요
#     kapso watch ./campaign
# 세션에 여러분만 제공할 수 있는 것(API 키, 파일)이 필요하면
# 캠페인이 일시 중지되고, `kapso inbox reply`가 이를 재개합니다:
#     https://docs.leeroo.com/docs/build/inbox
# 풀 리퀘스트나 이슈에 쓸 수 있도록 결과를 Markdown으로 공유하려면
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 또는 생략하여 처음부터 시작
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Learn: 방금 실행한 캠페인에서 학습합니다. 궤적을 채굴하고, 교훈을
# 채점하고, 증거에 따라 값이 매겨진 지식 카드를 뱅크에 넣습니다. 뱅크(로컬
# git 저장소)는 처음 사용할 때 자동으로 생성됩니다. 교훈은 공유하기 전까지
# 여러분의 머신에 남아 있습니다:
#     kapso experience share <git-url>
# 그 이후에는 learn()을 실행할 때마다 뱅크를 그곳으로 푸시합니다.
lesson = kapso.learn(solution)
print(lesson.explain())

# 다시 Build: 방금 얻은 카드를 제공받습니다. 빌드는 자신이 로드한
# 경험을 정확히 그대로 제공받습니다.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

스토어가 실행 중이면 외부 지식을 인덱스로 수집하여
같은 방식으로 제공할 수도 있습니다:

```python
from kapso import Kapso

kapso = Kapso()

# 한 번만: 저장소, 또는 질문에 대해 조사한 웹을 인덱스로 수집합니다
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# 인덱스를 로드하는 모든 빌드가 그 인덱스를 검색합니다. Leeroopedia는 자체 키로 동작합니다
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

예산에 관한 참고 사항: 수집은 자료가 뒷받침하는 모든 것을 추출합니다.
질문 하나가 수십 개의 연결된 페이지와 몇 시간짜리
수집이 될 수 있습니다. 수집 시간은 추출 가능한 내용의 양에 비례하므로,
가진 소스를 전부 넣지 말고 검색할 소스만 추가하세요.

그리고 솔루션을 실행되는 소프트웨어로 바꾸려면:

```python
software = kapso.deploy(solution, target="local")   # 또는 docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # 배포가 증명한 호출
software.stop()
```

### 캠페인의 한도는 어떻게 정하나요?

`max_iterations`, `time_budget_minutes`, `cost_budget`을 사용해 반복 횟수, 경과 시간, 그리고 최선의 노력으로 추정한 USD 기준 모델 지출의 한도를 정합니다.
플랫폼 비용의 경우, `search_strategy.params.cost`에서 여러분의 추정기, `cap_per_candidate`, `campaign_budget`과 함께 [비용 인식 선택](https://docs.leeroo.com/docs/evolve/cost-aware-selection)을 활성화하세요. 그 `weight`를 사용하면 선택 과정에서 더 저렴한 후보를 선호하게 할 수도 있습니다.

### 모델 선택

Kapso가 사용하는 모든 모델은 하나의 설정 파일에 명시됩니다. 패키지에 포함된 기본값은
빌드 세션을 `claude-opus-5`로, 학습 크루를
`claude-fable-5`로, codex 역할을 `gpt-5.6-sol`로 실행합니다. 다만 모델 접근 권한은
구독에 따라 달라집니다(요금제가 한 모델은 제공하면서 다른 모델에는 한도를 둘 수 있습니다).
다른 모델로 실행하려면 패키지에 포함된 설정을 복사해 편집한 다음,
Kapso가 여러분의 설정을 가리키도록 하세요:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 예: 학습 크루를 fable 대신 opus로 실행:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

텍스트 전용 추론 역할은 `openai_compatible` Chat Completions를 선택적으로 사용할 수 있습니다.
모델과 엔드포인트는 설정에 지정하고, 자격 증명은 환경에 그대로 둡니다.
`Kapso(config=...)` 또는 `kapso build --config ...`로 설정을 로드하세요.
코딩과 웹 리서치에는 여전히 해당 도구를 갖춘 에이전트가 필요합니다.
[설정 예시와 제한 사항](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)을 참고하세요.

긴 실행에 앞서, 설정에 명시된 모든 모델을 여러분의 실제 구독에 대해
프로브하세요. 여러분의 로그인으로 제공받을 수 없는 모델은 실행 후 몇 시간이 지나서가 아니라
여기서 몇 초 만에 실패합니다. 제공받을 수 있는 모델에 걸린 사용량 한도는
토큰 하나짜리 프로브로는 보이지 않습니다:

```bash
kapso doctor --models                            # 패키지에 포함된 설정
kapso doctor --models --config kapso-config.yaml # 여러분의 설정
kapso doctor learn --models                      # 학습 크루만
```

이 프로브를 기억해서 실행해야 하는 일이 아니라 모든 호출의 일부로 만들려면
설정에서 켜세요. 서로 다른 `{cli, model}` 쌍마다 버리는 토큰 하나가 들며,
사람이 지켜보지 않는 여러 시간짜리 `learn()` 전이라면
그만한 가치가 있습니다:

```yaml
preflight:
  enabled: true            # 모든 동사 실행 전 정적 검사(기본값)
  live_model_probe: true   # + 토큰 하나짜리 모델 프로브(기본값: false)
```

모델을 바꾸면 *속도*도 달라집니다. 설정의 크루 `timeout_minutes` 상한은
기본 모델을 기준으로 보정되었으므로, 더 오래 추론하는 모델로 바꾸면
상한을 올려야 할 수 있습니다.

자세한 통합 단계는 [빠른 시작](https://docs.leeroo.com/docs/first-build) 및 [설치](https://docs.leeroo.com/docs/installation) 가이드를 참고하세요.

## 예제

모든 예제는 하나의 과제입니다. 브리프, 시작 저장소, 데이터 또는 데이터를 얻는 방법, 그리고 최대 열두 줄짜리 드라이버로 이루어집니다. [examples/README.md](../../examples/README.md)에는 소요 시간(분), 실행할 명령 하나, 결과가 기록되어 있는지 여부가 나와 있습니다.

| 예제 | 보여 주는 것 |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | 첫 번째 빌드: 이미 있는 평가기, 베이스라인, 목표치, 기록 |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | 평가기 없음: 캠페인의 첫 노드가 브리프의 프로토콜로부터 평가기를 만들고, 모든 후보를 그에 비추어 측정합니다 |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | 내보낸 결과물에 결함이 있는 DuckDB 기반 dbt 파이프라인: 데이터 계약이 지켜지도록 만듭니다 |
| [**trading**](../../examples/trading/README.md) | 빌드에 적용한 harness: 설정 세션, 모든 세션의 페이지와 훅, 시도별 검증 |
| [**full loop**](../../examples/full-loop/README.md) | build, 리뷰, learn, 교훈을 제공받아 다시 build, deploy: Kaggle 테이블형 모델을 처음부터 끝까지 |
| [**prompt harness**](../../examples/prompt-harness/README.md) | 프롬프트 자체가 테스트 대상이며, AIME 문제로 채점합니다 |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | 비전-언어 추출기, 정답 테이블과 비교해 채점합니다 |
| [**torch performance**](../../examples/torch-performance/README.md) | 수치 허용 오차 안에서 CUDA에서 더 빠르게 동작하는 모듈 |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | CUDA에서의 멀티 헤드 어텐션: flash attention 또는 Triton |

## 지원 벤치마크

| 벤치마크 | 설명 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI의 ML 엔지니어링 벤치마크: 테이블형, 비전, 텍스트, 오디오 전반에 걸친 완전한 대회를 원시 데이터부터 채점되는 제출물까지 다룹니다 |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI의 알고리즘 최적화 벤치마크: 몇 시간에 걸친 탐색으로 대회용 휴리스틱을 설계하고, 구현하고, 반복 개선합니다 |
| [**RelBench**](../../benchmarks/relbench/README.md) | 관계형 데이터에 대한 예측 ML을 다루는 Stanford의 벤치마크: Amazon, H&M, Stack Exchange, 임상 시험 등의 다중 테이블 데이터베이스에서 곧바로 분류, 회귀, 추천을 수행합니다 |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 비전, 언어, 최적화에 걸친 시간 제한 올림피아드 ML: 전문가가 출제한 과제, 대회용 하드웨어, 사람의 도움 없음 |

각 벤치마크에는 실행 방법, CLI 옵션, 출력에 담기는 내용을 다루는 문서 페이지도 있습니다:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 문서 및 지원

- **전체 문서**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [설치](https://docs.leeroo.com/docs/installation): 코딩 에이전트 CLI, 패키지, 그리고 `kapso doctor`
  - [빠른 시작](https://docs.leeroo.com/docs/first-build): 여러분의 첫 캠페인
  - [CLI 레퍼런스](https://docs.leeroo.com/docs/reference/cli): 모든 명령, 플래그, 기본값
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso`의 모든 공개 메서드
  - [설정](https://docs.leeroo.com/docs/reference/configuration): `config.yaml`의 모든 키
  - [learn이 만들어 내는 것](https://docs.leeroo.com/docs/learn/what-learn-produces): 각 교훈마다 신뢰도 점수, 경계, 증거를 갖춘 교훈 뱅크, 그리고 [개발 모드](https://docs.leeroo.com/docs/learn/development-mode)
  - [실험 루프](https://docs.leeroo.com/docs/evolve/overview): 실험 트리와 그 모듈, 그리고 [하나의 실험 안에서 일어나는 일](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience): 여러분의 뱅크, Leeroopedia, 검색 가능한 지식으로 바뀐 저장소와 논문
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal 및 그 밖의 대상
  - [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code, Codex, Cursor를 위한 명령, 스킬, MCP 서버
- **코딩 에이전트에서**: 문서 사이트는 `https://docs.leeroo.com/mcp`에서 MCP 서버를 운영하며, 모든 페이지가 Markdown으로 제공됩니다. [llms.txt](https://docs.leeroo.com/llms.txt)와 [코딩 에이전트에서 Kapso 사용하기](https://docs.leeroo.com/docs/coding-agent-skills)를 참고하세요.
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **코딩 에이전트가 Kapso를 실행하게 하기**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 아래의 플러그인에는 열 개의 명령, 여섯 개의 스킬, MCP 서버가 들어 있으며, 호스트마다 설치 명령 한 줄이면 됩니다(위의 0단계). OpenCode는 스킬을 폴더로 로드합니다(`ln -s <clone>/plugin/skills .opencode/skills`). [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills)을 참고하세요.
- **코딩 에이전트에 Kapso의 지식 도구 제공하기**: 패키지와 함께 설치되는 `kapso-mcp`는 웹 리서치, 저장소의 메모리, 캠페인의 실험 이력, 교훈 뱅크를 여러분이 작업 중인 세션에 제공합니다. [MCP 게이트](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)를 참고하세요.
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; 그런 다음 ~/.codex/config.toml의 [mcp_servers.kapso] 아래에 `startup_timeout_sec = 60`을 넣으세요(Codex의 십 초는 바쁜 머신에서는 짧습니다)
  ```
- **커뮤니티**: [Discord](https://discord.gg/hqVbPNNEZM)
- **웹사이트**: [leeroo.com](https://leeroo.com)


## 엔터프라이즈용 Kapso

저희 엔터프라이즈 제품은 이 엔진 위에 구축되었으며, 어떤 사람도 데이터 자산 전체를 조사하고 이해할 수 없는 복잡하게 얽힌 기업 환경에 배포되어 왔습니다. 저희는 한 회사를 기존 데이터 플랫폼에서 새로운 레이크하우스로 이전했는데, 그 범위는 30개가 넘는 소스 시스템 계열, 수천 개의 테이블과 변환, 수백 개의 BI 보고서에 걸쳐 있었습니다. 제품이 수행하는 모든 작업은 여러분의 시스템, 데이터, 어렵게 얻은 규칙에 관한 교훈 뱅크에 반영되므로, 오래 일할수록 여러분의 회사에 더 능숙해집니다. 까다로운 엔터프라이즈 과제에 Kapso를 도입하려면 [저희에게 문의하세요](https://leeroo.com/contact-us).

## 번역

README는 다른 언어로도 제공됩니다. 상단의 전환기를 사용하거나 [`docs/translations/`](../../docs/translations/)를 둘러보세요. 각 번역은 `scripts/translate_readme.py`로 README.md에서 생성되며, `scripts/check_readme_translations.py`가 번역을 원문과 일치하게 유지합니다. 이 스크립트는 번역이 링크, 숫자, 코드 한 줄에서 README.md와 어긋나면 실패합니다. 번역을 개선하려면 해당 폴더의 파일에 대해 풀 리퀘스트를 여세요.

## 기여하기

기여를 환영합니다! 시작하는 방법에 대한 자세한 내용은 [기여 가이드](../../CONTRIBUTING.md)를 참고하세요.

## 인용

연구나 작업에 Kapso를 사용하신다면 기술 보고서를 인용해 주세요:

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
