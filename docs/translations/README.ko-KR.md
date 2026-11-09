<!-- Korean translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">실험을 실행하고, 최선의 솔루션을 배포하며, 모든 실행에서 학습하는 엔진</h4>
<p align="center"><sub>스마트 실험 · 학습 · 배포 · 도메인 적응</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <b>한국어</b> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">더 알아보기</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Discord 참여</a> ·
  <a href="https://leeroo.com">웹사이트</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  이 프로젝트가 마음에 드신다면 스타 ⭐ 를 눌러 응원해 주세요
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **이 저장소를 보고 있는 AI 에이전트라면:** 설정, 목표 진술, harness 작성,
> 실행 검토를 위한 스킬이 담긴 Kapso 플러그인은
> 한 줄로 설치됩니다: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## 뉴스

- **IOAI 2026 · AI 모델 트랙.** [국제 AI 올림피아드](https://ioai-official.org)는 AI 시대의 IMO입니다. 108개 국가 및 지역에서 온 471명의 참가자가 단일 GPU 예산 아래 전문가가 설계한 여섯 개의 과제를 풉니다. 2026년에는 [IOAI²](https://ioai-official.org/ai-model-track/)를 열어, AI 시스템이 두 번의 무인 6시간 세션에서 *같은* 시험을 치르게 했습니다. 시계가 시작되면 어떤 사람도 무엇 하나 풀거나 고치거나 개선할 수 없습니다. Kapso는 14개 창립 AI 참가자 중 하나로 출전했습니다.
  - 🎓 **471명의 인간 참가자 전원을 앞섰습니다**: 총점 536.07로, 경기장의 모든 참가자보다 높았습니다.
  - 🏆 **IOAI² 그랜드 마스터 트로피**: 출전한 모든 AI 시스템 중 상위 3위 이내. [전체 결과 →](../../benchmarks/ioai2026/README.md) · [트랙에 대한 IOAI의 리뷰 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **[RelBench](../../benchmarks/relbench/README.md)의 세 리더보드 모두에서 선두**: 관계형 데이터에 대한 예측 ML을 위한 Stanford의 벤치마크에서 Kapso는 공식 리더보드의 분류(12개 과제에서 81.2 AUROC, [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview)의 79.3과 Stanford STAR의 [RT-J](https://star-project.stanford.edu/rt-j)의 79.0 대비), 회귀(9개 과제에서 0.2476 NMAE, RT-J의 0.2711과 KumoRFM-v2의 0.2847 대비, 오차 8.7% 감소), 추천(10개 과제에서 18.4 MAP, 그 외 최고 항목인 ID-GNN의 9.3 대비) 부문을 이끌고 있습니다. 모든 숫자는 [공식 RelBench 리더보드](https://star-project.stanford.edu/relbench/leaderboard/)의 항목입니다. harness에 보관된 학습이 모델 가중치에 구워 넣은 학습을 이겼고, 교훈 뱅크를 업데이트하는 비용은 표 형식 파운데이션 모델을 재학습하는 것보다 훨씬 적게 듭니다.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)**: 데이터 및 ML 도메인에서 얻은 우리 교훈의 위키로, 논문과 저장소를 대상으로 실험하여 추출한 약 100k 페이지가 있으며, 월 약 6M 요청을 처리하고 그중 약 28%는 AI 에이전트에서 옵니다. Kapso는 MCP를 통해 여기에 연결됩니다: 세션들은 아이디어 발상과 구현 중에 이를 사용해 지식을 검색하고, 계획을 세우고, 실패를 진단합니다.
- **[MLE-Bench](../../benchmarks/mle/README.md) 1위**: KAPSO는 Kaggle ML 대회(MLE 벤치마크)에서 오픈 소스 시스템 중 최상위 순위를 달성했습니다.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **[ALE-Bench](../../benchmarks/ale/README.md) 1위**: KAPSO는 장기 알고리즘 발견 문제(ALE 벤치마크)에서 최상위 순위를 달성했습니다.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## Kapso란 무엇인가요?

회사 안에서 AI 또는 데이터 성과를 내려면 두 가지를 실험으로 배워야 합니다. 데이터 자산이 어떻게 연결되고 운영되는지는 어떤 개인이나 AI 에이전트도 알지 못합니다. 이를 배우는 유일한 방법은 회사의 산출물(저장소, 파이프라인, 문서, 데이터 소스)을 조사하고, 가설을 제안하고, 실험을 실행하고 그로부터 배우기를, 각 시스템의 동작과 그 조건이 이해될 때까지 반복하는 것입니다. 그리고 그런 이해가 생긴 뒤에도, 웨어하우스 최적화든 이탈 예측이든 최선의 솔루션을 미리 고를 수는 없으므로, 환경에 맞는 것이 나올 때까지 실험하고 학습합니다. 환경에 맞는 노하우는 RL에서처럼 축적됩니다. RAG는 그 반대입니다: 이 환경에서 한 번도 검증되지 않은 일반 지식을 모두 같은 비중으로 에이전트에 공급하는데, 그 지식은 거기서 시도되기 전까지는 가치가 거의 없습니다.

Kapso는 우리가 이를 위해 만든 엔진입니다. 도메인 harness와 빈 교훈 뱅크에서 시작하여, 목표와 성공 기준이 주어지면 실험 트리를 실행하며 가설을 제안한 뒤 각각을 구현하고 검증하고, 최선의 솔루션을 대상 스택에 배포합니다. 그런 다음 실행에서 학습하여, 각 가설과 구현에 결과에 미친 영향에 따라 기여도를 부여하고 뱅크를 업데이트합니다. 뱅크의 모든 교훈은 신뢰도 점수와 적용 범위에 대한 경계를 지닙니다. 시간이 지나면 도메인 harness는 그 환경에서 실험으로 검증된 모범 사례를 축적한 교훈 뱅크를 지니게 됩니다. 이 시스템은 모델에 구애받지 않고 코딩 에이전트에도 구애받지 않습니다: [Leeroo](https://leeroo.com)가 만든 오픈 소스이며, PyPI에 [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/)로 게시되어 있습니다.

### 네 가지 구성 요소

| 구성 요소 | 동사 | 하는 일 |
|------|------|--------------|
| [**스마트 실험**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 특화된 모듈들이 하나의 데이터 구조, 즉 실험 트리를 공유합니다. 각 노드는 가설, 구현(git 브랜치와 기술적 난점), 사용한 예산, 목표까지의 거리를 기록합니다. 각 실험마다 아이디어 발상 모듈이 이전 실험과 교훈 뱅크에서 아이디어를 제안하고, 각 아이디어가 사용할 예산 대비 기대 수익으로 재순위를 매긴 뒤, 상위 아이디어를 구현 모듈로 보내며, 구현 모듈은 부모 노드에서 포크한 브랜치 위에 구축합니다. 뱅크가 바로 그 환경에서 무엇이 통하는지를 계속 학습하기 때문에, 아이디어는 순수 LLM이나 RAG가 내놓는 일반적인 것이 아니라 그 환경에 특화된 것이 됩니다. 결과는 어떤 실험도 접근할 수 없는 핑거프린트된 평가기가 검증하며, 평가 역시 진화합니다: 실험 모듈들이 검증기의 승인을 받도록 지표와 기준에 대한 업데이트를 제안할 수 있습니다. [작동 방식 →](https://docs.leeroo.com/docs/evolve/overview) |
| [**학습**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 뱅크의 모든 것은 실험에서 나옵니다. 각 교훈은 실험 실행의 추적에서 귀납 또는 연역되며, 신뢰도 점수, 적용 범위에 대한 경계 조건, 그 뒤를 받치는 증거를 지니고, 매 실행 후 `learn()`이 최종 결과에 대한 그 교훈의 효과를 바탕으로 이를 업데이트합니다. 뱅크의 변경도 실험입니다: 업데이트된 모든 교훈은 같은 빌드를 그 교훈이 있을 때와 없을 때로 실행하여 A/B 테스트되므로, 개선은 추측이 아니라 측정됩니다. 뱅크는 git 위에 구축되어 있어 다른 사람과 경험을 공유하고(`kapso experience share`), 사용할 수 있는 뱅크를 나열하고(`kapso experience list`), 업데이트할 수 있습니다(`kapso experience update`). [경험 →](https://docs.leeroo.com/docs/experience)은 빌드에 제공되는 것, 즉 내 뱅크, 팀이 공유하는 뱅크, Leeroopedia, 또는 저장소나 조사한 질문에서 수집한 코퍼스를 가리킵니다. |
| [**배포**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 최선의 솔루션을 대상 스택에 맞추기 위해 같은 종류의 더 짧은 실험을 실행합니다: 세션이 노드의 사본을 적응시키고, 배포하고, 상태 확인과 동작하는 호출을 시연합니다. Docker, Modal, BentoCloud, LangGraph Platform이 기본 제공되며, harness에 직접 대상 스택을 추가할 수 있습니다. |
| [**도메인 적응**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harness는 Kapso의 도메인 패키지입니다. 예측 모델링 같은 영역의 지식과 워크플로를 담아, 빌드가 처음부터 실험하고 학습하지 않아도 되게 합니다. 예측 모델링과 경쟁 프로그래밍이 Kapso와 함께 제공됩니다. 실행은 도메인 harness와 빈 교훈 뱅크에서 시작하며, 시간이 지나면 harness는 그 환경에서 통한 것을 축적한 뱅크를 지니게 됩니다. 초기 도메인 harness, 즉 지침, 코드, 가드레일, 검사를 플러그인의 `/kapso:harness` 또는 CLI의 `kapso harness`로 작성하세요. |

## 🚀 빠른 시작

### 설치

**0. 코딩 에이전트 안에 계신가요?** Kapso 플러그인을 설치하고 아래 단계를
함께 진행하도록 하세요:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

그런 다음 `/kapso:setup`이 설치하고 검증하며, `/kapso:build <what you want>`가
첫 캠페인을 시작합니다(Codex에서는 `$kapso-setup`, `$kapso-build <what you want>`). [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills)을 참고하세요.

**1. 사전 요구 사항.** Kapso는 코딩 에이전트 CLI를 통해 추론을 실행하므로
(직접 API 폴백은 없습니다), 무엇이든 동작하려면 먼저 Node.js와 두 에이전트
CLI 모두 로그인되어 있어야 합니다:

```bash
# Node.js 18+ (https://nodejs.org), 그런 다음:
npm install -g @openai/codex            # 조사, 판정, 유틸리티
codex login

npm install -g @anthropic-ai/claude-code  # 아이디어 발상 + 구현(기본 호스트)
claude auth login
```

OpenAI 키는 선택 사항입니다: 지식 그래프와 과거 실험에 대한 의미 검색
(설정의 `models.embedding`)만 텍스트를 임베딩하며, 생성에는 절대 사용되지
않습니다. 둘 중 하나를 켤 때:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia 연결(권장).** 키 하나로 모든 캠페인 세션에
ML 및 AI 프레임워크의 호스팅 지식 베이스인 [Leeroopedia](https://leeroopedia.com)를
제공합니다: 프레임워크가 실제로 어떻게 동작하는지, 계획 리뷰,
문서화된 하이퍼파라미터 범위, 실패 진단. 설치할 것은 없습니다.
[Leeroopedia 대시보드](https://app.leeroopedia.com/dashboard)에서 새 계정을 만들면
$20의 무료 크레딧이 제공됩니다.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. 패키지 설치**(Python 3.10+):

```bash
pip install leeroo-kapso
```

패키지 이름은 `leeroo-kapso`입니다. PyPI의 `kapso`라는 패키지는 관련 없는
WhatsApp 도구로, 자체 `kapso` 명령을 설치하여 이 명령을 가립니다.
설치되어 있다면 먼저 `pip uninstall kapso`를 실행하세요.

**4. 설정 검증:**

```bash
kapso doctor
```

`doctor`는 **여러분의 설정**이 실제로 필요로 하는 것을 보고하고, 빠진 것이
있으면 정확한 해결 방법을 알려 줍니다. 요구 사항은 설정을 따르므로, codex만
쓰는 설정에서는 `claude`를 요구하지 않습니다. 하나의 동사로 좁히면 그 동사의
요구 사항만 볼 수 있습니다:

```bash
kapso doctor build             # 또는 learn / deploy / experience
```

이것은 동사 자체가 작업을 시작하기 전에 실행하는 것과 같은 검사입니다.
`kapso.build(...)`는 CLI가 빠져 있으면 세션 깊숙이 들어간 뒤가 아니라
몇 초 안에 실패합니다. `[-- ]`로 표시된 항목은 선택 사항입니다. 필요하지
않을 수도 있는 기능(docker, Weaviate, Neo4j, 배포 대상)만 제한합니다.

**또는 코딩 에이전트가 이 모든 것을 하게 하세요:**
[설정 프롬프트](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)를
Claude Code, Codex 또는 OpenCode에 붙여 넣으세요.

**인덱스에 필요한 스토어(선택 사항)**: `kapso experience add --repo`와
`--research`는 코퍼스를 로컬 Weaviate + Neo4j에 수집합니다.
소스 체크아웃에서:

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

레거시 aider 어댑터는 엑스트라입니다(`pip install "leeroo-kapso[aider]"`,
Python <3.13). 기본 claude/codex 에이전트에는 엑스트라가 필요 없습니다.

### 기본 사용법

핵심 동사들은 위의 사전 요구 사항 외에 아무것도 필요로 하지 않습니다:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 시작하는 데 지식 그래프는 필요 없음

# Build: 실험을 통한 솔루션. 캠페인은 시작 시
# `status: <path>`를 출력합니다. 다른 터미널에서 실시간으로 지켜보려면
#     kapso watch ./campaign
# 세션에 여러분만 제공할 수 있는 것(API 키, 파일)이 필요하면
# 캠페인이 일시 정지되고 `kapso inbox reply`가 재개합니다:
#     https://docs.leeroo.com/docs/build/inbox
# 결과를 풀 리퀘스트나 이슈용 Markdown으로 공유하려면
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 또는 생략하여 처음부터 시작
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 방금 실행한 캠페인에서 학습: 궤적을 채굴하고, 교훈을
# 채점하고, 증거로 가격이 매겨진 지식 카드를 뱅크에 저장합니다. 뱅크(로컬
# git 저장소)는 처음 사용할 때 자동으로 생성됩니다. 교훈은 공유하기 전까지
# 여러분의 머신에 남아 있습니다:
#     kapso experience share <git-url>
# 그 후에는 모든 learn()이 뱅크를 거기로 푸시합니다.
lesson = kapso.learn(solution)
print(lesson.explain())

# 방금 얻은 카드를 제공받아 다시 빌드: 빌드에는 정확히
# 로드한 경험이 제공됩니다.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

스토어가 실행 중이면 외부 지식을 인덱스에 수집하여 같은 방식으로
제공할 수도 있습니다:

```python
from kapso import Kapso

kapso = Kapso()

# 한 번: 저장소, 또는 질문에 대해 조사한 웹을 인덱스로
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# 인덱스를 로드한 모든 빌드가 이를 검색합니다. Leeroopedia는 키만 있으면 따라옵니다
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

예산에 관한 참고: 수집은 자료가 뒷받침하는 모든 것을 추출합니다.
질문 하나가 수십 개의 연결된 페이지와 몇 시간짜리 수집이 될 수
있습니다. 수집 시간은 추출 가능한 내용의 양에 비례하므로, 가진 소스
전부가 아니라 검색할 소스만 추가하세요.

그리고 솔루션을 실행 중인 소프트웨어로 바꾸려면:

```python
software = kapso.deploy(solution, target="local")   # 또는 docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # 배포가 입증한 호출
software.stop()
```

### 캠페인을 어떻게 제한하나요?

`max_iterations`, `time_budget_minutes`, `cost_budget`를 사용해 반복 횟수, 경과 시간, USD 기준 최선 노력 모델 지출을 제한하세요.
플랫폼 비용에 대해서는 `search_strategy.params.cost`에서 추정기, `cap_per_candidate`, `campaign_budget`과 함께 [비용 인식 선택](https://docs.leeroo.com/docs/evolve/cost-aware-selection)을 활성화하세요. 그 `weight`로 선택이 더 저렴한 후보를 선호하게 할 수도 있습니다.

### 모델 선택

Kapso가 사용하는 모든 모델은 하나의 설정 파일에 명시됩니다. 패키지 기본값은
빌드 세션을 `claude-opus-5`에서, 학습 크루를 `claude-fable-5`에서,
codex 역할을 `gpt-5.6-sol`에서 실행합니다. 하지만 모델 접근은 구독에 따라
다릅니다(어떤 플랜은 한 모델을 제한하면서 다른 모델은 제공할 수 있습니다).
다른 모델로 실행하려면 패키지 설정을 복사하고, 편집한 뒤, Kapso가 여러분의
설정을 가리키게 하세요:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# 예: 학습 크루를 fable 대신 opus에서 실행:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

텍스트 전용 추론 역할은 `openai_compatible` Chat Completions를 선택적으로 사용할 수 있습니다.
설정에서 모델과 엔드포인트를 지정하세요. 자격 증명은 환경에 남겨 둡니다.
설정은 `Kapso(config=...)` 또는 `kapso build --config ...`로 로드합니다.
코딩과 웹 조사에는 여전히 해당 도구를 갖춘 에이전트가 필요합니다.
[설정 예시와 제한 사항](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)을 참고하세요.

긴 실행 전에, 설정에 명시된 모든 모델을 실제 구독에 대해 점검하세요.
로그인으로 제공받을 수 없는 모델은 실행 몇 시간 뒤가 아니라 여기서
몇 초 안에 실패합니다. 제공받을 수 있는 모델의 사용량 한도는 단일 토큰
프로브로는 보이지 않습니다:

```bash
kapso doctor --models                            # 패키지 설정
kapso doctor --models --config kapso-config.yaml # 여러분의 설정
kapso doctor learn --models                      # 학습 크루만
```

그 프로브를 기억해서 실행하는 것이 아니라 모든 호출의 일부로 만들려면
설정에서 켜세요. 고유한 `{cli, model}` 쌍마다 일회용 토큰 하나가 들지만,
무인으로 여러 시간 돌아가는 `learn()` 전에는 그만한 가치가 있습니다:

```yaml
preflight:
  enabled: true            # 모든 동사 전에 정적 검사(기본값)
  live_model_probe: true   # + 단일 토큰 모델 프로브(기본값: false)
```

모델 교체는 *속도*도 바꿉니다: 설정의 크루 `timeout_minutes` 상한은
기본 모델 기준으로 보정된 것이므로, 더 오래 추론하는 모델로 바꾸면
올려야 할 수 있습니다.

자세한 통합 단계는 [빠른 시작](https://docs.leeroo.com/docs/first-build)과 [설치](https://docs.leeroo.com/docs/installation) 가이드를 참고하세요.

## 지원 벤치마크

| 벤치마크 | 설명 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI의 ML 엔지니어링 벤치마크. 표 형식, 비전, 텍스트, 오디오에 걸친 전체 대회를 원시 데이터부터 채점된 제출물까지 수행합니다 |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI의 알고리즘 최적화 벤치마크. 몇 시간에 걸친 탐색 동안 대회 휴리스틱을 설계하고, 구현하고, 반복합니다 |
| [**RelBench**](../../benchmarks/relbench/README.md) | 관계형 데이터에 대한 예측 ML을 위한 Stanford의 벤치마크. Amazon, H&M, Stack Exchange, 임상 시험 등의 다중 테이블 데이터베이스에서 바로 분류, 회귀, 추천을 수행합니다 |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 비전, 언어, 최적화에 걸친 시간제한 올림피아드 ML. 전문가가 출제한 과제, 대회 하드웨어, 인간의 도움 없음 |

각 벤치마크에는 실행 방법, CLI 옵션, 출력에 담긴 내용을 다루는 문서 페이지도 있습니다:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 예제

모든 예제는 하나의 과제입니다: 브리프, 시작 저장소, 데이터 또는 데이터를 얻는 방법, 그리고 최대 열두 줄의 드라이버. [examples/README.md](../../examples/README.md)에 소요 시간(분), 실행 명령 하나, 결과 기록 여부가 있습니다.

| 예제 | 보여 주는 것 |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | 첫 번째 빌드: 평가기가 있는 상태, 베이스라인, 목표치, 기록 |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | 평가기 없음: 캠페인의 첫 번째 노드가 브리프의 프로토콜로부터 평가기를 만들고 모든 후보가 그것에 대해 측정됩니다 |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | 내보내기에 결함이 있는 DuckDB 위의 dbt 파이프라인: 데이터 계약이 지켜지도록 만듭니다 |
| [**trading**](../../examples/trading/README.md) | 빌드에 적용된 harness: 설정 세션, 모든 세션의 페이지와 훅, 시도마다 검증 |
| [**end to end**](../../examples/end-to-end/README.md) | 빌드, 리뷰, 학습, 교훈을 제공받은 재빌드, 배포: Kaggle 표 형식 모델을 처음부터 끝까지 |
| [**prompt harness**](../../examples/prompt-harness/README.md) | 테스트 대상으로서의 프롬프트, AIME 문제로 채점 |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | 비전-언어 추출기, 정답 테이블에 대해 채점 |
| [**torch performance**](../../examples/torch-performance/README.md) | 수치 허용 오차 아래에서 CUDA상의 더 빠른 모듈 |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | CUDA상의 멀티 헤드 어텐션: 플래시 어텐션 또는 Triton |

## 📚 문서 및 지원

- **전체 문서**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [설치](https://docs.leeroo.com/docs/installation): 코딩 에이전트 CLI, 패키지, `kapso doctor`
  - [빠른 시작](https://docs.leeroo.com/docs/first-build): 첫 캠페인
  - [CLI 참조](https://docs.leeroo.com/docs/reference/cli): 모든 명령, 플래그, 기본값
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso`의 모든 공개 메서드
  - [설정](https://docs.leeroo.com/docs/reference/configuration): `config.yaml`의 모든 키
  - [learn이 만들어 내는 것](https://docs.leeroo.com/docs/learn/what-learn-produces): 각각 신뢰도 점수, 경계, 증거를 지닌 교훈 뱅크, 그리고 [개발 모드](https://docs.leeroo.com/docs/learn/development-mode)
  - [스마트 실험](https://docs.leeroo.com/docs/evolve/overview): 실험 트리와 그 모듈들, 그리고 [하나의 실험 안에서 일어나는 일](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [경험](https://docs.leeroo.com/docs/experience): 내 뱅크, Leeroopedia, 검색 가능한 지식으로 변환된 저장소와 논문
  - [배포](https://docs.leeroo.com/docs/deploy/targets): 로컬, Docker, Modal 및 기타 대상
  - [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code, Codex, Cursor용 명령, 스킬, MCP 서버
- **코딩 에이전트에서**: 문서는 `https://docs.leeroo.com/mcp`에서 MCP 서버를 실행하며, 모든 페이지가 Markdown으로 제공됩니다. [llms.txt](https://docs.leeroo.com/llms.txt)와 [코딩 에이전트 안의 Kapso](https://docs.leeroo.com/docs/coding-agent-skills)를 참고하세요.
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **코딩 에이전트가 Kapso를 실행하게 하기**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 아래의 플러그인에는 열 개의 명령, 여섯 개의 스킬, MCP 서버가 들어 있으며, 호스트마다 설치 한 줄이면 됩니다(위의 0단계). OpenCode는 스킬을 폴더로 로드합니다(`ln -s <clone>/plugin/skills .opencode/skills`). [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills)을 참고하세요.
- **코딩 에이전트에 Kapso의 지식 도구 제공**: 패키지와 함께 설치되는 `kapso-mcp`는 웹 조사, 저장소의 메모리, 캠페인의 실험 이력, 교훈 뱅크를 여러분이 작업하는 세션에 제공합니다. [MCP 게이트](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)를 참고하세요.
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; 그런 다음 ~/.codex/config.toml의 [mcp_servers.kapso] 아래에 `startup_timeout_sec = 60`(Codex의 열 초는 바쁜 머신에서는 짧습니다)
  ```
- **커뮤니티**: [Discord](https://discord.gg/hqVbPNNEZM)
- **웹사이트**: [leeroo.com](https://leeroo.com)


## 기업용 Kapso

기업 환경에서 Kapso 에이전트는 초기 데이터 및 AI 엔지니어링 harness와 교훈 뱅크를 갖추고 제공되며, 회사의 산출물, 저장소, 파이프라인, 데이터 소스 등을 조사하면서 비즈니스 목표를 스스로 동작하는 AI 및 데이터 시스템으로 바꾸고, 그 과정에서 기업 데이터 자산을 구축해 나갑니다. 기업 과제에 Kapso를 도입하려면 [문의해 주세요](https://leeroo.com/contact-us).

## 기여하기

기여를 환영합니다! 시작하는 방법에 대한 자세한 내용은 [기여 가이드](../../CONTRIBUTING.md)를 참고하세요.

## 기여자

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

[contrib.rocks](https://contrib.rocks)로 만들었습니다.
