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
  이 프로젝트가 마음에 드신다면 스타 ⭐를 눌러 응원해 주세요
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **이 저장소를 보고 있는 AI 에이전트라면:** 설정, 목표 서술, harness 작성, 실행 검토를 위한
> 스킬이 담긴 Kapso 플러그인은 한 줄로 설치됩니다:
> `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## 뉴스

- **IOAI 2026 · AI 모델 트랙.** [국제 AI 올림피아드](https://ioai-official.org)는 AI 시대의 IMO입니다: 참가자 471명, 전문가가 설계한 여섯 개의 과제, 단일 GPU 예산. 2026년 [IOAI²](https://ioai-official.org/ai-model-track/) 트랙은 AI 시스템에 같은 시험을 한 번의 자율 세션으로 치르게 했습니다. 14개의 창립 AI 참가자 중 하나인 Kapso는 주최 측 채점으로 536.07점을 기록해 모든 인간 참가자를 앞섰고, 출전한 AI 시스템 가운데 상위 3위 안에 드는 IOAI² Grand Master Trophy를 받았습니다. [전체 결과 →](../../benchmarks/ioai2026/README.md) · [IOAI의 트랙 리뷰 →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **세 개의 [RelBench](../../benchmarks/relbench/README.md) 리더보드 모두에서 선두.** 관계형 데이터에 대한 예측 ML을 위한 Stanford의 벤치마크에서 Kapso는 분류([KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview)의 79.3과 Stanford STAR의 [RT-J](https://star-project.stanford.edu/rt-j)의 79.0 대비 81.2 AUROC), 회귀(0.2711과 0.2847 대비 0.2476 NMAE), 추천(차순위 항목인 ID-GNN의 9.3 대비 18.4 MAP)에서 [공식 리더보드](https://star-project.stanford.edu/relbench/leaderboard/)를 이끌고 있습니다.

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0 in classification; 0.2476 NMAE against 0.2711 and 0.2847 in regression, lower being better; 18.4 MAP against ID-GNN's 9.3 in recommendations. The first two panels use a truncated axis, the third starts at zero" width="574"></p>

- **[Leeroopedia](https://leeroopedia.com)**: 데이터 및 ML 도메인의 공개 교훈 뱅크로, 논문과 저장소를 대상으로 실험하여 추출한 약 100k 페이지를 담고 있으며, 월 약 6M 건의 요청을 처리하고 그중 약 28%는 AI 에이전트에서 옵니다. Kapso의 세션은 아이디어 구상과 구현 중에 MCP를 통해 이를 질의합니다.
- **[MLE-Bench](../../benchmarks/mle/README.md) 1위**: KAPSO는 Kaggle ML 대회(MLE Benchmark)에서 오픈소스 시스템 중 최상위 순위를 달성했습니다.

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574"></p>

- **[ALE-Bench](../../benchmarks/ale/README.md) 1위**: KAPSO는 장기 알고리즘 발견 문제(ALE Benchmark)에서 최상위 순위를 달성했습니다.

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574"></p>

## Kapso란 무엇인가?

Kapso는 실험을 실행하고, 최선의 솔루션을 배포하며, 모든 실행에서 학습하는 오픈소스 엔진입니다. 도메인 harness와 빈 교훈 뱅크에서 출발하여, 목표와 성공 기준이 주어지면:

- **실험 트리를 실행합니다**: 가설을 제안한 뒤 각각을 구현하고 검증합니다;
- **최선의 솔루션을 대상 스택에 배포합니다**;
- **실행에서 학습합니다**: 각 가설과 구현에는 결과에 미친 영향에 따라 기여도가 부여되고, 뱅크가 갱신되며, 모든 교훈은 신뢰도 점수와 적용 범위에 대한 경계를 지닙니다.

시간이 지남에 따라 도메인 harness는 교훈 뱅크, 즉 해당 환경에서 실험으로 검증된 모범 사례를 축적합니다. Kapso는 모델에 구애받지 않고 코딩 에이전트에도 구애받지 않으며, [Leeroo](https://leeroo.com)가 만들어 PyPI에 [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/)로 게시했습니다.

### 네 가지 구성 요소

| 구성 요소 | 동사 | 하는 일 |
|------|------|--------------|
| [**스마트 실험**](https://docs.leeroo.com/docs/build/brief) | `.build()` | 각 실험은 하나의 가설을 자체 브랜치에서 구현하고 점수를 매긴 것으로, 트리를 이룹니다. 아이디어 구상은 이전 실험과 교훈 뱅크를 바탕으로 하며 예산 대비 기대 수익에 따라 아이디어를 재정렬하므로, 순수 LLM이나 RAG의 일반적인 아이디어가 아니라 이 환경에 맞는 아이디어가 나옵니다; 어떤 실험도 손댈 수 없는 지문이 찍힌 평가기가 결과를 검증하고, 모듈은 검증기가 승인하도록 평가 갱신을 제안할 수 있습니다. [작동 방식 →](https://docs.leeroo.com/docs/evolve/overview) |
| [**학습**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | 모든 교훈은 실험 트레이스에서 귀납 또는 연역되며 신뢰도 점수, 적용 범위에 대한 경계, 그 증거를 지닙니다; `learn()`은 각 실행의 결과로부터 이를 갱신하고, 모든 변경을 해당 변경이 있는 경우와 없는 경우로 다시 빌드하여 A/B 테스트하므로 개선은 추측이 아니라 측정됩니다. 뱅크는 git 위에 구축됩니다: `kapso experience share`, `list`, `update`. [경험 →](https://docs.leeroo.com/docs/experience)은 빌드에 무엇이 제공되는지를 정의합니다. |
| [**배포**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | 같은 종류의 더 짧은 실험이 최선의 솔루션을 대상 스택에 맞게 적응시키고, 배포하며, 상태 점검과 실제 호출이 동작함을 증명합니다. Docker, Modal, BentoCloud, LangGraph Platform이 기본 제공되며, harness가 대상을 추가할 수 있습니다. |
| [**도메인 적응**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | harness는 Kapso의 도메인 패키지입니다: 예측 모델링과 같은 영역의 지식과 워크플로를 담아 빌드가 결코 맨땅에서 시작하지 않게 하며, 교훈 뱅크가 축적되는 곳입니다. 예측 모델링과 경쟁 프로그래밍이 기본 제공되며, 플러그인에서는 `/kapso:harness`로, CLI에서는 `kapso harness`로 직접 작성하세요. |

## 🚀 빠른 시작

### 설치

**0. 코딩 에이전트 안에 계신가요?** Kapso 플러그인을 설치하고 아래 단계를
함께 진행하게 하세요:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

그런 다음 `/kapso:setup`이 설치하고 검증하며, `/kapso:build <what you want>`가
첫 캠페인을 시작합니다(Codex에서는 `$kapso-setup`, `$kapso-build <what you want>`). [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills)을 참고하세요.

**1. 사전 요구 사항.** Kapso는 코딩 에이전트 CLI를 통해 추론을 실행하므로
(직접 API 폴백은 없습니다), 무엇이든 동작하려면 Node.js와 두 에이전트
CLI가 모두 로그인되어 있어야 합니다:

```bash
# Node.js 18+ (https://nodejs.org), 그다음:
npm install -g @openai/codex            # 리서치, 판정, 유틸리티
codex login

npm install -g @anthropic-ai/claude-code  # 아이디어 구상 + 구현 (기본 호스트)
claude auth login
```

OpenAI 키는 선택 사항입니다: 지식 그래프와 과거 실험에 대한 시맨틱 검색
(설정의 `models.embedding`)만 텍스트를 임베딩하며, 생성에는 결코
사용되지 않습니다. 둘 중 하나를 켤 때:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia 연결(권장).** 키 하나로 모든 세션에 ML 및 AI 프레임워크에
대한 호스팅 교훈 뱅크인 [Leeroopedia](https://leeroopedia.com)가 제공됩니다.
설치할 것은 없으며, [Leeroopedia 대시보드](https://app.leeroopedia.com/dashboard)에서
새 계정을 만들면 $20의 무료 크레딧이 제공됩니다.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. 패키지 설치**(Python 3.10+):

```bash
pip install leeroo-kapso
```

**4. 설정 검증:**

```bash
kapso doctor
```

`doctor`는 **여러분의 설정**이 실제로 필요로 하는 것을 보고하고, 누락된
항목마다 정확한 해결 방법을 알려 줍니다. 요구 사항은 설정을 따르므로,
codex만 사용하는 구성에서는 결코 `claude`를 요구하지 않습니다. 동사 하나로
범위를 좁히면 해당 동사의 요구 사항만 볼 수 있습니다:

```bash
kapso doctor build             # 또는 learn / deploy / experience
```

이는 각 동사가 작업을 시작하기 전에 직접 실행하는 것과 동일한 검사입니다.
`kapso.build(...)`는 CLI가 없으면 세션 깊숙한 곳에서가 아니라 몇 초 만에
실패합니다. `[-- ]`로 표시된 항목은 선택 사항이며, 필요하지 않을 수도 있는
기능(docker, Weaviate, Neo4j, 배포 대상)만 제한합니다.

**또는 코딩 에이전트에게 이 모든 것을 맡기세요:**
[설정 프롬프트](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)를
Claude Code, Codex 또는 OpenCode에 붙여 넣으세요.

**인덱스에 필요한 스토어(선택 사항)**: `kapso experience add --repo`와
`--research`는 코퍼스를 로컬 Weaviate + Neo4j에 수집합니다. 소스
체크아웃에서:

```bash
bash scripts/start_infra.sh   # docker로 둘 다 시작합니다
```

**소스에서 설치(개발용)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

레거시 aider 어댑터는 추가 기능입니다(`pip install "leeroo-kapso[aider]"`,
Python <3.13); 기본 claude/codex 에이전트에는 추가 기능이 필요 없습니다.

### 기본 사용법

핵심 동사들은 위의 사전 요구 사항 외에는 아무것도 필요로 하지 않습니다:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # 시작하는 데 지식 그래프는 필요 없습니다

# Build: 실험을 통해 솔루션을 만듭니다. 캠페인은 시작 시
# `status: <path>`를 출력합니다. 다른 터미널에서 실시간으로 지켜보려면
#     kapso watch ./campaign
# 세션이 여러분만 제공할 수 있는 것(API 키, 파일)을 필요로 하면
# 캠페인은 일시 중지되고 `kapso inbox reply`가 이를 재개합니다:
#     https://docs.leeroo.com/docs/build/inbox
# 결과를 풀 리퀘스트나 이슈용 Markdown으로 공유하려면
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # 또는 생략하여 맨땅에서 시작
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# 방금 실행한 캠페인에서 학습합니다: 궤적을 채굴하고, 교훈에 등급을
# 매기고, 증거로 가격이 매겨진 지식 카드를 뱅크에 저장합니다. 뱅크(로컬
# git 저장소)는 첫 사용 시 자동으로 생성됩니다. 교훈은 공유하기 전까지
# 여러분의 머신에 머뭅니다:
#     kapso experience share <git-url>
# 그 이후에는 모든 learn()이 뱅크를 그곳으로 푸시합니다.
lesson = kapso.learn(solution)
print(lesson.explain())

# 방금 얻은 카드를 제공받아 다시 빌드합니다: 빌드에는 정확히
# 로드한 경험이 제공됩니다.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

스토어가 실행 중이면 외부 지식을 인덱스에 수집하고 같은 방식으로
제공할 수도 있습니다:

```python
from kapso import Kapso

kapso = Kapso()

# 한 번만: 저장소 하나, 또는 질문에 대해 리서치한 웹을 인덱스로
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# 인덱스를 로드하는 모든 빌드가 이를 검색합니다; Leeroopedia는 자체 키로 함께 제공됩니다
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

예산에 대한 참고: 수집은 자료가 뒷받침하는 모든 것을 추출합니다.
질문 하나가 수십 개의 연결된 페이지와 몇 시간에 걸친 수집이 될 수
있습니다; 수집 시간은 추출 가능한 내용의 양에 비례하므로, 보유한 모든
소스가 아니라 검색할 소스만 추가하세요.

그리고 솔루션을 실행 중인 소프트웨어로 바꾸려면:

```python
software = kapso.deploy(solution, target="local")   # 또는 docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # 배포가 증명한 호출
software.stop()
```

### 캠페인의 범위는 어떻게 제한하나요?

`max_iterations`, `time_budget_minutes`, `cost_budget`를 사용하여 반복 횟수, 경과 시간, USD 기준 최선 노력의 모델 지출을 제한하세요.
플랫폼 비용의 경우 `search_strategy.params.cost`에서 여러분의 추정기, `cap_per_candidate`, `campaign_budget`와 함께 [비용 인식 선택](https://docs.leeroo.com/docs/evolve/cost-aware-selection)을 활성화하세요; 그 `weight`를 통해 선택이 더 저렴한 후보를 선호하게 할 수도 있습니다.

### 모델 선택

Kapso가 사용하는 모든 모델은 하나의 설정 파일에 명시됩니다. 패키지
기본값은 빌드 세션을 `claude-opus-5`에서, 학습 크루를
`claude-fable-5`에서, codex 역할을 `gpt-5.6-sol`에서 실행합니다. 하지만
모델 접근 권한은 구독에 따라 다릅니다(한 요금제가 어떤 모델은 제한하면서
다른 모델은 제공할 수 있습니다). 다른 모델로 실행하려면 패키지 설정을
복사하고 편집한 뒤 Kapso가 여러분의 설정을 가리키게 하세요:

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
설정에서 모델과 엔드포인트를 지정하세요; 자격 증명은 환경 변수에 둡니다.
설정은 `Kapso(config=...)` 또는 `kapso build --config ...`로 로드합니다.
코딩과 웹 리서치에는 여전히 해당 도구를 갖춘 에이전트가 필요합니다.
[설정 예시와 제한 사항](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)을 참고하세요.

긴 실행 전에 설정에 명시된 모든 모델을 실제 구독에 대해 점검하세요.
로그인으로 제공받을 수 없는 모델은 실행 몇 시간 뒤가 아니라 여기서
몇 초 만에 실패합니다. 제공받을 수 있는 모델에 걸린 사용량 한도는
토큰 한 개짜리 점검으로는 보이지 않습니다:

```bash
kapso doctor --models                            # 패키지 기본 설정
kapso doctor --models --config kapso-config.yaml # 여러분의 설정
kapso doctor learn --models                      # 학습 크루만
```

이 점검을 기억해서 실행하는 일이 아니라 모든 호출의 일부로 만들려면
설정에서 켜세요. 서로 다른 `{cli, model}` 쌍마다 버리는 토큰 한 개의
비용이 들며, 무인으로 몇 시간 돌아가는 `learn()` 전에는 그만한 가치가
있습니다:

```yaml
preflight:
  enabled: true            # 모든 동사 전의 정적 검사 (기본값)
  live_model_probe: true   # + 토큰 한 개짜리 모델 점검 (기본값: false)
```

모델 교체는 *속도 조절*도 바꿉니다: 설정의 크루 `timeout_minutes` 상한은
기본 모델을 기준으로 보정된 것이며, 더 오래 추론하는 모델로 교체하면
상한을 올려야 할 수 있습니다.

자세한 통합 단계는 [빠른 시작](https://docs.leeroo.com/docs/first-build)과 [설치](https://docs.leeroo.com/docs/installation) 가이드를 참고하세요.

## 지원되는 벤치마크

| 벤치마크 | 설명 |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI의 ML 엔지니어링 벤치마크로, 표 형식, 비전, 텍스트, 오디오에 걸친 전체 대회를 원시 데이터에서 채점된 제출물까지 수행합니다 |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI의 알고리즘 최적화 벤치마크로, 몇 시간에 걸친 탐색 동안 대회 휴리스틱을 설계, 구현, 반복합니다 |
| [**RelBench**](../../benchmarks/relbench/README.md) | 관계형 데이터에 대한 예측 ML을 위한 Stanford의 벤치마크로, Amazon, H&M, Stack Exchange, 임상 시험 등의 다중 테이블 데이터베이스에서 바로 분류, 회귀, 추천을 수행합니다 |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | 비전, 언어, 최적화에 걸친 시간제한 올림피아드 ML로, 전문가가 출제한 과제, 대회 하드웨어, 인간의 도움은 전혀 없음 |

각 벤치마크에는 실행 방법, CLI 옵션, 출력 내용을 다루는 문서 페이지도 있습니다:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 예제

모든 예제는 하나의 과제입니다: 브리프, 시작 저장소, 데이터 또는 데이터를 얻는 방법, 그리고 최대 열두 줄의 드라이버; [examples/README.md](../../examples/README.md)에 소요 시간(분)과 단일 명령이 있습니다.

| 예제 | 과제 |
|---------|----------|
| [**데이터 엔지니어링(Snowflake의 dbt 모델)**](../../examples/dbt-snowflake/README.md) | Snowflake 기반 소매 웨어하우스에서 데이터 품질 문제가 있는 dbt 모델을 고치고, 열일곱 개 티켓으로 이루어진 백로그가 요구하는 새 모델을 빌드하여 모든 티켓이 프로덕션 테스트를 통과할 때까지 진행합니다. 그런 다음 티켓마다 pull request를 하나씩 전달하고 프로덕션에 릴리스합니다. |
| [**harness 최적화(고객 지원 에이전트)**](../../examples/support-agent-harness/README.md) | 고객 지원 에이전트의 harness를 최적화하여, 고객 대화의 해결률을 잃지 않으면서 기반 LLM을 Fable 5.1에서 로컬 Qwen 27B로 전환합니다. 그런 다음 Docker에 배포하고 그 실행에서 학습합니다. |
| [**예측 모델링(고객 이탈)**](../../examples/churn-first-build/README.md) | 어떤 고객이 떠날지 예측합니다: 캠페인이 편집할 수 없는 평가기 아래에서 이탈 모델의 표본 외 ROC AUC를 목표치까지 끌어올립니다. |
| [**사기 탐지**](../../examples/fraud-no-evaluator/README.md) | 저장소에 평가기가 없을 때 거래의 사기 여부를 점수화합니다: 브리프가 프로토콜을 명시하고, 캠페인이 먼저 평가기를 만든 뒤 점수기를 개선합니다. |
| [**퀀트 금융**](../../examples/trading/README.md) | 합성 상품에 대한 일간 트레이딩 전략: 모든 시도마다 검사가 수행되는 트레이딩 harness에서 표본 외 Sharpe를 끌어올립니다. 리서치 예제이며 투자 조언이 아닙니다. |
| [**표 형식 ML**](../../examples/end-to-end/README.md) | Kaggle 대회를 처음부터 끝까지: Spaceship Titanic을 더미 분류기에서 실제 모델로, 그런 다음 검토하고, 학습하고, 교훈을 가지고 다시 빌드하고, 배포합니다. |
| [**PyTorch 최적화**](../../examples/torch-performance/README.md) | 수치 허용 오차 내에서 PyTorch 모듈을 CUDA에서 더 빠르게 만듭니다: 퓨전, 컴파일, 메모리 트래픽 감소. |
| [**GPU 커널 최적화**](../../examples/gpu-kernels/README.md) | 수치 허용 오차 내에서 플래시 어텐션 또는 Triton 커널로 CUDA의 멀티헤드 셀프 어텐션 속도를 높입니다. |

## 📚 문서 및 지원

- **전체 문서**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [설치](https://docs.leeroo.com/docs/installation): 코딩 에이전트 CLI, 패키지, 그리고 `kapso doctor`
  - [빠른 시작](https://docs.leeroo.com/docs/first-build): 첫 캠페인
  - [CLI 레퍼런스](https://docs.leeroo.com/docs/reference/cli): 모든 명령, 플래그, 기본값
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso`의 모든 공개 메서드
  - [설정](https://docs.leeroo.com/docs/reference/configuration): `config.yaml`의 모든 키
  - [learn이 만들어 내는 것](https://docs.leeroo.com/docs/learn/what-learn-produces): 각각 신뢰도 점수, 경계, 증거를 지닌 교훈 뱅크, 그리고 [개발 모드](https://docs.leeroo.com/docs/learn/development-mode)
  - [스마트 실험](https://docs.leeroo.com/docs/evolve/overview): 실험 트리와 그 모듈, 그리고 [하나의 실험 안에서 일어나는 일](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [경험](https://docs.leeroo.com/docs/experience): 여러분의 뱅크, Leeroopedia, 검색 가능한 지식으로 변환된 저장소와 논문
  - [배포](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal 및 기타 대상
  - [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code, Codex, Cursor용 명령, 스킬, MCP 서버
- **코딩 에이전트에서**: 문서는 `https://docs.leeroo.com/mcp`에서 MCP 서버를 운영하며, 모든 페이지는 Markdown으로 제공됩니다. [llms.txt](https://docs.leeroo.com/llms.txt)와 [코딩 에이전트에서의 Kapso](https://docs.leeroo.com/docs/coding-agent-skills)를 참고하세요.
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **코딩 에이전트가 Kapso를 실행하게 하세요**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) 아래의 플러그인은 열 개의 명령, 여섯 개의 스킬, MCP 서버를 담고 있으며 호스트마다 설치 한 줄이면 됩니다(위 0단계); OpenCode는 스킬을 폴더로 로드합니다(`ln -s <clone>/plugin/skills .opencode/skills`). [Kapso 플러그인](https://docs.leeroo.com/docs/coding-agent-skills)을 참고하세요.
- **코딩 에이전트에 Kapso의 지식 도구를 제공하세요**: 패키지와 함께 설치되는 `kapso-mcp`는 웹 리서치, 저장소의 메모리, 캠페인의 실험 이력, 교훈 뱅크를 여러분이 작업하는 세션에 제공합니다. [MCP 게이트](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)를 참고하세요.
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; 그다음 ~/.codex/config.toml의 [mcp_servers.kapso] 아래에 `startup_timeout_sec = 60` 추가 (Codex의 십 초는 바쁜 머신에서는 짧습니다)
  ```
- **커뮤니티**: [Discord](https://discord.gg/hqVbPNNEZM)
- **웹사이트**: [leeroo.com](https://leeroo.com)


## 기업용 Kapso

기업 환경에서 Kapso 에이전트는 초기 데이터 및 AI 엔지니어링 harness와 교훈 뱅크를 갖추고 출시되며, 회사의 아티팩트, 저장소, 파이프라인, 데이터 소스 등을 조사하면서 비즈니스 목표를 스스로 동작하는 AI 및 데이터 시스템으로 바꾸고, 그 과정에서 기업의 데이터 자산을 구축해 나갑니다. 기업 과제에 Kapso를 도입하려면 [문의해 주세요](https://leeroo.com/contact-us).

## 기여하기

기여를 환영합니다! 시작하는 방법에 대한 자세한 내용은 [기여 가이드](../../CONTRIBUTING.md)를 참고하세요.

## 기여자

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

[contrib.rocks](https://contrib.rocks)로 만들었습니다.
