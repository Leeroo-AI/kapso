<!-- Russian translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Движок, который проводит эксперименты, развёртывает лучшее решение и учится на каждом запуске</h4>
<p align="center"><sub>умное экспериментирование · обучение · развёртывание · адаптация к предметной области</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <b>Русский</b> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">Подробнее</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Присоединяйтесь к Discord</a> ·
  <a href="https://leeroo.com">Сайт</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  Если вам нравится этот проект, поддержите нас, поставив ему звезду ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Если вы ИИ-агент и смотрите на этот репозиторий:** плагин Kapso с навыками для
> настройки, формулирования цели, написания harness и разбора запуска устанавливается
> одной строкой: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Новости

- **IOAI 2026 · трек AI Model Track.** [Международная олимпиада по ИИ](https://ioai-official.org) является IMO эры ИИ: 471 участник, шесть задач, составленных экспертами, бюджет в один GPU. Её трек [IOAI²](https://ioai-official.org/ai-model-track/) 2026 года предложил ИИ-системам тот же экзамен в одной автономной сессии. Kapso, один из 14 участников-основателей (Founding AI Participants), набрал 536.07 балла по оценке организаторов, выше любого участника-человека, и получил трофей IOAI² Grand Master Trophy, войдя в топ-3 среди заявленных ИИ-систем. [Полные результаты →](../../benchmarks/ioai2026/README.md) · [Обзор трека от IOAI →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Первое место во всех трёх таблицах лидеров [RelBench](../../benchmarks/relbench/README.md).** На бенчмарке Stanford для предиктивного ML над реляционными данными Kapso лидирует в [официальной таблице лидеров](https://star-project.stanford.edu/relbench/leaderboard/) в классификации (81.2 AUROC против 79.3 у [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) и 79.0 у [RT-J](https://star-project.stanford.edu/rt-j) от Stanford STAR), регрессии (0.2476 NMAE против 0.2711 и 0.2847) и рекомендациях (18.4 MAP против 9.3 у ID-GNN, лучшей из остальных заявок).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0 in classification; 0.2476 NMAE against 0.2711 and 0.2847 in regression, lower being better; 18.4 MAP against ID-GNN's 9.3 in recommendations. The first two panels use a truncated axis, the third starts at zero" width="574"></p>

- **[Leeroopedia](https://leeroopedia.com)**: наш публичный банк уроков в областях данных и ML, около 100k страниц, извлечённых экспериментированием над статьями и репозиториями, обслуживающий около 6M запросов в месяц, из них около 28% от ИИ-агентов. Сессии Kapso обращаются к нему по MCP во время генерации идей и реализации.
- **#1 на [MLE-Bench](../../benchmarks/mle/README.md)**: KAPSO занял первое место среди систем с открытым исходным кодом на ML-соревнованиях Kaggle (MLE Benchmark).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574"></p>

- **#1 на [ALE-Bench](../../benchmarks/ale/README.md)**: KAPSO занял первое место на задачах алгоритмического поиска с длинным горизонтом (ALE Benchmark).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574"></p>

## Что такое Kapso?

Kapso представляет собой движок с открытым исходным кодом, который проводит эксперименты, развёртывает лучшее решение и учится на каждом запуске. Он начинает с harness предметной области и пустого банка уроков и, получив цель и критерии успеха:

- **проводит дерево экспериментов**: предлагает гипотезы, затем реализует и проверяет каждую из них;
- **развёртывает лучшее решение** на целевом стеке;
- **учится на запуске**: заслуга присваивается каждой гипотезе и реализации по её влиянию на результат, и банк обновляется, причём каждый урок несёт оценку доверия и границу своей применимости.

Со временем harness предметной области накапливает банк уроков: проверенные экспериментами лучшие практики этой среды. Kapso не привязан ни к модели, ни к кодирующему агенту; его создаёт [Leeroo](https://leeroo.com), и он опубликован на PyPI как [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### Четыре части

| Часть | Глагол | Что делает |
|------|------|--------------|
| [**Умное экспериментирование**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Каждый эксперимент представляет собой гипотезу, реализованную в собственной ветке и получившую балл, в составе дерева. Генерация идей опирается на предыдущие эксперименты и банк уроков и переранжирует идеи по ожидаемой отдаче относительно бюджета, чтобы они подходили именно этой среде, а не были общими идеями голой LLM или RAG; снабжённый отпечатком оценщик, который не может тронуть ни один эксперимент, проверяет результат, а модули могут предлагать обновления оценки на утверждение верификатору. [Как это работает →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Обучение**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Каждый урок выводится индуктивно или дедуктивно из трасс экспериментирования и несёт оценку доверия, границу своей применимости и свои свидетельства; `learn()` обновляет их по результату каждого запуска и A/B-тестирует каждое изменение, перестраивая решение с ним и без него, так что улучшение измеряется, а не угадывается. Банк построен на git: `kapso experience share`, `list` и `update`. [Опыт →](https://docs.leeroo.com/docs/experience) описывает, какой опыт подаётся сборке. |
| [**Развёртывание**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Более короткое экспериментирование того же рода адаптирует лучшее решение к целевому стеку, развёртывает его и подтверждает проверки работоспособности и рабочий вызов. В поставку входят Docker, Modal, BentoCloud и LangGraph Platform; harness может добавить целевую платформу. |
| [**Адаптация к предметной области**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harness является пакетом предметной области Kapso: знания и рабочие процессы какой-либо области, например предиктивного моделирования, чтобы сборка никогда не начиналась с нуля, и место, где накапливается банк уроков. В поставку входят предиктивное моделирование и спортивное программирование; напишите свой с помощью `/kapso:harness` в плагине или `kapso harness` в CLI. |

## 🚀 Быстрый старт

### Установка

**0. Вы в кодирующем агенте?** Установите плагин Kapso и позвольте ему пройти
шаги ниже вместе с вами:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Затем `/kapso:setup` устанавливает и проверяет, а `/kapso:build <what you want>`
запускает первую кампанию (в Codex: `$kapso-setup`, `$kapso-build <what you want>`). См. [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Предварительные требования.** Kapso выполняет инференс через CLI кодирующих
агентов (запасного варианта с прямым API нет), поэтому до того, как что-либо заработает,
вам нужны Node.js и оба CLI агентов с выполненным входом:

```bash
# Node.js 18+ (https://nodejs.org), затем:
npm install -g @openai/codex            # исследование, судейство, утилиты
codex login

npm install -g @anthropic-ai/claude-code  # генерация идей + реализация (хост по умолчанию)
claude auth login
```

Ключ OpenAI необязателен: текст эмбеддят только граф знаний и семантический поиск
по прошлым экспериментам (`models.embedding` в вашем конфиге), а генерация
его никогда не использует. Когда вы включаете любой из них:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Подключите Leeroopedia (рекомендуется).** Один ключ даёт каждой сессии
доступ к [Leeroopedia](https://leeroopedia.com), нашему размещённому в облаке банку уроков по ML и
ИИ-фреймворкам. Устанавливать ничего не нужно; новая учётная запись в
[панели Leeroopedia](https://app.leeroopedia.com/dashboard) идёт с
$20 бесплатного кредита.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Установите пакет** (Python 3.10+):

```bash
pip install leeroo-kapso
```

**4. Проверьте настройку:**

```bash
kapso doctor
```

`doctor` сообщает, что на самом деле нужно **вашему конфигу**, и называет точное
исправление для всего недостающего. Требования следуют за конфигом, поэтому у настройки
только с codex никогда не спросят `claude`. Сузьте до одного глагола, чтобы увидеть
требования только этого глагола:

```bash
kapso doctor build             # или learn / deploy / experience
```

Это те же проверки, которые сам глагол выполняет до начала любой работы:
`kapso.build(...)` падает за секунды на отсутствующем CLI, а не глубоко
внутри сессии. Пункты с пометкой `[-- ]` необязательны; они ограничивают функции,
которые вам могут не понадобиться (docker, Weaviate, Neo4j, целевые платформы развёртывания).

**Или позвольте вашему кодирующему агенту сделать всё это:** вставьте
[промпт настройки](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
в Claude Code, Codex или OpenCode.

**Хранилища, нужные индексу (необязательно):** `kapso experience add --repo`
и `--research` загружают корпус в локальные Weaviate + Neo4j. Из
клонированных исходников:

```bash
bash scripts/start_infra.sh   # запускает оба через docker
```

**Из исходников (для разработки)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

Устаревший адаптер aider является дополнительной зависимостью (`pip install "leeroo-kapso[aider]"`,
Python <3.13); агентам claude/codex по умолчанию дополнения не нужны.

### Базовое использование

Основным глаголам не нужно ничего, кроме предварительных требований выше:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # для старта граф знаний не нужен

# Build: решение через экспериментирование. Кампания печатает
# `status: <path>` при запуске; следите за ней вживую из другого терминала через
#     kapso watch ./campaign
# Если сессии нужно что-то, что можете дать только вы (API-ключ, файл),
# кампания ставится на паузу, а `kapso inbox reply` возобновляет её:
#     https://docs.leeroo.com/docs/build/inbox
# Поделитесь результатом в виде Markdown для pull request или issue через
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # или опустите, чтобы начать с нуля
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Учитесь на только что проведённой кампании: извлеките траекторию, оцените
# уроки и положите в банк карточки знаний, оценённые по свидетельствам. Банк (локальный
# git-репозиторий) создаётся автоматически при первом использовании; уроки остаются на вашей
# машине, пока вы не поделитесь ими:
#     kapso experience share <git-url>
# после чего каждый learn() пушит банк туда.
lesson = kapso.learn(solution)
print(lesson.explain())

# Соберите снова, подав карточки, которые только что заработаны: сборке подаётся ровно
# тот опыт, который она загружает.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

С запущенными хранилищами вы также можете загрузить внешние знания в
индекс и подавать их таким же образом:

```python
from kapso import Kapso

kapso = Kapso()

# Один раз: репозиторий или веб, исследованный по вопросу, в индекс
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Каждая сборка, загружающая индекс, ищет в нём; Leeroopedia подключается по своему ключу
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Замечание о бюджете: загрузка извлекает всё, что поддерживает материал.
Один вопрос может превратиться в десятки связанных страниц и многочасовую
загрузку; время загрузки растёт с объёмом извлекаемой сути, поэтому добавляйте
источники, по которым будете искать, а не все источники, что у вас есть.

А чтобы превратить решение в работающее программное обеспечение:

```python
software = kapso.deploy(solution, target="local")   # или docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # вызов, который подтвердило развёртывание
software.stop()
```

### Как ограничить кампанию?

Используйте `max_iterations`, `time_budget_minutes` и `cost_budget`, чтобы ограничить число итераций, затраченное время и приблизительные расходы на модели в USD.
Для платформенных затрат включите [отбор с учётом стоимости](https://docs.leeroo.com/docs/evolve/cost-aware-selection) в `search_strategy.params.cost` с вашей функцией оценки стоимости, `cap_per_candidate` и `campaign_budget`; его `weight` также позволяет отбору предпочитать более дешёвых кандидатов.

### Выбор моделей

Каждая модель, которую использует Kapso, названа в одном файле конфигурации. Упакованный
конфиг по умолчанию запускает сессии build на `claude-opus-5`, команды обучения на
`claude-fable-5`, а роли codex на `gpt-5.6-sol`, но доступ к моделям
зависит от подписки (план может ограничивать одну модель, обслуживая другую).
Чтобы работать на других моделях, скопируйте упакованный конфиг, отредактируйте и укажите
Kapso на ваш:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# например, запустить команды обучения на opus вместо fable:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Роли чисто текстового инференса могут опционально использовать `openai_compatible` Chat Completions.
Задайте модель и эндпоинт в вашем конфиге; учётные данные остаются в окружении.
Загрузите конфиг через `Kapso(config=...)` или `kapso build --config ...`.
Написание кода и веб-исследование по-прежнему требуют агентов с соответствующими инструментами.
См. [пример конфигурации и ограничения](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Перед долгим запуском проверьте каждую модель, названную в вашем конфиге, на ваших
реальных подписках: модель, которую ваш логин не может обслужить, падает здесь
за секунды, а не спустя часы запуска. Лимит использования модели, которую вы
можете обслужить, однотокенной пробе не виден:

```bash
kapso doctor --models                            # упакованный конфиг
kapso doctor --models --config kapso-config.yaml # ваш
kapso doctor learn --models                      # только команды обучения
```

Чтобы сделать эту пробу частью каждого вызова, а не тем, о чём нужно помнить,
включите её в конфиге. Она стоит один выбрасываемый токен на каждую уникальную
пару `{cli, model}`, что окупается перед многочасовым
`learn()` без присмотра:

```yaml
preflight:
  enabled: true            # статические проверки перед каждым глаголом (по умолчанию)
  live_model_probe: true   # + однотокенные пробы моделей (по умолчанию: false)
```

Смена моделей меняет и *темп*: ограничения `timeout_minutes` для команд в
конфиге были откалиброваны на моделях по умолчанию, и заменённой модели, которая
рассуждает дольше, может потребоваться их повышение.

Подробные шаги интеграции см. в руководствах [Быстрый старт](https://docs.leeroo.com/docs/first-build) и [Установка](https://docs.leeroo.com/docs/installation).

## Поддерживаемые бенчмарки

| Бенчмарк | Описание |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | Бенчмарк OpenAI по ML-инженерии: полные соревнования по табличным данным, зрению, тексту и аудио, от сырых данных до оценённой посылки |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Бенчмарк Sakana AI по алгоритмической оптимизации: проектируйте, реализуйте и итеративно улучшайте конкурсные эвристики в многочасовых поисках |
| [**RelBench**](../../benchmarks/relbench/README.md) | Бенчмарк Stanford для предиктивного ML над реляционными данными: классификация, регрессия и рекомендации прямо из многотабличных баз данных Amazon, H&M, Stack Exchange, клинических испытаний и других |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | Олимпиадный ML на время по зрению, языку и оптимизации: задачи, составленные экспертами, конкурсное оборудование, без помощи человека |

У каждого бенчмарка также есть страница документации о том, как его запустить, его опциях CLI и о том, что содержит вывод:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## Примеры

Каждый пример является задачей: бриф, стартовый репозиторий, данные или способ их получить и драйвер не более чем из двенадцати строк; в [examples/README.md](../../examples/README.md) указаны минуты и единственная команда.

| Пример | Задача |
|---------|----------|
| [**Инженерия данных (модели dbt в Snowflake)**](../../examples/dbt-snowflake/README.md) | Исправить модели dbt с проблемами качества данных и построить новые, которых требует бэклог из семнадцати тикетов, в розничном хранилище на Snowflake, пока каждый тикет не пройдёт свои производственные тесты; затем доставить по одному pull request на тикет и выпустить их в продакшен. |
| [**Оптимизация harness (агент клиентской поддержки)**](../../examples/support-agent-harness/README.md) | Оптимизировать harness агента клиентской поддержки, чтобы перевести базовую LLM с Fable 5.1 на локальную Qwen 27B без потери доли решённых клиентских диалогов, затем развернуть его в Docker и извлечь уроки из запуска. |
| [**Предиктивное моделирование (отток клиентов)**](../../examples/churn-first-build/README.md) | Предсказать, какие клиенты уйдут: поднять ROC AUC модели оттока на отложенной выборке к целевому значению под оценщиком, который кампания не может редактировать. |
| [**Обнаружение мошенничества**](../../examples/fraud-no-evaluator/README.md) | Оценивать транзакции на мошенничество, когда в репозитории нет оценщика: бриф задаёт протокол, кампания сначала строит оценщик, затем улучшает скорер. |
| [**Количественные финансы**](../../examples/trading/README.md) | Дневная торговая стратегия на синтетических инструментах: поднять коэффициент Шарпа на отложенной выборке на торговом harness с его проверками на каждой попытке. Исследовательский пример, а не финансовый совет. |
| [**Табличный ML**](../../examples/end-to-end/README.md) | Соревнование Kaggle от начала до конца: Spaceship Titanic от фиктивного классификатора до настоящей модели, затем разбор, обучение, повторная сборка с уроком и развёртывание. |
| [**Оптимизация PyTorch**](../../examples/torch-performance/README.md) | Ускорить модуль PyTorch на CUDA в пределах численного допуска: слияние операций, компиляция, меньше обращений к памяти. |
| [**Оптимизация GPU-ядер**](../../examples/gpu-kernels/README.md) | Ускорить многоголовое self-attention на CUDA с помощью flash attention или ядра Triton в пределах численного допуска. |

## 📚 Документация и поддержка

- **Полная документация**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Установка](https://docs.leeroo.com/docs/installation): CLI кодирующих агентов, пакет и `kapso doctor`
  - [Быстрый старт](https://docs.leeroo.com/docs/first-build): ваша первая кампания
  - [Справочник CLI](https://docs.leeroo.com/docs/reference/cli): каждая команда, флаг и значение по умолчанию
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): каждый публичный метод `Kapso`
  - [Конфигурация](https://docs.leeroo.com/docs/reference/configuration): каждый ключ в `config.yaml`
  - [Что производит learn](https://docs.leeroo.com/docs/learn/what-learn-produces): банк уроков, каждый с оценкой доверия, границей и своими свидетельствами, и [режим разработки](https://docs.leeroo.com/docs/learn/development-mode)
  - [Умное экспериментирование](https://docs.leeroo.com/docs/evolve/overview): дерево экспериментов и его модули, а также [что происходит внутри одного эксперимента](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Опыт](https://docs.leeroo.com/docs/experience): ваш банк, Leeroopedia, репозитории и статьи, превращённые в знания с поиском
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal и другие целевые платформы
  - [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills): команды, навыки и MCP-сервер для Claude Code, Codex и Cursor
- **Из вашего кодирующего агента**: документация запускает MCP-сервер по адресу `https://docs.leeroo.com/mcp`, и каждая страница отдаётся как Markdown; см. [llms.txt](https://docs.leeroo.com/llms.txt) и [Kapso в вашем кодирующем агенте](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Позвольте вашему кодирующему агенту запускать Kapso**: плагин в [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) содержит десять команд, шесть навыков и MCP-сервер, одна строка установки на хост (шаг 0 выше); OpenCode загружает навыки как папку (`ln -s <clone>/plugin/skills .opencode/skills`). См. [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills).
- **Дайте вашему кодирующему агенту инструменты знаний Kapso**: `kapso-mcp`, устанавливаемый вместе с пакетом, предоставляет веб-исследование, память репозитория, историю экспериментов кампании и банк уроков той сессии, в которой вы работаете; см. [MCP-шлюзы](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; затем `startup_timeout_sec = 60` в [mcp_servers.kapso] в ~/.codex/config.toml (десяти секунд Codex мало на занятой машине)
  ```
- **Сообщество**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Сайт**: [leeroo.com](https://leeroo.com)


## Kapso для предприятий

На предприятии агенты Kapso поставляются с начальными harness по инженерии данных и ИИ и банком уроков и самостоятельно превращают бизнес-цели в работающие системы ИИ и данных, исследуя артефакты компании, репозитории, пайплайны, источники данных и многое другое, попутно наращивая корпоративное хозяйство данных. Чтобы подключить Kapso для задач вашего предприятия, [свяжитесь с нами](https://leeroo.com/contact-us).

## Участие в разработке

Мы приветствуем вклад в проект! Подробности о том, как начать, см. в нашем [руководстве для участников](../../CONTRIBUTING.md).

## Участники

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

Сделано с помощью [contrib.rocks](https://contrib.rocks).
