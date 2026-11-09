<!-- Russian translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Движок, который проводит эксперименты, развёртывает лучшее решение и учится на каждом запуске</h4>
<p align="center"><sub>умное экспериментирование · обучение · развёртывание · доменная адаптация</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <b>Русский</b> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">Узнать больше</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Присоединиться к Discord</a> ·
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

> **Если вы ИИ-агент и смотрите на этот репозиторий:** плагин Kapso, с навыками для
> настройки, формулировки цели, написания harness и разбора запуска, устанавливается
> одной строкой: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Новости

- **IOAI 2026 · трек AI Model Track.** [Международная олимпиада по ИИ](https://ioai-official.org) является IMO эпохи ИИ: 471 участник из 108 стран и территорий, шесть разработанных экспертами задач при бюджете в один GPU. В 2026 году на ней открылся [IOAI²](https://ioai-official.org/ai-model-track/), где системы ИИ сдают *тот же* экзамен в двух 6-часовых сессиях без участия человека. Как только запускается отсчёт времени, ни один человек не может ничего решать, исправлять или улучшать. Kapso выступил как один из 14 участников-основателей среди систем ИИ.
  - 🎓 **Обошёл всех 471 участников-людей**: 536.07 балла в сумме, выше любого участника в зале.
  - 🏆 **IOAI² Grand Master Trophy**: топ-3 среди всех заявленных систем ИИ. [Полные результаты →](../../benchmarks/ioai2026/README.md) · [Обзор трека от IOAI →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Первое место во всех трёх таблицах лидеров [RelBench](../../benchmarks/relbench/README.md)**: на бенчмарке Stanford для предиктивного ML на реляционных данных Kapso возглавляет официальную таблицу лидеров в классификации (81.2 AUROC по 12 задачам против 79.3 у [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) и 79.0 у [RT-J](https://star-project.stanford.edu/rt-j) от Stanford STAR), регрессии (0.2476 NMAE по 9 задачам против 0.2711 у RT-J и 0.2847 у KumoRFM-v2, на 8.7% меньше ошибки) и рекомендациях (18.4 MAP по 10 задачам против 9.3 у ID-GNN, лучшей из остальных заявок). Каждое число является записью в [официальной таблице лидеров RelBench](https://star-project.stanford.edu/relbench/leaderboard/). Обучение, хранящееся в harness, превзошло обучение, зашитое в веса модели, а обновление банка уроков обходится намного дешевле, чем переобучение табличной фундаментальной модели.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)**: вики наших уроков в областях данных и ML, около 100k страниц, извлечённых в ходе экспериментов над статьями и репозиториями, обслуживающая около 6M запросов в месяц, около 28% из них от ИИ-агентов. Kapso подключается к ней по MCP: его сессии используют её во время генерации идей и реализации для поиска знаний, построения планов и диагностики сбоев.
- **#1 на [MLE-Bench](../../benchmarks/mle/README.md)**: KAPSO занял первое место среди систем с открытым исходным кодом на ML-соревнованиях Kaggle (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **#1 на [ALE-Bench](../../benchmarks/ale/README.md)**: KAPSO занял первое место на задачах алгоритмических открытий с длинным горизонтом (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## Что такое Kapso?

Получить результат в области ИИ или данных внутри компании означает узнать две вещи экспериментальным путём. Ни один человек и ни один ИИ-агент не знает, как устроен и как эксплуатируется ландшафт данных; единственный способ узнать это состоит в том, чтобы исследовать артефакты компании (репозитории, пайплайны, документацию, источники данных), выдвигать гипотезы, проводить эксперименты и учиться на них, пока не станут понятны поведение каждой системы и её условия. И даже когда такое понимание есть, лучшее решение всё равно нельзя выбрать заранее, идёт ли речь об оптимизации хранилища или о предсказании оттока, поэтому вы экспериментируете и учитесь, пока одно из них не подойдёт окружению. Ноу-хау, подходящее окружению, накапливается, во многом как в RL. RAG делает обратное: он снабжает агента общими знаниями, которые никогда не проверялись в этом окружении, причём все они имеют одинаковый вес, и эти знания мало чего стоят, пока их там не опробовали.

Именно для этого мы и построили движок Kapso. Он начинает с доменного harness и пустого банка уроков; получив цель и критерии успеха, он выполняет дерево экспериментов, выдвигая гипотезы, затем реализуя и проверяя каждую из них, и развёртывает лучшее решение на целевом стеке. Затем он учится на запуске, распределяя заслуги между гипотезами и реализациями по их влиянию на результат и обновляя банк, где каждый урок несёт показатель достоверности и границу его применимости. Со временем доменный harness несёт в себе банк уроков, накопивший проверенные экспериментами лучшие практики в этом окружении. Система не зависит ни от модели, ни от кодинг-агента: открытый исходный код, от [Leeroo](https://leeroo.com), опубликована на PyPI как [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### Четыре части

| Часть | Глагол | Что делает |
|------|------|--------------|
| [**Умное экспериментирование**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Специализированные модули разделяют одну структуру данных, дерево экспериментов: каждый узел записывает гипотезу, реализацию (git-ветку и технические трудности), использованный бюджет и расстояние до цели. Для каждого эксперимента модуль генерации идей предлагает идеи на основе предыдущих экспериментов и банка уроков, переранжирует их по ожидаемой отдаче относительно бюджета, который потребовала бы каждая из них, и отправляет лучшие по рангу реализатору, который строит на ветке, ответвлённой от родительского узла. Поскольку банк продолжает учиться тому, что работает именно в этом окружении, идеи становятся специфичными для него, а не общими, какие предложил бы голый LLM или RAG. Результаты проверяются оценщиком с цифровым отпечатком, к которому ни один эксперимент не имеет доступа, и оценка тоже эволюционирует: модули экспериментирования могут предлагать обновления метрик и критериев на утверждение верификатору. [Как это работает →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Обучение**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Всё в банке происходит из экспериментов: каждый урок индуцируется или дедуцируется из трасс запусков экспериментирования, несёт показатель достоверности, граничные условия его применимости и стоящие за ним свидетельства, и после каждого запуска `learn()` обновляет их по влиянию урока на итоговый результат. Изменения банка тоже являются экспериментами: каждый обновлённый урок проходит A/B-тест, в котором одна и та же сборка запускается с ним и без него, так что улучшение измеряется, а не угадывается. Банк построен на git, поэтому вы можете делиться своим опытом с другими (`kapso experience share`), перечислять доступные вам банки (`kapso experience list`) и обновлять их (`kapso experience update`). [Опыт →](https://docs.leeroo.com/docs/experience) называет то, что предоставляется сборке: ваш банк, банк, которым делится ваша команда, Leeroopedia или корпус, загруженный из репозитория или исследованного вопроса. |
| [**Развёртывание**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Выполняет более короткое экспериментирование того же рода, чтобы адаптировать лучшее решение к целевому стеку: сессия адаптирует копию узла, развёртывает её и демонстрирует проверки работоспособности и рабочий вызов. Docker, Modal, BentoCloud и LangGraph Platform поставляются в комплекте, и вы можете добавить в harness свой собственный целевой стек. |
| [**Доменная адаптация**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harness представляют собой доменные пакеты Kapso: знания и рабочие процессы для области, такой как предиктивное моделирование, чтобы сборка не экспериментировала и не училась с нуля; предиктивное моделирование и спортивное программирование поставляются вместе с Kapso. Запуск начинается с доменного harness и пустого банка уроков, и со временем harness несёт в себе банк, накопивший то, что сработало в этом окружении. Напишите свой начальный доменный harness, свои инструкции, код, ограничители и проверки, с помощью `/kapso:harness` в плагине или `kapso harness` в CLI. |

## 🚀 Быстрый старт

### Установка

**0. Вы в кодинг-агенте?** Установите плагин Kapso и позвольте ему пройти
с вами шаги ниже:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Затем `/kapso:setup` устанавливает и проверяет, а `/kapso:build <what you want>`
запускает первую кампанию (в Codex: `$kapso-setup`, `$kapso-build <what you want>`). См. [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Предварительные требования.** Kapso выполняет инференс через CLI кодинг-агентов
(запасного варианта с прямым API нет), поэтому, прежде чем что-либо заработает, нужны Node.js и оба
CLI агентов с выполненным входом:

```bash
# Node.js 18+ (https://nodejs.org), затем:
npm install -g @openai/codex            # исследование, судейство, утилиты
codex login

npm install -g @anthropic-ai/claude-code  # генерация идей + реализация (хост по умолчанию)
claude auth login
```

Ключ OpenAI необязателен: только граф знаний и семантический поиск
по прошлым экспериментам (`models.embedding` в вашем конфиге) создают эмбеддинги текста, а
генерация его никогда не использует. Когда вы включаете любой из них:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Подключите Leeroopedia (рекомендуется).** Один ключ даёт каждой сессии
кампании [Leeroopedia](https://leeroopedia.com), размещённую в облаке базу знаний о
фреймворках ML и ИИ: как фреймворк ведёт себя на самом деле, ревью планов,
задокументированные диапазоны гиперпараметров, диагнозы сбоев. Устанавливать ничего не нужно;
новая учётная запись в [панели Leeroopedia](https://app.leeroopedia.com/dashboard)
поставляется с $20 бесплатного кредита.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Установите пакет** (Python 3.10+):

```bash
pip install leeroo-kapso
```

Пакет называется `leeroo-kapso`. Пакет PyPI с именем `kapso` является не связанным с нами
инструментом для WhatsApp, который устанавливает собственную команду `kapso` и затеняет эту;
если он присутствует, сначала выполните `pip uninstall kapso`.

**4. Проверьте установку:**

```bash
kapso doctor
```

`doctor` сообщает, что на самом деле нужно **вашему конфигу**, и называет точное
исправление для всего недостающего. Требования следуют за конфигом, поэтому у установки,
использующей только codex, никогда не потребуют `claude`. Сузьте до одного глагола, чтобы увидеть
требования только этого глагола:

```bash
kapso doctor build             # или learn / deploy / experience
```

Это те же проверки, которые сам глагол выполняет, прежде чем что-либо делать:
`kapso.build(...)` падает за секунды на отсутствующем CLI, а не глубоко
внутри сессии. Пункты, помеченные `[-- ]`, необязательны; они ограничивают возможности,
которые могут вам не понадобиться (docker, Weaviate, Neo4j, цели развёртывания).

**Или поручите всё это своему кодинг-агенту:** вставьте
[промпт настройки](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
в Claude Code, Codex или OpenCode.

**Хранилища, которые нужны индексу (необязательно)**: `kapso experience add --repo`
и `--research` загружают корпус в локальные Weaviate + Neo4j. Из
клона исходного кода:

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

Устаревший адаптер aider является дополнением (`pip install "leeroo-kapso[aider]"`,
Python <3.13); агентам claude/codex по умолчанию дополнения не нужны.

### Базовое использование

Основным глаголам не нужно ничего сверх предварительных требований выше:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # для начала граф знаний не нужен

# Build: решение через экспериментирование. Кампания печатает
# `status: <path>` при запуске; следите за ней вживую из другого терминала с помощью
#     kapso watch ./campaign
# Если сессии нужно что-то, что можете предоставить только вы (ключ API, файл),
# кампания приостанавливается, а `kapso inbox reply` возобновляет её:
#     https://docs.leeroo.com/docs/build/inbox
# Поделитесь результатом в виде Markdown для pull request или issue с помощью
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # или опустите, чтобы начать с нуля
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Учитесь на только что выполненной кампании: извлеките траекторию, оцените
# уроки и положите в банк карточки знаний, оценённые по свидетельствам. Банк (локальный
# git-репозиторий) создаётся автоматически при первом использовании; уроки остаются на вашей
# машине, пока вы ими не поделитесь:
#     kapso experience share <git-url>
# после чего каждый learn() отправляет банк туда.
lesson = kapso.learn(solution)
print(lesson.explain())

# Соберите снова, предоставив карточки, которые только что заработаны: сборке предоставляется ровно
# тот опыт, который она загружает.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

Когда хранилища запущены, вы также можете загрузить внешние знания в
индекс и предоставлять его тем же способом:

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
источники, в которых будете искать, а не все источники, которые у вас есть.

А чтобы превратить решение в работающее программное обеспечение:

```python
software = kapso.deploy(solution, target="local")   # или docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # вызов, который доказало развёртывание
software.stop()
```

### Как ограничить кампанию?

Используйте `max_iterations`, `time_budget_minutes` и `cost_budget`, чтобы ограничить число итераций, прошедшее время и приблизительные расходы на модели в USD.
Для платформенных затрат включите [выбор с учётом стоимости](https://docs.leeroo.com/docs/evolve/cost-aware-selection) в `search_strategy.params.cost` с вашей функцией оценки стоимости, `cap_per_candidate` и `campaign_budget`; его `weight` также позволяет выбору предпочитать более дешёвых кандидатов.

### Выбор моделей

Каждая модель, которую использует Kapso, названа в одном файле конфигурации. Упакованный конфиг
по умолчанию запускает сессии сборки на `claude-opus-5`, бригады обучения на
`claude-fable-5`, а роли codex на `gpt-5.6-sol`, но доступ к моделям
зависит от подписки (тариф может ограничивать одну модель, обслуживая другую).
Чтобы работать на других моделях, скопируйте упакованный конфиг, отредактируйте и укажите
Kapso на свой:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# например, запустить бригады обучения на opus вместо fable:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Роли инференса, работающие только с текстом, могут перейти на `openai_compatible` Chat Completions.
Задайте модель и конечную точку в своём конфиге; учётные данные остаются в окружении.
Загрузите конфиг с помощью `Kapso(config=...)` или `kapso build --config ...`.
Написание кода и веб-исследование по-прежнему требуют агентов с соответствующими инструментами.
См. [пример конфигурации и ограничения](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Перед долгим запуском проверьте каждую модель, названную в вашем конфиге, на ваших
реальных подписках: модель, которую ваш логин не может обслужить, падает здесь за
секунды, а не через часы после начала запуска. Лимит использования модели, которую вы можете
обслужить, однотокенной пробе не виден:

```bash
kapso doctor --models                            # упакованный конфиг
kapso doctor --models --config kapso-config.yaml # ваш
kapso doctor learn --models                      # только бригады обучения
```

Чтобы сделать эту пробу частью каждого вызова, а не тем, о чём нужно не забыть,
включите её в своём конфиге. Она стоит один одноразовый токен на каждую уникальную
пару `{cli, model}`, что оправдано перед многочасовым `learn()` без присмотра:

```yaml
preflight:
  enabled: true            # статические проверки перед каждым глаголом (по умолчанию)
  live_model_probe: true   # + однотокенные пробы моделей (по умолчанию: false)
```

Замена моделей меняет и *темп*: ограничения `timeout_minutes` для бригад в
конфиге были откалиброваны на моделях по умолчанию, и заменяющей модели, которая
рассуждает дольше, может потребоваться их повышение.

Подробные шаги интеграции см. в руководствах [Быстрый старт](https://docs.leeroo.com/docs/first-build) и [Установка](https://docs.leeroo.com/docs/installation).

## Поддерживаемые бенчмарки

| Бенчмарк | Описание |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | Бенчмарк OpenAI по ML-инженерии: полные соревнования по табличным данным, зрению, тексту и аудио, от сырых данных до оценённой посылки |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Бенчмарк Sakana AI по алгоритмической оптимизации: проектирование, реализация и итерация конкурсных эвристик в ходе многочасовых поисков |
| [**RelBench**](../../benchmarks/relbench/README.md) | Бенчмарк Stanford для предиктивного ML на реляционных данных: классификация, регрессия и рекомендации напрямую из многотабличных баз данных Amazon, H&M, Stack Exchange, клинических испытаний и других |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | Олимпиадный ML на время по зрению, языку и оптимизации: задачи, составленные экспертами, конкурсное оборудование, ноль человеческой помощи |

У каждого бенчмарка также есть страница документации, описывающая, как его запускать, его опции CLI и что содержит вывод:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## Примеры

Каждый пример представляет собой задачу: бриф, стартовый репозиторий, данные или способ их получить и драйвер не более чем в двенадцать строк; в [examples/README.md](../../examples/README.md) указаны минуты, единственная команда и то, записан ли результат.

| Пример | Что показывает |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | Ваша первая сборка: имеющийся оценщик, базовый уровень, целевое значение, запись результата |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | Без оценщика: первый узел кампании строит его по протоколу из брифа, и каждый кандидат измеряется относительно него |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | Пайплайн dbt на DuckDB, экспорты которого содержат дефекты: контракт данных, доведённый до соблюдения |
| [**trading**](../../examples/trading/README.md) | Harness на сборке: сессия настройки, страницы и хуки в каждой сессии, верификация на каждую попытку |
| [**end to end**](../../examples/end-to-end/README.md) | Build, разбор, learn, снова build с предоставленным уроком, deploy: табличная модель Kaggle от начала до конца |
| [**prompt harness**](../../examples/prompt-harness/README.md) | Промпт как объект тестирования, оцениваемый на задачах AIME |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | Визуально-языковой экстрактор, оцениваемый по эталонным таблицам |
| [**torch performance**](../../examples/torch-performance/README.md) | Более быстрый модуль на CUDA в пределах численного допуска |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | Многоголовое внимание на CUDA: flash attention или Triton |

## 📚 Документация и поддержка

- **Полная документация**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Установка](https://docs.leeroo.com/docs/installation): CLI кодинг-агентов, пакет и `kapso doctor`
  - [Быстрый старт](https://docs.leeroo.com/docs/first-build): ваша первая кампания
  - [Справочник по CLI](https://docs.leeroo.com/docs/reference/cli): каждая команда, флаг и значение по умолчанию
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): каждый публичный метод `Kapso`
  - [Конфигурация](https://docs.leeroo.com/docs/reference/configuration): каждый ключ в `config.yaml`
  - [Что производит learn](https://docs.leeroo.com/docs/learn/what-learn-produces): банк уроков, каждый с показателем достоверности, границей и свидетельствами, и [режим разработки](https://docs.leeroo.com/docs/learn/development-mode)
  - [Умное экспериментирование](https://docs.leeroo.com/docs/evolve/overview): дерево экспериментов и его модули, и [что происходит внутри одного эксперимента](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Опыт](https://docs.leeroo.com/docs/experience): ваш банк, Leeroopedia, репозитории и статьи, превращённые в знания с возможностью поиска
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal и другие цели развёртывания
  - [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills): команды, навыки и MCP-сервер для Claude Code, Codex и Cursor
- **Из вашего кодинг-агента**: документация запускает MCP-сервер по адресу `https://docs.leeroo.com/mcp`, и каждая страница отдаётся как Markdown; см. [llms.txt](https://docs.leeroo.com/llms.txt) и [Kapso в вашем кодинг-агенте](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Поручите своему кодинг-агенту запускать Kapso**: плагин в [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) содержит десять команд, шесть навыков и MCP-сервер, одна строка установки на каждый хост (шаг 0 выше); OpenCode загружает навыки как папку (`ln -s <clone>/plugin/skills .opencode/skills`). См. [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills).
- **Дайте своему кодинг-агенту инструменты знаний Kapso**: `kapso-mcp`, устанавливаемый вместе с пакетом, предоставляет веб-исследование, память репозитория, историю экспериментов кампании и банк уроков той сессии, в которой вы работаете; см. [MCP-шлюзы](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; затем `startup_timeout_sec = 60` в разделе [mcp_servers.kapso] в ~/.codex/config.toml (десяти секунд Codex мало на загруженной машине)
  ```
- **Сообщество**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Сайт**: [leeroo.com](https://leeroo.com)


## Kapso для предприятий

В корпоративной среде агенты Kapso поставляются с начальными harness по инженерии данных и ИИ и банком уроков и самостоятельно превращают бизнес-цели в работающие системы ИИ и данных, исследуя артефакты компании, репозитории, пайплайны, источники данных и многое другое, попутно выстраивая корпоративный ландшафт данных. Чтобы подключить Kapso для задач вашего предприятия, [свяжитесь с нами](https://leeroo.com/contact-us).

## Участие в разработке

Мы приветствуем вклад! Подробности о том, как начать, см. в нашем [руководстве для участников](../../CONTRIBUTING.md).

## Участники

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

Сделано с помощью [contrib.rocks](https://contrib.rocks).
