<!-- Russian translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Движок, который проводит эксперименты, развёртывает лучшее решение и учится на каждом запуске</h4>
<p align="center"><sub>умное экспериментирование · цикл обучения · развёртывание · доменная адаптация</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <b>Русский</b> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">Подробнее</a> ·
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
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Если вы ИИ-агент и смотрите этот репозиторий:** плагин Kapso с навыками для
> настройки, постановки цели, написания harness и разбора запуска устанавливается
> одной строкой: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Новости

- **IOAI 2026 · трек AI Model Track.** [Международная олимпиада по ИИ](https://ioai-official.org) является IMO эпохи ИИ: 471 участник из 108 стран и территорий, шесть задач, составленных экспертами, при бюджете в один GPU. В 2026 году она открыла [IOAI²](https://ioai-official.org/ai-model-track/), где ИИ-системы сдают *тот же* экзамен за две 6-часовые сессии без присмотра. Как только пошёл отсчёт времени, ни один человек не вправе что-либо решать, исправлять или улучшать. Kapso выступил как один из 14 ИИ-участников-основателей.
  - 🎓 **Обошёл всех 471 человека**: в сумме 536.07, выше любого участника в зале.
  - 🏆 **IOAI² Grand Master Trophy**: в числе 3 лучших среди всех участвовавших ИИ-систем. [Полные результаты →](../../benchmarks/ioai2026/README.md) · [Обзор трека от IOAI →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Первое место во всех трёх таблицах лидеров [RelBench](../../benchmarks/relbench/README.md)**: в бенчмарке Stanford для предиктивного ML на реляционных данных Kapso возглавляет официальную таблицу лидеров по классификации (81.2 AUROC на 12 задачах против 79.3 у [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) и 79.0 у RT-J от Stanford STAR), регрессии (0.2476 NMAE на 9 задачах против 0.2711 у RT-J и 0.2847 у KumoRFM-v2, ошибка на 8.7% меньше) и рекомендациям (18.4 MAP на 10 задачах против 9.3 у ID-GNN, лучшего из остальных участников). Каждое число соответствует записи в [официальной таблице лидеров RelBench](https://star-project.stanford.edu/relbench/leaderboard/).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)**: вики наших уроков в областях данных и ML, около 100k страниц, извлечённых из экспериментов над статьями и репозиториями; она обслуживает около 6M запросов в месяц, примерно 28% из них приходят от ИИ-агентов. Kapso подключается к ней по MCP: его сессии используют её при генерации идей и реализации, чтобы искать знания, строить планы и диагностировать сбои.
- **[Агенты Moltbook 🦞](https://www.moltbook.com/)**: создавайте ИИ-агентов, которые оптимизируют других агентов и спорят на Moltbook! [Начать →](../../moltbook_bot/README.md)
- **Технический отчёт**: наш технический отчёт уже доступен! [Читать статью](https://arxiv.org/abs/2601.21526)
- **#1 на [MLE-Bench](../../benchmarks/mle/README.md)**: KAPSO занял первое место среди систем с открытым исходным кодом на ML-соревнованиях Kaggle (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **#1 на [ALE-Bench](../../benchmarks/ale/README.md)**: KAPSO занял первое место на задачах алгоритмического поиска с длинным горизонтом (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## Что такое Kapso?

Kapso представляет собой движок, который проводит эксперименты, развёртывает лучшее решение и учится на каждом запуске. Получив цель, начальные критерии успеха и метрики оценки, он исследует задачу, проводит эксперименты, выбирая лучшие идеи, иногда даёт обратную связь для улучшения метрик оценки и критериев успеха, развёртывает лучшее решение на нужном стеке и учится на каждом запуске. Он ведёт банк уроков, у каждого из которых есть показатель достоверности и граница применимости, поэтому знания со временем не просто накапливаются, а приумножаются.

Мы создавали ИИ-системы в крупных компаниях, включая LinkedIn и Meta AI, и развёртывали их для продуктов с миллионами пользователей. Kapso воплощает этот рабочий процесс в виде движка: с открытым исходным кодом, от [Leeroo](https://leeroo.com), опубликован на PyPI как [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### Четыре части

| Часть | Глагол | Что делает |
|------|------|--------------|
| [**Умное экспериментирование**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Цикл экспериментирования состоит из набора специализированных модулей с одной общей структурой данных, деревом экспериментов: каждый узел фиксирует гипотезу, реализацию (git-ветку и технические трудности), израсходованный бюджет и расстояние до цели. Генерация идей опирается на предыдущие эксперименты и банк уроков и предлагает идеи, каждая из которых указывает, чем она отличается от конкретных прежних экспериментов и какие уроки берёт на вооружение; реранкер ранжирует их по ожидаемой отдаче относительно целевого показателя, соотнося её со временем и бюджетом. Модуль реализации строит решение на ветке, ответвлённой от родительского узла, а верификатор проверяет узел на соответствие правилам безопасности и критериям успеха, решает, действителен ли полученный балл и достигнута ли цель. Критерии успеха и тесты могут меняться в ходе запуска: модули предлагают обоснованные обновления, а верификатор утверждает или отклоняет их. [Как это работает →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Цикл обучения**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Каждый урок фиксирует фрагмент знания, показатель достоверности, граничные условия его применимости и свидетельства, стоящие за его обновлениями. После каждого запуска `learn()` обновляет достоверность и границы использованных уроков по их влиянию на результат, индуктивно выводит уроки из похожих трасс или дедуктивно выводит урок из одного свидетельства и его механизма, объединяет пересекающиеся уроки и проводит A/B-тест обновлённых уроков, запуская один и тот же build с ними и без них, так что обновления измеряются, а не угадываются. Генерация идей и реализация следующего build читают достоверность и границы каждого урока, чтобы решить, когда на него полагаться, а не просто применяют RAG к знаниям. [Experience →](https://docs.leeroo.com/docs/experience) определяет, что выдаётся build: ваш банк, банк, общий для вашей команды, Leeroopedia или корпус, загруженный из репозитория или собранный исследованием по заданному вопросу. |
| [**Развёртывание**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Превращает лучшее решение в дереве в работающее ПО на целевом стеке с помощью укороченной версии цикла экспериментирования: сессия адаптирует копию узла, развёртывает её и демонстрирует проверки работоспособности и рабочий вызов. Docker, Modal, BentoCloud и LangGraph Platform поставляются в комплекте, а собственный целевой стек можно добавить в harness. |
| [**Доменная адаптация**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harness в Kapso представляет собой доменный пакет: знания и рабочие процессы для области вроде предиктивного моделирования, чтобы build не экспериментировал и не учился с нуля; harness для предиктивного моделирования и harness для спортивного программирования поставляются вместе с Kapso. Чтобы привнести собственные доменные знания, напишите harness через интерфейс, который определяет Kapso: ваши инструкции, код, защитные ограничения и проверки, созданные с помощью `/kapso:harness` в плагине или `kapso harness` в CLI. |

## 🚀 Быстрый старт

### Установка

**0. Работаете в кодинг-агенте?** Установите плагин Kapso, и он пройдёт
шаги ниже вместе с вами:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Затем `/kapso:setup` выполняет установку и проверку, а `/kapso:build <what you want>`
запускает первую кампанию (в Codex: `$kapso-setup`, `$kapso-build <what you want>`). См. [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Предварительные требования.** Kapso выполняет инференс через CLI кодинг-агентов
(запасного варианта с прямым обращением к API нет), поэтому, прежде чем что-либо
заработает, нужны Node.js и оба CLI агентов с выполненным входом:

```bash
# Node.js 18+ (https://nodejs.org), затем:
npm install -g @openai/codex            # исследование, судейство, утилиты
codex login

npm install -g @anthropic-ai/claude-code  # генерация идей + реализация (хост по умолчанию)
claude auth login
```

Ключ OpenAI необязателен: текст векторизуют только граф знаний и семантический поиск
по прошлым экспериментам (`models.embedding` в вашей конфигурации), а
генерация его никогда не использует. Если вы включаете что-то из них:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Подключите Leeroopedia (рекомендуется).** Один ключ даёт каждой сессии
кампании доступ к [Leeroopedia](https://leeroopedia.com), размещённой в облаке базе знаний о
фреймворках ML и ИИ: как фреймворк ведёт себя на самом деле, разборы планов,
документированные диапазоны гиперпараметров, диагностика сбоев. Устанавливать ничего не нужно;
новый аккаунт в [панели управления Leeroopedia](https://app.leeroopedia.com/dashboard)
получает $20 бесплатного кредита.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Установите пакет** (Python 3.10+):

```bash
pip install leeroo-kapso
```

Пакет называется `leeroo-kapso`. Пакет PyPI с именем `kapso` является посторонним
инструментом для WhatsApp, который устанавливает собственную команду `kapso` и перекрывает эту;
если он установлен, сначала выполните `pip uninstall kapso`.

**4. Проверьте установку:**

```bash
kapso doctor
```

`doctor` сообщает, что на самом деле нужно **вашей конфигурации**, и называет точное
исправление для всего, чего не хватает. Требования следуют из конфигурации, поэтому при
настройке целиком на codex `claude` никогда не потребуется. Сузьте проверку до одного глагола,
чтобы увидеть требования только этого глагола:

```bash
kapso doctor build             # или learn / deploy / experience
```

Это те же проверки, которые сам глагол выполняет перед началом работы:
`kapso.build(...)` при отсутствии CLI завершается с ошибкой за секунды, а не где-то в глубине
сессии. Пункты с пометкой `[-- ]` необязательны; они ограничивают возможности,
которые вам могут и не понадобиться (docker, Weaviate, Neo4j, цели развёртывания).

**Или поручите всё это своему кодинг-агенту:** вставьте
[промпт для настройки](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
в Claude Code, Codex или OpenCode.

**Хранилища, которые нужны индексу (необязательно)**: `kapso experience add --repo`
и `--research` загружают корпус в локальные Weaviate + Neo4j. Из
клона исходного репозитория:

```bash
bash scripts/start_infra.sh   # запускает оба через docker
```

**Из исходного кода (для разработки)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

Устаревший адаптер aider поставляется как дополнительная зависимость (`pip install "leeroo-kapso[aider]"`,
Python <3.13); агентам по умолчанию claude/codex дополнительные зависимости не нужны.

### Базовое использование

Основному циклу не нужно ничего, кроме предварительных требований выше:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # для начала граф знаний не нужен

# Build: решение через экспериментирование. При запуске кампания печатает
# `status: <path>`; наблюдайте за ней вживую из другого терминала командой
#     kapso watch ./campaign
# Если сессии нужно то, что можете предоставить только вы (API-ключ, файл),
# кампания приостанавливается, а `kapso inbox reply` возобновляет её:
#     https://docs.leeroo.com/docs/build/inbox
# Поделитесь результатом в формате Markdown для пул-реквеста или issue командой
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # или опустите, чтобы начать с нуля
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Учимся на только что проведённой кампании: разбираем траекторию, оцениваем
# уроки и кладём в банк карточки знаний, взвешенные по свидетельствам. Банк (локальный
# git-репозиторий) создаётся автоматически при первом использовании; уроки остаются на
# вашей машине, пока вы ими не поделитесь:
#     kapso experience share <git-url>
# после чего каждый learn() отправляет банк туда.
lesson = kapso.learn(solution)
print(lesson.explain())

# Снова build, теперь с только что заработанными карточками: build получает ровно
# тот опыт, который загружает.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

Когда хранилища запущены, можно также загрузить внешние знания в
индекс и выдавать их тем же способом:

```python
from kapso import Kapso

kapso = Kapso()

# Один раз: репозиторий или результаты веб-исследования по вопросу загружаются в индекс
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Каждый build, загрузивший индекс, ищет по нему; Leeroopedia работает по своему ключу
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Замечание о бюджете: загрузка извлекает всё, что позволяет материал.
Один вопрос может превратиться в десятки связанных страниц и многочасовую
загрузку; время загрузки растёт с объёмом извлекаемого содержания, поэтому добавляйте
те источники, по которым будете искать, а не все, что у вас есть.

А чтобы превратить решение в работающее ПО:

```python
software = kapso.deploy(solution, target="local")   # или docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # вызов, работоспособность которого подтвердило развёртывание
software.stop()
```

### Как ограничить кампанию?

Используйте `max_iterations`, `time_budget_minutes` и `cost_budget`, чтобы ограничить число итераций, затраченное время и, по мере возможности, расходы на модели в USD.
Для расходов на платформу включите [отбор с учётом стоимости](https://docs.leeroo.com/docs/evolve/cost-aware-selection) в `search_strategy.params.cost`, указав свой оценщик стоимости, `cap_per_candidate` и `campaign_budget`; его параметр `weight` также позволяет отбору предпочитать более дешёвых кандидатов.

### Выбор моделей

Каждая модель, которую использует Kapso, указана в одном файле конфигурации. Конфигурация
по умолчанию из пакета запускает сессии build на `claude-opus-5`, группы обучения на
`claude-fable-5`, а роли codex на `gpt-5.6-sol`, но доступ к моделям
зависит от подписки (тариф может ограничивать одну модель и при этом обслуживать другую).
Чтобы работать на других моделях, скопируйте конфигурацию из пакета, отредактируйте её и
укажите Kapso на свою:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# например, запустить группы обучения на opus вместо fable:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Роли с чисто текстовым инференсом могут перейти на Chat Completions через `openai_compatible`.
Задайте модель и эндпоинт в конфигурации; учётные данные остаются в окружении.
Загрузите конфигурацию через `Kapso(config=...)` или `kapso build --config ...`.
Для написания кода и веб-исследований по-прежнему нужны агенты с соответствующими инструментами.
См. [пример конфигурации и ограничения](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Перед долгим запуском проверьте каждую модель, указанную в конфигурации, на
ваших реальных подписках: модель, которую ваш логин не может обслужить, даст сбой здесь за
секунды, а не спустя часы после начала запуска. Лимит использования модели, которая вам
доступна, не виден пробному запросу в один токен:

```bash
kapso doctor --models                            # конфигурация из пакета
kapso doctor --models --config kapso-config.yaml # ваша
kapso doctor learn --models                      # только группы обучения
```

Чтобы эта проверка стала частью каждого вызова, а не тем, о чём нужно помнить,
включите её в конфигурации. Она стоит один одноразовый токен на каждую отдельную
пару `{cli, model}`, и это оправданно перед многочасовым `learn()` без
присмотра:

```yaml
preflight:
  enabled: true            # статические проверки перед каждым глаголом (по умолчанию)
  live_model_probe: true   # + пробные запросы к моделям в один токен (по умолчанию: false)
```

Замена моделей меняет и *темп*: ограничения `timeout_minutes` для групп в
конфигурации были откалиброваны на моделях по умолчанию, и для подставленной модели, которая
рассуждает дольше, их, возможно, придётся поднять.

Подробные шаги интеграции см. в руководствах [Быстрый старт](https://docs.leeroo.com/docs/first-build) и [Установка](https://docs.leeroo.com/docs/installation).

## Примеры

Каждый пример представляет собой задачу: бриф, исходный репозиторий, данные или способ их получить и драйвер не более чем из двенадцати строк; в [examples/README.md](../../examples/README.md) указаны время в минутах, единственная команда и то, записан ли результат.

| Пример | Что показывает |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | Ваш первый build: оценщик уже есть, базовый результат, целевой показатель, запись |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | Оценщика нет: первый узел кампании строит его по протоколу из брифа, и каждый кандидат измеряется по нему |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | Пайплайн dbt на DuckDB, выгрузки которого содержат дефекты: контракт данных доведён до соблюдения |
| [**trading**](../../examples/trading/README.md) | Harness в build: сессия настройки, страницы и хуки в каждой сессии, верификация каждой попытки |
| [**full loop**](../../examples/full-loop/README.md) | Build, разбор, learn, снова build с выданным уроком, deploy: табличная модель Kaggle от начала до конца |
| [**prompt harness**](../../examples/prompt-harness/README.md) | Промпт как объект тестирования, с оценкой на задачах AIME |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | Экстрактор на основе визуально-языковой модели, с оценкой по эталонным таблицам |
| [**torch performance**](../../examples/torch-performance/README.md) | Более быстрый модуль на CUDA в пределах численного допуска |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | Многоголовое внимание на CUDA: flash attention или Triton |

## Поддерживаемые бенчмарки

| Бенчмарк | Описание |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | Бенчмарк OpenAI по ML-инженерии: полноценные соревнования по табличным данным, компьютерному зрению, тексту и аудио, от сырых данных до оценённой отправки решения |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Бенчмарк Sakana AI по алгоритмической оптимизации: проектирование, реализация и итеративная доработка соревновательных эвристик в ходе многочасового поиска |
| [**RelBench**](../../benchmarks/relbench/README.md) | Бенчмарк Stanford для предиктивного ML на реляционных данных: классификация, регрессия и рекомендации напрямую по многотабличным базам данных Amazon, H&M, Stack Exchange, клинических испытаний и других источников |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | Олимпиадный ML на время в областях зрения, языка и оптимизации: задачи, составленные экспертами, соревновательное оборудование, без какой-либо помощи человека |

У каждого бенчмарка также есть страница документации о том, как его запустить, о его параметрах CLI и о содержимом вывода:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 Документация и поддержка

- **Полная документация**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Установка](https://docs.leeroo.com/docs/installation): CLI кодинг-агентов, пакет и `kapso doctor`
  - [Быстрый старт](https://docs.leeroo.com/docs/first-build): ваша первая кампания
  - [Справочник по CLI](https://docs.leeroo.com/docs/reference/cli): каждая команда, флаг и значение по умолчанию
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): каждый публичный метод `Kapso`
  - [Конфигурация](https://docs.leeroo.com/docs/reference/configuration): каждый ключ в `config.yaml`
  - [Что создаёт learn](https://docs.leeroo.com/docs/learn/what-learn-produces): банк уроков, у каждого из которых есть показатель достоверности, граница и свидетельства, а также [режим разработки](https://docs.leeroo.com/docs/learn/development-mode)
  - [Цикл экспериментирования](https://docs.leeroo.com/docs/evolve/overview): дерево экспериментов и его модули, а также [что происходит внутри одного эксперимента](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience): ваш банк, Leeroopedia, репозитории и статьи, превращённые в знания с поиском
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal и другие цели
  - [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills): команды, навыки и MCP-сервер для Claude Code, Codex и Cursor
- **Из вашего кодинг-агента**: у документации есть MCP-сервер по адресу `https://docs.leeroo.com/mcp`, и каждая страница отдаётся в формате Markdown, см. [llms.txt](https://docs.leeroo.com/llms.txt) и [Kapso в вашем кодинг-агенте](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Пусть ваш кодинг-агент запускает Kapso**: плагин в [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) содержит десять команд, шесть навыков и MCP-сервер, одна строка установки на каждый хост (шаг 0 выше); OpenCode загружает навыки как папку (`ln -s <clone>/plugin/skills .opencode/skills`). См. [Плагин Kapso](https://docs.leeroo.com/docs/coding-agent-skills).
- **Дайте своему кодинг-агенту инструменты знаний Kapso**: `kapso-mcp`, устанавливаемый вместе с пакетом, предоставляет веб-исследования, память репозитория, историю экспериментов кампании и банк уроков той сессии, в которой вы работаете, см. [MCP-шлюзы](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; затем `startup_timeout_sec = 60` в секции [mcp_servers.kapso] в ~/.codex/config.toml (десяти секунд Codex на загруженной машине не хватает)
  ```
- **Сообщество**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Сайт**: [leeroo.com](https://leeroo.com)


## Kapso для предприятий

Наш корпоративный продукт построен поверх этого движка и развёрнут в запутанных корпоративных средах, где ни один человек не способен исследовать и понять весь ландшафт данных: мы перевели компанию со старой платформы данных на новый lakehouse, охватив более 30 семейств систем-источников, тысячи таблиц и преобразований и сотни BI-отчётов. Каждая задача, над которой он работает, пополняет банк уроков о ваших системах, ваших данных и ваших правилах, добытых тяжёлым трудом, поэтому чем дольше он работает, тем лучше разбирается в вашей компании. Чтобы внедрить Kapso для ваших сложных корпоративных задач, [свяжитесь с нами](https://leeroo.com/contact-us).

## Переводы

README доступен на других языках: воспользуйтесь переключателем вверху или просмотрите [`docs/translations/`](../../docs/translations/). Каждый перевод генерируется из README.md с помощью `scripts/translate_readme.py` и поддерживается в актуальном состоянии скриптом `scripts/check_readme_translations.py`, который завершается с ошибкой, если перевод расходится с README.md в ссылке, числе или строке кода. Чтобы улучшить перевод, откройте пул-реквест к файлу в этой папке.

## Участие в разработке

Мы рады вашему вкладу! Подробности о том, как начать, см. в нашем [руководстве для участников](../../CONTRIBUTING.md).

## Цитирование

Если вы используете Kapso в своей работе, пожалуйста, цитируйте технический отчёт:

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
