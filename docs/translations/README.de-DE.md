<!-- German translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Eine Engine, die Experimente durchführt, die beste Lösung bereitstellt und aus jedem Lauf lernt</h4>
<p align="center"><sub>intelligentes Experimentieren · Lernschleife · Bereitstellung · Domänenanpassung</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <b>Deutsch</b> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">Mehr erfahren</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Discord beitreten</a> ·
  <a href="https://leeroo.com">Website</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  Wenn dir dieses Projekt gefällt, unterstütze uns bitte mit einem Stern ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Wenn du ein KI-Agent bist, der sich dieses Repo ansieht:** Das Kapso-Plugin mit den Skills für
> das Einrichten, das Formulieren einer Zielvorgabe, das Schreiben eines Harness und das Prüfen eines Laufs lässt sich
> in einer Zeile installieren: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Neuigkeiten

- **IOAI 2026 · der AI Model Track.** Die [International Olympiad in AI](https://ioai-official.org) ist die IMO des KI-Zeitalters: 471 Teilnehmende aus 108 Ländern und Territorien, sechs von Experten entworfene Aufgaben mit dem Budget einer einzelnen GPU. 2026 eröffnete sie [IOAI²](https://ioai-official.org/ai-model-track/), wo KI-Systeme *dieselbe* Prüfung in zwei unbeaufsichtigten 6-stündigen Sitzungen ablegen. Sobald die Uhr läuft, darf kein Mensch etwas lösen, korrigieren oder verbessern. Kapso trat als einer von 14 Founding AI Participants an.
  - 🎓 **Mehr Punkte als alle 471 Menschen**: 536.07 insgesamt, mehr als alle Teilnehmenden im Saal.
  - 🏆 **IOAI² Grand Master Trophy**: Top 3 unter allen angetretenen KI-Systemen. [Vollständige Ergebnisse →](../../benchmarks/ioai2026/README.md) · [Der Rückblick der IOAI auf den Track →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Erster Platz auf allen drei Leaderboards von [RelBench](../../benchmarks/relbench/README.md)**: Auf Stanfords Benchmark für prädiktives ML über relationalen Daten führt Kapso das offizielle Leaderboard an, bei Klassifikation (81.2 AUROC über 12 Aufgaben, gegenüber 79.3 für [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) und 79.0 für RT-J von Stanford STAR), Regression (0.2476 NMAE über 9 Aufgaben, gegenüber 0.2711 für RT-J und 0.2847 für KumoRFM-v2, 8.7% weniger Fehler) und Empfehlungen (18.4 MAP über 10 Aufgaben, gegenüber 9.3 für ID-GNN, den besten anderen Eintrag). Jede Zahl ist ein Eintrag im [offiziellen RelBench-Leaderboard](https://star-project.stanford.edu/relbench/leaderboard/).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)**: ein Wiki unserer Lektionen in den Bereichen Daten und ML, rund 100k Seiten, gewonnen durch Experimentieren an Papern und Repositories, mit etwa 6M Anfragen pro Monat, rund 28% davon von KI-Agenten. Kapso verbindet sich über MCP damit: Seine Sitzungen nutzen es während Ideenfindung und Implementierung, um Wissen zu durchsuchen, Pläne zu erstellen und Fehler zu diagnostizieren.
- **[Moltbook-Agenten 🦞](https://www.moltbook.com/)**: Baue KI-Agenten, die andere Agenten optimieren und auf Moltbook debattieren! [Loslegen →](../../moltbook_bot/README.md)
- **Technischer Bericht**: Unser technischer Bericht ist jetzt verfügbar! [Paper lesen](https://arxiv.org/abs/2601.21526)
- **#1 auf [MLE-Bench](../../benchmarks/mle/README.md)**: KAPSO erreichte den Spitzenplatz unter den Open-Source-Systemen bei Kaggle-ML-Wettbewerben (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **#1 auf [ALE-Bench](../../benchmarks/ale/README.md)**: KAPSO erreichte den Spitzenplatz bei algorithmischen Entdeckungsproblemen mit langem Zeithorizont (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## Was ist Kapso?

Kapso ist eine Engine, die Experimente durchführt, die beste Lösung bereitstellt und aus jedem Lauf lernt. Ausgehend von einer Zielvorgabe, anfänglichen Erfolgskriterien und Evaluationsmetriken untersucht es das Problem, führt Experimente durch, indem es die besten Ideen auswählt, gibt gelegentlich Feedback zur Verbesserung der Evaluationsmetriken und Erfolgskriterien, stellt die beste Lösung auf dem gewünschten Stack bereit und lernt aus jedem Lauf. Es führt eine Lektionsbank, jede Lektion mit einem Glaubwürdigkeitswert und einer Grenze dafür, wo sie gilt, sodass sich das Wissen mit der Zeit potenziert, statt sich nur anzuhäufen.

Wir haben KI-Systeme in großen Unternehmen gebaut, darunter LinkedIn und Meta AI, und sie für Produkte mit Millionen von Nutzern bereitgestellt. Kapso ist dieser Workflow als Engine: Open Source, von [Leeroo](https://leeroo.com), auf PyPI veröffentlicht als [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### Die vier Teile

| Teil | Verb | Was er tut |
|------|------|--------------|
| [**Intelligentes Experimentieren**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Die Experimentierschleife besteht aus spezialisierten Modulen, die sich eine Datenstruktur teilen, einen Experimentbaum: Jeder Knoten hält die Hypothese, die Implementierung (git-Branch und technische Schwierigkeiten), das verbrauchte Budget und den Abstand zur Zielvorgabe fest. Die Ideenfindung stützt sich auf frühere Experimente und die Lektionsbank, um Ideen vorzuschlagen, von denen jede angibt, worin sie sich von bestimmten früheren Experimenten unterscheidet und welche Lektionen sie übernimmt; ein Reranker ordnet sie nach dem erwarteten Ertrag gegenüber dem Zielwert, abgewogen gegen Zeit und Budget. Der Implementierer baut auf einem Branch auf, der vom Elternknoten abgezweigt wurde, und ein Verifizierer prüft den Knoten gegen die Sicherheitsregeln und die Erfolgskriterien und entscheidet, ob der Score gültig ist und ob die Zielvorgabe erreicht ist. Erfolgskriterien und Tests können sich während eines Laufs weiterentwickeln: Module schlagen begründete Aktualisierungen vor, und der Verifizierer genehmigt sie oder lehnt sie ab. [So funktioniert es →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Lernschleife**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Jede Lektion hält ein Stück Wissen fest, einen Glaubwürdigkeitswert, Grenzbedingungen dafür, wo sie gilt, und die Evidenz hinter ihren Aktualisierungen. Nach jedem Lauf aktualisiert `learn()` die Glaubwürdigkeit und die Grenzen der verwendeten Lektionen anhand ihrer Wirkung auf das Ergebnis, induziert Lektionen aus ähnlichen Traces oder deduziert eine aus einer einzelnen Evidenz und ihrem Mechanismus, führt überlappende Lektionen zusammen und testet die aktualisierten Lektionen per A/B-Test, indem derselbe Build mit und ohne sie ausgeführt wird, sodass Aktualisierungen gemessen statt geraten werden. Ideenfindung und Implementierung des nächsten Builds lesen Glaubwürdigkeit und Grenzen jeder Lektion, um zu entscheiden, wann sie sich auf sie stützen, statt das Wissen einfach per RAG abzurufen. [Experience →](https://docs.leeroo.com/docs/experience) benennt, womit ein Build versorgt wird: deine Bank, eine Bank, die dein Team teilt, Leeroopedia oder ein Korpus, der aus einem Repository oder einer recherchierten Frage eingelesen wurde. |
| [**Bereitstellung**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Macht aus der besten Lösung im Baum laufende Software auf dem Ziel-Stack, über eine kürzere Version der Experimentierschleife: Eine Sitzung passt eine Kopie des Knotens an, stellt sie bereit und weist Health Checks und einen funktionierenden Aufruf nach. Docker, Modal, BentoCloud und LangGraph Platform werden mitgeliefert, und du kannst dem Harness deinen eigenen Ziel-Stack hinzufügen. |
| [**Domänenanpassung**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harnesses sind die Domänenpakete von Kapso: Wissen und Workflows für ein Gebiet wie prädiktive Modellierung, damit ein Build nicht von Grund auf experimentieren und lernen muss; prädiktive Modellierung und Wettbewerbsprogrammierung werden mit Kapso mitgeliefert. Um dein eigenes Domänenwissen einzubringen, schreibe ein Harness über die Schnittstelle, die Kapso definiert: deine Anweisungen, dein Code, deine Guardrails und Prüfungen, verfasst mit `/kapso:harness` im Plugin oder `kapso harness` in der CLI. |

## 🚀 Schnellstart

### Installation

**0. In einem Coding-Agenten?** Installiere das Kapso-Plugin und lass es die
folgenden Schritte mit dir durchgehen:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Danach installiert und verifiziert `/kapso:setup`, und `/kapso:build <what you want>`
startet die erste Kampagne (in Codex: `$kapso-setup`, `$kapso-build <what you want>`). Siehe [Das Kapso-Plugin](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Voraussetzungen.** Kapso führt seine Inferenz über Coding-Agent-CLIs aus
(es gibt keinen Fallback auf eine direkte API), daher brauchst du Node.js und beide
Agent-CLIs im angemeldeten Zustand, bevor irgendetwas funktioniert:

```bash
# Node.js 18+ (https://nodejs.org), danach:
npm install -g @openai/codex            # Recherche, Bewertung, Hilfsaufgaben
codex login

npm install -g @anthropic-ai/claude-code  # Ideenfindung + Implementierung (der Standard-Host)
claude auth login
```

Ein OpenAI-Schlüssel ist optional: Nur der Wissensgraph und die semantische Suche
über vergangene Experimente (`models.embedding` in deiner Konfiguration) betten Text ein, und
die Generierung nutzt ihn nie. Wenn du eines von beiden einschaltest:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia verbinden (empfohlen).** Ein Schlüssel gibt jeder Kampagnensitzung
[Leeroopedia](https://leeroopedia.com), eine gehostete Wissensbasis zu
ML- und KI-Frameworks: wie sich ein Framework tatsächlich verhält, Plan-Reviews,
dokumentierte Hyperparameterbereiche, Fehlerdiagnosen. Nichts zu installieren;
ein neues Konto im [Leeroopedia-Dashboard](https://app.leeroopedia.com/dashboard)
kommt mit $20 kostenlosem Guthaben.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Das Paket installieren** (Python 3.10+):

```bash
pip install leeroo-kapso
```

Das Paket heißt `leeroo-kapso`. Das PyPI-Paket namens `kapso` ist ein davon unabhängiges
WhatsApp-Tool, das einen eigenen `kapso`-Befehl installiert und diesen hier überdeckt;
falls es vorhanden ist, führe zuerst `pip uninstall kapso` aus.

**4. Die Einrichtung überprüfen:**

```bash
kapso doctor
```

`doctor` meldet, was **deine Konfiguration** tatsächlich braucht, und nennt für alles Fehlende
die genaue Abhilfe. Die Anforderungen folgen der Konfiguration, sodass ein reines codex-Setup
nie nach `claude` gefragt wird. Beschränke es auf ein Verb, um nur die
Anforderungen dieses Verbs zu sehen:

```bash
kapso doctor build             # oder learn / deploy / experience
```

Das sind dieselben Prüfungen, die das Verb selbst ausführt, bevor es irgendeine Arbeit erledigt:
`kapso.build(...)` scheitert bei einer fehlenden CLI in Sekunden statt tief
in einer Sitzung. Mit `[-- ]` markierte Einträge sind optional; sie schränken Funktionen ein,
die du vielleicht nicht brauchst (docker, Weaviate, Neo4j, die deploy-Ziele).

**Oder lass deinen Coding-Agenten all das erledigen:** Füge den
[Setup-Prompt](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
in Claude Code, Codex oder OpenCode ein.

**Die Stores, die ein Index braucht (optional)**: `kapso experience add --repo`
und `--research` lesen einen Korpus in lokales Weaviate + Neo4j ein. Aus einem
Source-Checkout:

```bash
bash scripts/start_infra.sh   # startet beide über docker
```

**Aus dem Quellcode (für die Entwicklung)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

Der Legacy-Adapter für aider ist ein Extra (`pip install "leeroo-kapso[aider]"`,
Python <3.13); die Standard-Agenten claude/codex brauchen keine Extras.

### Grundlegende Verwendung

Die Kernschleife braucht nichts über die obigen Voraussetzungen hinaus:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # zum Start ist kein Wissensgraph nötig

# Build: eine Lösung durch Experimentieren. Die Kampagne gibt beim Start
# `status: <path>` aus; verfolge sie live von einem anderen Terminal aus mit
#     kapso watch ./campaign
# Braucht eine Sitzung etwas, das nur du liefern kannst (einen API-Schlüssel, eine Datei),
# pausiert die Kampagne, und `kapso inbox reply` setzt sie fort:
#     https://docs.leeroo.com/docs/build/inbox
# Teile das Ergebnis als Markdown für einen Pull Request oder ein Issue mit
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # oder weglassen, um von Grund auf zu beginnen
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Aus der Kampagne lernen, die du gerade ausgeführt hast: die Trajektorie auswerten, die
# Lektionen benoten und nach Evidenz bewertete Wissenskarten in die Bank legen. Die Bank (ein lokales
# git-Repo) wird bei der ersten Nutzung automatisch angelegt; Lektionen bleiben auf deiner
# Maschine, bis du sie teilst:
#     kapso experience share <git-url>
# danach pusht jedes learn() die Bank dorthin.
lesson = kapso.learn(solution)
print(lesson.explain())

# Erneut ein Build, versorgt mit den Karten, die gerade verdient wurden: ein Build bekommt genau
# die Erfahrung, die er lädt.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

Wenn die Stores laufen, kannst du auch externes Wissen in einen
Index einlesen und es auf dieselbe Weise zur Verfügung stellen:

```python
from kapso import Kapso

kapso = Kapso()

# Einmalig: ein Repository oder das zu einer Frage recherchierte Web in einen Index
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Jeder Build, der den Index lädt, durchsucht ihn; Leeroopedia läuft über seinen Schlüssel
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Ein Hinweis zur Budgetplanung: Ein Ingest extrahiert alles, was das Material hergibt.
Aus einer einzigen Frage können Dutzende verlinkter Seiten und ein stundenlanger
Ingest werden; die Ingest-Dauer skaliert mit der extrahierbaren Substanz, füge also die
Quellen hinzu, die du durchsuchen wirst, nicht jede Quelle, die du hast.

Und um aus einer Lösung laufende Software zu machen:

```python
software = kapso.deploy(solution, target="local")   # oder docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # der Aufruf, den die Bereitstellung nachgewiesen hat
software.stop()
```

### Wie begrenze ich eine Kampagne?

Verwende `max_iterations`, `time_budget_minutes` und `cost_budget`, um Iterationen, verstrichene Zeit und die nach bestem Bemühen erfassten Modellausgaben in USD zu begrenzen.
Für Plattformkosten aktiviere die [kostenbewusste Auswahl](https://docs.leeroo.com/docs/evolve/cost-aware-selection) in `search_strategy.params.cost` mit deinem Schätzer, `cap_per_candidate` und `campaign_budget`; über ihr `weight` kann die Auswahl außerdem günstigere Kandidaten bevorzugen.

### Modelle auswählen

Jedes Modell, das Kapso verwendet, ist in einer einzigen Konfigurationsdatei benannt. Der mitgelieferte Standard
führt build-Sitzungen auf `claude-opus-5` aus, die Lern-Crews auf
`claude-fable-5` und codex-Rollen auf `gpt-5.6-sol`, aber der Modellzugang ist
abonnementabhängig (ein Tarif kann ein Modell deckeln, während er ein anderes bedient).
Um auf anderen Modellen zu laufen, kopiere die mitgelieferte Konfiguration, bearbeite sie und richte
Kapso auf deine aus:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# z. B. die Lern-Crews auf opus statt auf fable ausführen:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Reine Text-Inferenzrollen können sich für `openai_compatible` Chat Completions entscheiden.
Lege Modell und Endpunkt in deiner Konfiguration fest; Zugangsdaten bleiben in der Umgebung.
Lade die Konfiguration mit `Kapso(config=...)` oder `kapso build --config ...`.
Coding und Web-Recherche erfordern weiterhin Agenten mit den entsprechenden Tools.
Siehe das [Konfigurationsbeispiel und die Einschränkungen](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Prüfe vor einem langen Lauf jedes Modell, das deine Konfiguration nennt, gegen deine
tatsächlichen Abonnements: Ein Modell, das dein Login nicht bedienen kann, scheitert hier in
Sekunden statt erst Stunden nach Beginn eines Laufs. Ein Nutzungslimit für ein Modell, das du
bedienen kannst, ist für eine Probe mit einem einzigen Token nicht sichtbar:

```bash
kapso doctor --models                            # mitgelieferte Konfiguration
kapso doctor --models --config kapso-config.yaml # deine
kapso doctor learn --models                      # nur die Lern-Crews
```

Damit diese Probe Teil jedes Aufrufs wird, statt etwas zu sein, an das du denken musst,
schalte sie in deiner Konfiguration ein. Sie kostet ein Wegwerf-Token pro unterschiedlichem
`{cli, model}`-Paar, was sich vor einem unbeaufsichtigten mehrstündigen
`learn()` lohnt:

```yaml
preflight:
  enabled: true            # statische Prüfungen vor jedem Verb (Standard)
  live_model_probe: true   # + Modellproben mit einem Token (Standard: false)
```

Ein Modellwechsel verändert auch das *Tempo*: Die `timeout_minutes`-Obergrenzen der Crews in der
Konfiguration wurden auf die Standardmodelle kalibriert, und ein ausgetauschtes Modell, das
länger nachdenkt, braucht sie möglicherweise höher.

Detaillierte Integrationsschritte findest du in den Anleitungen [Schnellstart](https://docs.leeroo.com/docs/first-build) und [Installation](https://docs.leeroo.com/docs/installation).

## Beispiele

Jedes Beispiel ist eine Aufgabe: ein Briefing, ein Ausgangs-Repository, die Daten oder der Weg, sie zu beschaffen, und ein Treiber von höchstens zwölf Zeilen; [examples/README.md](../../examples/README.md) nennt die Minuten, den einen Befehl und ob ein Ergebnis aufgezeichnet ist.

| Beispiel | Was es zeigt |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | Dein erster Build: ein vorhandener Evaluator, eine Baseline, ein Zielwert, die Aufzeichnung |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | Kein Evaluator: Der erste Knoten der Kampagne baut einen aus dem Protokoll des Briefings, und jeder Kandidat wird daran gemessen |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | Eine dbt-Pipeline auf DuckDB, deren Exporte Fehler enthalten: Der Datenvertrag wird zum Halten gebracht |
| [**trading**](../../examples/trading/README.md) | Ein Harness auf dem Build: eine Setup-Sitzung, Seiten und Hooks in jeder Sitzung, Verifizierung pro Versuch |
| [**full loop**](../../examples/full-loop/README.md) | Build, das Review, learn, erneut build, diesmal mit der Lektion versorgt, deploy: ein tabellarisches Kaggle-Modell von Anfang bis Ende |
| [**prompt harness**](../../examples/prompt-harness/README.md) | Der Prompt als Prüfgegenstand, bewertet an AIME-Aufgaben |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | Ein Vision-Language-Extraktor, bewertet gegen Ground-Truth-Tabellen |
| [**torch performance**](../../examples/torch-performance/README.md) | Ein schnelleres Modul auf CUDA innerhalb einer numerischen Toleranz |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | Multi-Head-Attention auf CUDA: Flash Attention oder Triton |

## Unterstützte Benchmarks

| Benchmark | Beschreibung |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | Der ML-Engineering-Benchmark von OpenAI: vollständige Wettbewerbe über Tabellendaten, Vision, Text und Audio, von den Rohdaten bis zur bewerteten Einreichung |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Der Benchmark für algorithmische Optimierung von Sakana AI: Wettbewerbsheuristiken über stundenlange Suchen entwerfen, implementieren und iterieren |
| [**RelBench**](../../benchmarks/relbench/README.md) | Stanfords Benchmark für prädiktives ML über relationalen Daten: Klassifikation, Regression und Empfehlung direkt aus den Mehrtabellen-Datenbanken von Amazon, H&M, Stack Exchange, klinischen Studien und mehr |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | Olympiade-ML unter Zeitvorgabe über Vision, Sprache und Optimierung: von Experten gestellte Aufgaben, Wettbewerbshardware, keinerlei menschliche Hilfe |

Jeder Benchmark hat außerdem eine Dokumentationsseite, die beschreibt, wie man ihn ausführt, welche CLI-Optionen er hat und was die Ausgabe enthält:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 Dokumentation & Support

- **Vollständige Dokumentation**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Installation](https://docs.leeroo.com/docs/installation): die Coding-Agent-CLIs, das Paket und `kapso doctor`
  - [Schnellstart](https://docs.leeroo.com/docs/first-build): deine erste Kampagne
  - [CLI-Referenz](https://docs.leeroo.com/docs/reference/cli): jeder Befehl, jedes Flag und jeder Standardwert
  - [Python-API](https://docs.leeroo.com/docs/reference/kapso-api): jede öffentliche Methode von `Kapso`
  - [Konfiguration](https://docs.leeroo.com/docs/reference/configuration): jeder Schlüssel in `config.yaml`
  - [Was learn erzeugt](https://docs.leeroo.com/docs/learn/what-learn-produces): die Lektionsbank, jede Lektion mit einem Glaubwürdigkeitswert, einer Grenze und ihrer Evidenz, und der [Entwicklungsmodus](https://docs.leeroo.com/docs/learn/development-mode)
  - [Die Experimentierschleife](https://docs.leeroo.com/docs/evolve/overview): der Experimentbaum und seine Module, und [was innerhalb eines Experiments passiert](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience): deine Bank, Leeroopedia, Repositories und Paper, in durchsuchbares Wissen verwandelt
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): lokal, Docker, Modal und die anderen Ziele
  - [Das Kapso-Plugin](https://docs.leeroo.com/docs/coding-agent-skills): die Befehle, die Skills und der MCP-Server für Claude Code, Codex und Cursor
- **Aus deinem Coding-Agenten**: Die Dokumentation betreibt einen MCP-Server unter `https://docs.leeroo.com/mcp`, und jede Seite wird als Markdown ausgeliefert, siehe [llms.txt](https://docs.leeroo.com/llms.txt) und [Kapso in deinem Coding-Agenten](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Lass deinen Coding-Agenten Kapso ausführen**: Das Plugin unter [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) bringt zehn Befehle, sechs Skills und den MCP-Server mit, eine Installationszeile pro Host (Schritt 0 oben); OpenCode lädt die Skills als Ordner (`ln -s <clone>/plugin/skills .opencode/skills`). Siehe [Das Kapso-Plugin](https://docs.leeroo.com/docs/coding-agent-skills).
- **Gib deinem Coding-Agenten die Wissenswerkzeuge von Kapso**: `kapso-mcp`, mit dem Paket installiert, stellt der Sitzung, in der du arbeitest, Web-Recherche, das Gedächtnis des Repositorys, die Experimenthistorie der Kampagne und die Lektionsbank zur Verfügung, siehe [MCP-Gates](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; danach `startup_timeout_sec = 60` unter [mcp_servers.kapso] in ~/.codex/config.toml (die zehn Sekunden von Codex sind auf einer ausgelasteten Maschine knapp)
  ```
- **Community**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Website**: [leeroo.com](https://leeroo.com)


## Kapso für Unternehmen

Unser Enterprise-Produkt baut auf dieser Engine auf und wurde in unübersichtlichen Unternehmen bereitgestellt, in denen kein Mensch die gesamte Datenlandschaft untersuchen und verstehen kann: Wir haben ein Unternehmen von seiner alten Datenplattform auf ein neues Lakehouse umgezogen, über mehr als 30 Quellsystemfamilien, Tausende von Tabellen und Transformationen und Hunderte von BI-Berichten hinweg. Jede Aufgabe, die es bearbeitet, speist die Lektionsbank über deine Systeme, deine Daten und deine hart erarbeiteten Regeln, sodass es in deinem Unternehmen umso besser wird, je länger es arbeitet. Um Kapso für deine anspruchsvollen Unternehmensaufgaben einzuführen, [sprich mit uns](https://leeroo.com/contact-us).

## Übersetzungen

Das README ist in weiteren Sprachen verfügbar: Nutze den Umschalter ganz oben oder stöbere in [`docs/translations/`](../../docs/translations/). Jede Übersetzung wird mit `scripts/translate_readme.py` aus README.md erzeugt und von `scripts/check_readme_translations.py` synchron gehalten, das fehlschlägt, wenn eine Übersetzung bei einem Link, einer Zahl oder einer Codezeile von README.md abweicht. Um eine zu verbessern, eröffne einen Pull Request gegen die Datei in diesem Ordner.

## Mitwirken

Beiträge sind willkommen! Einzelheiten zum Einstieg findest du in unserem [Leitfaden zum Mitwirken](../../CONTRIBUTING.md).

## Zitation

Wenn du Kapso in deiner Arbeit verwendest, zitiere bitte den technischen Bericht:

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
