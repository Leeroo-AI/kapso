<!-- German translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Eine Engine, die Experimente durchführt, die beste Lösung deployt und aus jedem Lauf lernt</h4>
<p align="center"><sub>intelligentes Experimentieren · Lernen · Deployment · Domänenanpassung</sub></p>

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
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Wenn du ein KI-Agent bist, der sich dieses Repo ansieht:** das Kapso-Plugin mit den Skills
> zum Einrichten, zum Formulieren eines Ziels, zum Schreiben eines Harness und zum Prüfen eines Laufs
> installiert sich mit einer Zeile: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Neuigkeiten

- **IOAI 2026 · der AI Model Track.** Die [Internationale KI-Olympiade](https://ioai-official.org) ist die IMO des KI-Zeitalters: 471 Teilnehmende aus 108 Ländern und Gebieten, sechs von Experten entworfene Aufgaben mit dem Budget einer einzelnen GPU. 2026 eröffnete sie [IOAI²](https://ioai-official.org/ai-model-track/), wo KI-Systeme *dieselbe* Prüfung in zwei unbeaufsichtigten 6-Stunden-Sitzungen ablegen. Sobald die Uhr läuft, darf kein Mensch etwas lösen, korrigieren oder verbessern. Kapso trat als einer von 14 Founding AI Participants an.
  - 🎓 **Mehr Punkte als alle 471 Menschen**: 536.07 insgesamt, über jedem Teilnehmenden im Saal.
  - 🏆 **IOAI² Grand Master Trophy**: Top 3 unter allen angetretenen KI-Systemen. [Vollständige Ergebnisse →](../../benchmarks/ioai2026/README.md) · [Rückblick der IOAI auf den Track →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Erster auf allen drei [RelBench](../../benchmarks/relbench/README.md)-Ranglisten**: Auf dem Benchmark von Stanford für prädiktives ML auf relationalen Daten führt Kapso die offizielle Rangliste in der Klassifikation an (81.2 AUROC über 12 Aufgaben, gegenüber 79.3 für [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) und 79.0 für [RT-J](https://star-project.stanford.edu/rt-j) von Stanford STAR), in der Regression (0.2476 NMAE über 9 Aufgaben, gegenüber 0.2711 für RT-J und 0.2847 für KumoRFM-v2, 8.7% weniger Fehler) und bei Empfehlungen (18.4 MAP über 10 Aufgaben, gegenüber 9.3 für ID-GNN, den besten anderen Eintrag). Jede Zahl ist ein Eintrag auf der [offiziellen RelBench-Rangliste](https://star-project.stanford.edu/relbench/leaderboard/). Lernen, das im Harness gehalten wird, schlug Lernen, das in Modellgewichte eingebacken ist, und eine Lektionenbank zu aktualisieren kostet weit weniger, als ein tabellarisches Foundation Model neu zu trainieren.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)**: ein Wiki unserer Lektionen in den Domänen Daten und ML, rund 100k Seiten, extrahiert durch Experimentieren an Papers und Repositories, das etwa 6M Anfragen pro Monat bedient, rund 28% davon von KI-Agenten. Kapso verbindet sich darüber per MCP: Seine Sitzungen nutzen es während Ideenfindung und Implementierung, um Wissen zu durchsuchen, Pläne zu erstellen und Fehler zu diagnostizieren.
- **#1 auf [MLE-Bench](../../benchmarks/mle/README.md)**: KAPSO erreichte den Spitzenplatz unter den Open-Source-Systemen bei Kaggle-ML-Wettbewerben (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **#1 auf [ALE-Bench](../../benchmarks/ale/README.md)**: KAPSO erreichte den Spitzenplatz bei Problemen algorithmischer Entdeckung mit langem Horizont (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## Was ist Kapso?

Ein KI- oder Datenergebnis in einem Unternehmen zu liefern heißt, zwei Dinge durch Experimente zu lernen. Kein Mensch und kein KI-Agent weiß, wie eine Datenlandschaft verbunden ist und betrieben wird; der einzige Weg, es zu lernen, besteht darin, die Artefakte des Unternehmens zu untersuchen (Repos, Pipelines, Dokumentation, Datenquellen), Hypothesen aufzustellen, Experimente durchzuführen und aus ihnen zu lernen, bis das Verhalten jedes Systems und seine Bedingungen verstanden sind. Und selbst wenn dieses Verständnis vorhanden ist, lässt sich die beste Lösung nicht im Voraus bestimmen, ob es nun darum geht, ein Warehouse zu optimieren oder Abwanderung vorherzusagen; also experimentiert und lernt man, bis eine zur Umgebung passt. Das Know-how, das zur Umgebung passt, sammelt sich an, ähnlich wie beim RL. RAG macht das Gegenteil: Es füttert den Agenten mit allgemeinem Wissen, das in dieser Umgebung nie getestet wurde und durchweg dasselbe Gewicht trägt, und dieses Wissen hat wenig Wert, bevor es dort ausprobiert wurde.

Kapso ist die Engine, die wir dafür gebaut haben. Sie startet mit einem Domänen-Harness und einer leeren Lektionenbank; mit einem Ziel und Erfolgskriterien führt sie einen Experimentbaum aus, stellt Hypothesen auf, implementiert und verifiziert dann jede davon und deployt die beste Lösung auf dem Ziel-Stack. Anschließend lernt sie aus dem Lauf, indem sie jeder Hypothese und Implementierung ihren Anteil am Ergebnis zuschreibt und die Bank aktualisiert, in der jede Lektion einen Glaubwürdigkeitswert und eine Geltungsgrenze trägt, die angibt, wo sie anwendbar ist. Mit der Zeit trägt das Domänen-Harness eine Lektionenbank, in der sich experimentell erprobte Best Practices für diese Umgebung angesammelt haben. Das System ist modellagnostisch und unabhängig vom Coding-Agent: Open Source, von [Leeroo](https://leeroo.com), veröffentlicht auf PyPI als [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### Die vier Teile

| Teil | Verb | Was er tut |
|------|------|--------------|
| [**Intelligentes Experimentieren**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Spezialisierte Module teilen sich eine Datenstruktur, einen Experimentbaum: Jeder Knoten hält die Hypothese, die Implementierung (Git-Branch und technische Schwierigkeiten), das verbrauchte Budget und die Distanz zum Ziel fest. Für jedes Experiment schlägt ein Ideenfindungsmodul Ideen aus früheren Experimenten und der Lektionenbank vor, ordnet sie nach erwartetem Ertrag gemessen am Budget, das jede verbrauchen würde, neu und schickt die bestplatzierten an den Implementierer, der auf einem vom Elternknoten abgezweigten Branch aufbaut. Weil die Bank fortlaufend lernt, was in genau dieser Umgebung funktioniert, werden die Ideen spezifisch dafür, statt der generischen, die ein bloßes LLM oder RAG liefern würde. Ergebnisse werden von einem mit Fingerabdruck versehenen Evaluator verifiziert, auf den kein Experiment zugreifen kann, und auch die Evaluation entwickelt sich weiter: Die Experimentiermodule können Aktualisierungen der Metriken und Kriterien zur Genehmigung durch den Verifizierer vorschlagen. [So funktioniert es →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Lernen**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Alles in der Bank stammt aus Experimenten: Jede Lektion wird aus den Traces von Experimentierläufen induziert oder deduziert, trägt einen Glaubwürdigkeitswert, Geltungsgrenzen, die angeben, wo sie anwendbar ist, und die Evidenz dahinter, und nach jedem Lauf aktualisiert `learn()` diese anhand der Wirkung der Lektion auf das Endergebnis. Änderungen an der Bank sind ebenfalls Experimente: Jede aktualisierte Lektion wird per A/B-Test geprüft, indem derselbe Build mit und ohne sie ausgeführt wird, sodass eine Verbesserung gemessen und nicht geraten wird. Die Bank baut auf Git auf, du kannst deine Erfahrung also mit anderen teilen (`kapso experience share`), die dir verfügbaren Banken auflisten (`kapso experience list`) und sie aktualisieren (`kapso experience update`). [Experience →](https://docs.leeroo.com/docs/experience) benennt, was einem Build zur Verfügung gestellt wird: deine Bank, eine Bank, die dein Team teilt, Leeroopedia oder ein Korpus, der aus einem Repository oder einer recherchierten Frage ingestiert wurde. |
| [**Deployment**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Führt ein kürzeres Experimentieren derselben Art durch, um die beste Lösung an den Ziel-Stack anzupassen: Eine Sitzung passt eine Kopie des Knotens an, deployt sie und demonstriert Health-Checks und einen funktionierenden Aufruf. Docker, Modal, BentoCloud und LangGraph Platform werden mitgeliefert, und du kannst dem Harness deinen eigenen Ziel-Stack hinzufügen. |
| [**Domänenanpassung**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harnesses sind die Domänenpakete von Kapso: Wissen und Workflows für ein Gebiet wie prädiktive Modellierung, damit ein Build nicht von Grund auf experimentieren und lernen muss; prädiktive Modellierung und Wettbewerbsprogrammierung werden mit Kapso ausgeliefert. Ein Lauf startet mit einem Domänen-Harness und einer leeren Lektionenbank, und mit der Zeit trägt das Harness die Bank, in der sich angesammelt hat, was in dieser Umgebung funktioniert hat. Schreibe dein initiales Domänen-Harness, deine Anweisungen, deinen Code, deine Leitplanken und Prüfungen mit `/kapso:harness` im Plugin oder `kapso harness` in der CLI. |

## 🚀 Schnellstart

### Installation

**0. In einem Coding-Agent?** Installiere das Kapso-Plugin und lass es die
Schritte unten mit dir durchgehen:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Dann installiert und verifiziert `/kapso:setup`, und `/kapso:build <what you want>`
startet die erste Kampagne (auf Codex: `$kapso-setup`, `$kapso-build <what you want>`). Siehe [Das Kapso-Plugin](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Voraussetzungen.** Kapso führt seine Inferenz über Coding-Agent-CLIs aus
(es gibt keinen Fallback auf eine direkte API), daher brauchst du Node.js und beide
Agent-CLIs eingeloggt, bevor irgendetwas funktioniert:

```bash
# Node.js 18+ (https://nodejs.org), dann:
npm install -g @openai/codex            # Recherche, Beurteilung, Hilfsprogramme
codex login

npm install -g @anthropic-ai/claude-code  # Ideenfindung + Implementierung (der Standard-Host)
claude auth login
```

Ein OpenAI-Schlüssel ist optional: Nur der Wissensgraph und die semantische Suche
über vergangene Experimente (`models.embedding` in deiner Konfiguration) betten Text ein,
und die Generierung nutzt ihn nie. Wenn du eines von beiden einschaltest:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia verbinden (empfohlen).** Ein Schlüssel gibt jeder Kampagnensitzung
[Leeroopedia](https://leeroopedia.com), eine gehostete Wissensbasis über
ML- und KI-Frameworks: wie sich ein Framework tatsächlich verhält, Plan-Reviews,
dokumentierte Hyperparameter-Bereiche, Fehlerdiagnosen. Nichts zu installieren;
ein neues Konto im [Leeroopedia-Dashboard](https://app.leeroopedia.com/dashboard)
kommt mit $20 Gratisguthaben.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Paket installieren** (Python 3.10+):

```bash
pip install leeroo-kapso
```

Das Paket heißt `leeroo-kapso`. Das PyPI-Paket namens `kapso` ist ein nicht verwandtes
WhatsApp-Tool, das einen eigenen `kapso`-Befehl installiert und diesen hier überdeckt;
falls es vorhanden ist, zuerst `pip uninstall kapso`.

**4. Einrichtung verifizieren:**

```bash
kapso doctor
```

`doctor` meldet, was **deine Konfiguration** tatsächlich braucht, und nennt die genaue
Lösung für alles, was fehlt. Die Anforderungen folgen der Konfiguration, ein reines
Codex-Setup wird also nie nach `claude` gefragt. Grenze es auf ein Verb ein, um nur die
Anforderungen dieses Verbs zu sehen:

```bash
kapso doctor build             # oder learn / deploy / experience
```

Das sind dieselben Prüfungen, die das Verb selbst ausführt, bevor es irgendeine Arbeit tut:
`kapso.build(...)` schlägt bei einer fehlenden CLI in Sekunden fehl statt tief
in einer Sitzung. Mit `[-- ]` markierte Einträge sind optional; sie begrenzen Funktionen,
die du vielleicht nicht brauchst (docker, Weaviate, Neo4j, die Deployment-Ziele).

**Oder lass deinen Coding-Agent all das erledigen:** füge den
[Setup-Prompt](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
in Claude Code, Codex oder OpenCode ein.

**Die Stores, die ein Index braucht (optional)**: `kapso experience add --repo`
und `--research` ingestieren ein Korpus in lokales Weaviate + Neo4j. Aus einem
Quell-Checkout:

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

Der ältere aider-Adapter ist ein Extra (`pip install "leeroo-kapso[aider]"`,
Python <3.13); die Standard-Agenten claude/codex brauchen keine Extras.

### Grundlegende Verwendung

Die Kernverben brauchen nichts über die obigen Voraussetzungen hinaus:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # kein Wissensgraph nötig, um zu starten

# Build: eine Lösung durch Experimentieren. Die Kampagne gibt beim Start
# `status: <path>` aus; verfolge sie live aus einem anderen Terminal mit
#     kapso watch ./campaign
# Wenn eine Sitzung etwas braucht, das nur du liefern kannst (einen API-Schlüssel, eine Datei),
# pausiert die Kampagne, und `kapso inbox reply` setzt sie fort:
#     https://docs.leeroo.com/docs/build/inbox
# Teile das Ergebnis als Markdown für einen Pull Request oder ein Issue mit
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # oder weglassen, um bei null zu starten
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Lerne aus der Kampagne, die du gerade ausgeführt hast: die Trajektorie durchforsten, die
# Lektionen benoten und nach Evidenz bepreiste Wissenskarten einlagern. Die Bank (ein lokales
# Git-Repo) wird bei der ersten Nutzung automatisch angelegt; Lektionen bleiben auf deinem
# Rechner, bis du sie teilst:
#     kapso experience share <git-url>
# danach pusht jedes learn() die Bank dorthin.
lesson = kapso.learn(solution)
print(lesson.explain())

# Erneut bauen, versorgt mit den Karten, die es gerade verdient hat: Ein Build erhält genau
# die Experience, die er lädt.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

Mit laufenden Stores kannst du auch externes Wissen in einen
Index ingestieren und es auf dieselbe Weise bereitstellen:

```python
from kapso import Kapso

kapso = Kapso()

# Einmalig: ein Repository oder das zu einer Frage recherchierte Web in einen Index
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Jeder Build, der den Index lädt, durchsucht ihn; Leeroopedia läuft über seinen Schlüssel mit
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Eine Anmerkung zum Budget: Ein Ingest extrahiert alles, was das Material hergibt.
Eine einzige Frage kann zu Dutzenden verknüpfter Seiten und einem stundenlangen
Ingest werden; die Ingest-Zeit skaliert mit der extrahierbaren Substanz, füge also die
Quellen hinzu, die du durchsuchen wirst, nicht jede Quelle, die du hast.

Und um aus einer Lösung laufende Software zu machen:

```python
software = kapso.deploy(solution, target="local")   # oder docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # der Aufruf, den das Deployment nachgewiesen hat
software.stop()
```

### Wie begrenze ich eine Kampagne?

Nutze `max_iterations`, `time_budget_minutes` und `cost_budget`, um Iterationen, verstrichene Zeit und die bestmöglich geschätzten Modellausgaben in USD zu begrenzen.
Für Plattformkosten aktiviere die [kostenbewusste Auswahl](https://docs.leeroo.com/docs/evolve/cost-aware-selection) in `search_strategy.params.cost` mit deinem Schätzer, `cap_per_candidate` und `campaign_budget`; über ihr `weight` kann die Auswahl auch günstigere Kandidaten bevorzugen.

### Modelle auswählen

Jedes Modell, das Kapso verwendet, ist in einer Konfigurationsdatei benannt. Die mitgelieferte
Standardkonfiguration lässt Build-Sitzungen auf `claude-opus-5` laufen, die Lern-Crews auf
`claude-fable-5` und Codex-Rollen auf `gpt-5.6-sol`; der Modellzugang hängt aber
vom Abonnement ab (ein Plan kann ein Modell deckeln, während er ein anderes bedient).
Um andere Modelle zu verwenden, kopiere die mitgelieferte Konfiguration, bearbeite sie und
zeige Kapso auf deine:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# z. B. die Lern-Crews auf opus statt fable laufen lassen:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Reine Text-Inferenzrollen können optional `openai_compatible` Chat Completions nutzen.
Setze Modell und Endpunkt in deiner Konfiguration; Zugangsdaten bleiben in der Umgebung.
Lade die Konfiguration mit `Kapso(config=...)` oder `kapso build --config ...`.
Coding und Web-Recherche erfordern weiterhin Agenten mit den entsprechenden Tools.
Siehe das [Konfigurationsbeispiel und die Grenzen](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Prüfe vor einem langen Lauf jedes Modell, das deine Konfiguration benennt, gegen deine
tatsächlichen Abonnements: Ein Modell, das dein Login nicht bedienen kann, scheitert hier in
Sekunden statt Stunden nach Beginn eines Laufs. Eine Nutzungsgrenze auf einem Modell, das du
bedienen kannst, ist für eine Ein-Token-Probe nicht sichtbar:

```bash
kapso doctor --models                            # mitgelieferte Konfiguration
kapso doctor --models --config kapso-config.yaml # deine
kapso doctor learn --models                      # nur die Lern-Crews
```

Um diese Probe zu einem Teil jedes Aufrufs zu machen statt zu etwas, an das du denken
musst, schalte sie in deiner Konfiguration ein. Sie kostet ein Wegwerf-Token pro unterschiedlichem
`{cli, model}`-Paar, was sich vor einem unbeaufsichtigten, mehrstündigen
`learn()` lohnt:

```yaml
preflight:
  enabled: true            # statische Prüfungen vor jedem Verb (Standard)
  live_model_probe: true   # + Ein-Token-Modellproben (Standard: false)
```

Modellwechsel ändern auch das *Tempo*: Die `timeout_minutes`-Obergrenzen der Crews in der
Konfiguration wurden auf den Standardmodellen kalibriert, und ein getauschtes Modell, das
länger nachdenkt, braucht sie möglicherweise höher.

Ausführliche Integrationsschritte findest du in den Anleitungen [Schnellstart](https://docs.leeroo.com/docs/first-build) und [Installation](https://docs.leeroo.com/docs/installation).

## Unterstützte Benchmarks

| Benchmark | Beschreibung |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | Der ML-Engineering-Benchmark von OpenAI: vollständige Wettbewerbe über tabellarische Daten, Vision, Text und Audio, von Rohdaten bis zur bewerteten Einreichung |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Der Benchmark für algorithmische Optimierung von Sakana AI: Wettbewerbsheuristiken über stundenlange Suchen entwerfen, implementieren und iterieren |
| [**RelBench**](../../benchmarks/relbench/README.md) | Der Benchmark von Stanford für prädiktives ML auf relationalen Daten: Klassifikation, Regression und Empfehlung direkt aus den Multi-Tabellen-Datenbanken von Amazon, H&M, Stack Exchange, klinischen Studien und mehr |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | Olympiade-ML auf Zeit über Vision, Sprache und Optimierung: von Experten gestellte Aufgaben, Wettbewerbs-Hardware, null menschliche Hilfe |

Jeder Benchmark hat außerdem eine Dokumentationsseite, die beschreibt, wie man ihn ausführt, welche CLI-Optionen er hat und was die Ausgabe enthält:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## Beispiele

Jedes Beispiel ist eine Aufgabe: ein Briefing, ein Ausgangs-Repository, die Daten oder wie man sie bekommt, und ein Treiber von höchstens zwölf Zeilen; [examples/README.md](../../examples/README.md) nennt die Minuten, den einen Befehl und ob ein Ergebnis festgehalten ist.

| Beispiel | Was es zeigt |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | Dein erster Build: ein vorhandener Evaluator, eine Baseline, ein Zielwert, die Aufzeichnung |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | Kein Evaluator: Der erste Knoten der Kampagne baut einen aus dem Protokoll des Briefings, und jeder Kandidat wird daran gemessen |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | Eine dbt-Pipeline auf DuckDB, deren Exporte Defekte aufweisen: der Datenvertrag, der wieder hält |
| [**trading**](../../examples/trading/README.md) | Ein Harness auf dem Build: eine Setup-Sitzung, Seiten und Hooks in jeder Sitzung, Verifikation pro Versuch |
| [**end to end**](../../examples/end-to-end/README.md) | Build, das Review, learn, erneut build mit der Lektion, deploy: ein tabellarisches Kaggle-Modell von Anfang bis Ende |
| [**prompt harness**](../../examples/prompt-harness/README.md) | Der Prompt als Prüfgegenstand, bewertet an AIME-Aufgaben |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | Ein Vision-Language-Extraktor, bewertet gegen Ground-Truth-Tabellen |
| [**torch performance**](../../examples/torch-performance/README.md) | Ein schnelleres Modul auf CUDA unter einer numerischen Toleranz |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | Multi-Head-Attention auf CUDA: Flash Attention oder Triton |

## 📚 Dokumentation & Support

- **Vollständige Dokumentation**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Installation](https://docs.leeroo.com/docs/installation): die Coding-Agent-CLIs, das Paket und `kapso doctor`
  - [Schnellstart](https://docs.leeroo.com/docs/first-build): deine erste Kampagne
  - [CLI-Referenz](https://docs.leeroo.com/docs/reference/cli): jeder Befehl, jedes Flag und jeder Standardwert
  - [Python-API](https://docs.leeroo.com/docs/reference/kapso-api): jede öffentliche Methode von `Kapso`
  - [Konfiguration](https://docs.leeroo.com/docs/reference/configuration): jeder Schlüssel in `config.yaml`
  - [Was learn erzeugt](https://docs.leeroo.com/docs/learn/what-learn-produces): die Lektionenbank, jede Lektion mit einem Glaubwürdigkeitswert, einer Geltungsgrenze und ihrer Evidenz, und der [Entwicklungsmodus](https://docs.leeroo.com/docs/learn/development-mode)
  - [Intelligentes Experimentieren](https://docs.leeroo.com/docs/evolve/overview): der Experimentbaum und seine Module, und [was innerhalb eines Experiments passiert](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience): deine Bank, Leeroopedia, Repositories und Papers, zu durchsuchbarem Wissen gemacht
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal und die anderen Ziele
  - [Das Kapso-Plugin](https://docs.leeroo.com/docs/coding-agent-skills): die Befehle, die Skills und der MCP-Server für Claude Code, Codex und Cursor
- **Aus deinem Coding-Agent heraus**: Die Docs betreiben einen MCP-Server unter `https://docs.leeroo.com/mcp`, und jede Seite wird als Markdown ausgeliefert; siehe [llms.txt](https://docs.leeroo.com/llms.txt) und [Kapso in deinem Coding-Agent](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Lass deinen Coding-Agent Kapso ausführen**: Das Plugin unter [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) enthält zehn Befehle, sechs Skills und den MCP-Server, eine Installationszeile pro Host (Schritt 0 oben); OpenCode lädt die Skills als Ordner (`ln -s <clone>/plugin/skills .opencode/skills`). Siehe [Das Kapso-Plugin](https://docs.leeroo.com/docs/coding-agent-skills).
- **Gib deinem Coding-Agent die Wissenswerkzeuge von Kapso**: `kapso-mcp`, mit dem Paket installiert, stellt der Sitzung, in der du arbeitest, Web-Recherche, das Gedächtnis des Repositorys, die Experimenthistorie der Kampagne und die Lektionenbank bereit; siehe [MCP-Gates](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; dann `startup_timeout_sec = 60` unter [mcp_servers.kapso] in ~/.codex/config.toml (die zehn Sekunden von Codex sind auf einem ausgelasteten Rechner knapp)
  ```
- **Community**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Website**: [leeroo.com](https://leeroo.com)


## Kapso für Unternehmen

Im Unternehmen werden Kapso-Agenten mit initialen Harnesses für Daten- und KI-Engineering und einer Lektionenbank ausgeliefert und verwandeln Geschäftsziele eigenständig in funktionierende KI- und Datensysteme, indem sie die Artefakte des Unternehmens untersuchen, Repos, Pipelines, Datenquellen und mehr, und so nach und nach die Datenlandschaft des Unternehmens aufbauen. Um Kapso für deine Unternehmensaufgaben einzuführen, [sprich mit uns](https://leeroo.com/contact-us).

## Mitwirken

Wir freuen uns über Beiträge! Details zum Einstieg findest du in unserem [Leitfaden für Beiträge](../../CONTRIBUTING.md).

## Mitwirkende

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

Erstellt mit [contrib.rocks](https://contrib.rocks).
