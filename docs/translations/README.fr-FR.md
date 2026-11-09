<!-- French translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Un moteur qui mène des expériences, déploie la meilleure solution et apprend de chaque exécution</h4>
<p align="center"><sub>expérimentation intelligente · apprentissage · déploiement · adaptation au domaine</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <b>Français</b> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">En savoir plus</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Rejoindre le Discord</a> ·
  <a href="https://leeroo.com">Site web</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  Si ce projet vous plaît, soutenez-nous en lui attribuant une étoile ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Si vous êtes un agent IA en train de consulter ce dépôt :** le plugin Kapso, avec les compétences pour
> l'installation, la formulation d'un objectif, la rédaction d'un harness et la revue d'une exécution,
> s'installe en une ligne : `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex : `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Actualités

- **IOAI 2026 · le volet AI Model Track.** L'[Olympiade internationale d'IA](https://ioai-official.org) est l'IMO de l'ère de l'IA : 471 concurrents issus de 108 pays et territoires, six tâches conçues par des experts sous un budget d'un seul GPU. En 2026, elle a ouvert [IOAI²](https://ioai-official.org/ai-model-track/), où des systèmes d'IA passent le *même* examen en deux sessions de 6 heures sans supervision. Une fois le chronomètre lancé, aucun humain ne peut résoudre, corriger ni améliorer quoi que ce soit. Kapso y a participé comme l'un des 14 Participants IA fondateurs.
  - 🎓 **A devancé les 471 humains** : 536.07 au total, au-dessus de chaque concurrent présent dans la salle.
  - 🏆 **Trophée IOAI² Grand Master** : top 3 parmi tous les systèmes d'IA engagés. [Résultats complets →](../../benchmarks/ioai2026/README.md) · [Le bilan du volet par l'IOAI →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Premier sur les trois classements [RelBench](../../benchmarks/relbench/README.md)** : sur le benchmark de Stanford pour le ML prédictif sur données relationnelles, Kapso mène le classement officiel en classification (81.2 d'AUROC sur 12 tâches, contre 79.3 pour [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) et 79.0 pour [RT-J](https://star-project.stanford.edu/rt-j) de Stanford STAR), en régression (0.2476 de NMAE sur 9 tâches, contre 0.2711 pour RT-J et 0.2847 pour KumoRFM-v2, soit 8.7% d'erreur en moins) et en recommandation (18.4 de MAP sur 10 tâches, contre 9.3 pour ID-GNN, la meilleure autre entrée). Chaque chiffre est une entrée du [classement officiel RelBench](https://star-project.stanford.edu/relbench/leaderboard/). L'apprentissage conservé dans le harness a battu l'apprentissage figé dans les poids du modèle, et mettre à jour une banque de leçons coûte bien moins cher que réentraîner un modèle de fondation tabulaire.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)** : un wiki de nos leçons dans les domaines des données et du ML, environ 100k pages extraites en expérimentant sur des articles et des dépôts, servant environ 6M de requêtes par mois, dont environ 28% proviennent d'agents IA. Kapso s'y connecte via MCP : ses sessions l'utilisent pendant l'idéation et l'implémentation pour rechercher des connaissances, bâtir des plans et diagnostiquer des échecs.
- **N° 1 sur [MLE-Bench](../../benchmarks/mle/README.md)** : KAPSO a obtenu le meilleur classement parmi les systèmes open source sur les compétitions de ML Kaggle (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **N° 1 sur [ALE-Bench](../../benchmarks/ale/README.md)** : KAPSO a obtenu le meilleur classement sur des problèmes de découverte algorithmique à long horizon (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## Qu'est-ce que Kapso ?

Livrer un résultat d'IA ou de données au sein d'une entreprise suppose d'apprendre deux choses par l'expérimentation. Aucun individu ni agent IA ne sait comment un patrimoine de données est connecté et exploité ; la seule façon de l'apprendre est d'examiner les artefacts de l'entreprise (dépôts, pipelines, documentation, sources de données), de proposer des hypothèses, de mener des expériences et d'en tirer des leçons, jusqu'à ce que le comportement de chaque système et ses conditions soient compris. Et une fois cette compréhension acquise, la meilleure solution ne peut toujours pas être choisie à l'avance, qu'il s'agisse d'optimiser un entrepôt ou de prédire l'attrition, alors on expérimente et on apprend jusqu'à ce qu'une solution convienne à l'environnement. Le savoir-faire qui convient à l'environnement s'accumule, un peu comme en RL. Le RAG fait l'inverse : il fournit à l'agent des connaissances générales qui n'ont jamais été testées dans cet environnement, toutes avec le même poids, et ces connaissances ont peu de valeur tant qu'elles n'y ont pas été éprouvées.

Kapso est le moteur que nous avons construit pour cela. Il part d'un harness de domaine et d'une banque de leçons vide ; étant donné un objectif et des critères de réussite, il exécute un arbre d'expériences, en proposant des hypothèses puis en implémentant et en vérifiant chacune d'elles, et déploie la meilleure solution sur la pile cible. Il apprend ensuite de l'exécution, en attribuant à chaque hypothèse et implémentation un crédit selon son impact sur le résultat et en mettant à jour la banque, où chaque leçon porte un score de crédence et une frontière délimitant où elle s'applique. Avec le temps, le harness de domaine porte une banque de leçons qui a accumulé les bonnes pratiques éprouvées par l'expérimentation dans cet environnement. Le système est indépendant du modèle et de l'agent de codage : open source, par [Leeroo](https://leeroo.com), publié sur PyPI sous le nom [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### Les quatre composants

| Composant | Verbe | Ce qu'il fait |
|------|------|--------------|
| [**Expérimentation intelligente**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Des modules spécialisés partagent une même structure de données, un arbre d'expériences : chaque nœud enregistre l'hypothèse, l'implémentation (branche git et difficultés techniques), le budget consommé et la distance à l'objectif. Pour chaque expérience, un module d'idéation propose des idées à partir des expériences antérieures et de la banque de leçons, les reclasse selon le rendement attendu rapporté au budget que chacune consommerait, et envoie les mieux classées à l'implémenteur, qui construit sur une branche dérivée du nœud parent. Parce que la banque continue d'apprendre ce qui fonctionne dans cet environnement précis, les idées lui deviennent spécifiques, plutôt que les idées génériques qu'un LLM nu ou un RAG proposerait. Les résultats sont vérifiés par un évaluateur à empreinte numérique auquel aucune expérience ne peut accéder, et l'évaluation évolue elle aussi : les modules d'expérimentation peuvent proposer des mises à jour des métriques et des critères, que le vérificateur approuve. [Comment ça marche →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Apprentissage**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Tout ce que contient la banque provient d'expériences : chaque leçon est induite ou déduite des traces des exécutions d'expérimentation, porte un score de crédence, des conditions de frontière sur les cas où elle s'applique et les preuves qui la fondent, et après chaque exécution `learn()` les met à jour d'après l'effet de la leçon sur le résultat final. Les modifications de la banque sont elles aussi des expériences : chaque leçon mise à jour est testée en A/B en exécutant le même build avec et sans elle, de sorte qu'une amélioration est mesurée, et non supposée. La banque repose sur git, vous pouvez donc partager votre expérience avec d'autres (`kapso experience share`), lister les banques qui vous sont accessibles (`kapso experience list`) et les mettre à jour (`kapso experience update`). [Expérience →](https://docs.leeroo.com/docs/experience) nomme ce qui est servi à un build : votre banque, une banque partagée par votre équipe, Leeroopedia, ou un corpus ingéré depuis un dépôt ou issu d'une recherche sur une question. |
| [**Déploiement**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Lance une expérimentation plus courte du même type pour adapter la meilleure solution à la pile cible : une session adapte une copie du nœud, la déploie et démontre des contrôles de santé et un appel fonctionnel. Docker, Modal, BentoCloud et LangGraph Platform sont fournis, et vous pouvez ajouter votre propre pile cible au harness. |
| [**Adaptation au domaine**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Les harness sont les paquets de domaine de Kapso : des connaissances et des flux de travail pour un domaine tel que la modélisation prédictive, afin qu'un build n'expérimente pas et n'apprenne pas à partir de zéro ; la modélisation prédictive et la programmation compétitive sont fournies avec Kapso. Une exécution part d'un harness de domaine et d'une banque de leçons vide, et avec le temps le harness porte la banque qui a accumulé ce qui a fonctionné dans cet environnement. Rédigez votre harness de domaine initial, vos instructions, votre code, vos garde-fous et vos vérifications, avec `/kapso:harness` dans le plugin ou `kapso harness` dans la CLI. |

## 🚀 Démarrage rapide

### Installation

**0. Dans un agent de codage ?** Installez le plugin Kapso et laissez-le parcourir
avec vous les étapes ci-dessous :

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Ensuite `/kapso:setup` installe et vérifie, et `/kapso:build <what you want>`
lance la première campagne (sur Codex : `$kapso-setup`, `$kapso-build <what you want>`). Voir [Le plugin Kapso](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Prérequis.** Kapso exécute son inférence via les CLI d'agents de codage
(il n'y a pas de repli vers une API directe), vous avez donc besoin de Node.js et des deux
CLI d'agents connectées avant que quoi que ce soit ne fonctionne :

```bash
# Node.js 18+ (https://nodejs.org), puis :
npm install -g @openai/codex            # recherche, jugement, utilitaires
codex login

npm install -g @anthropic-ai/claude-code  # idéation + implémentation (l'hôte par défaut)
claude auth login
```

Une clé OpenAI est facultative : seuls le graphe de connaissances et la recherche sémantique
sur les expériences passées (`models.embedding` dans votre config) calculent des embeddings de texte, et
la génération ne l'utilise jamais. Quand vous activez l'un ou l'autre :

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Connectez Leeroopedia (recommandé).** Une seule clé donne à chaque session de
campagne [Leeroopedia](https://leeroopedia.com), une base de connaissances hébergée sur les
frameworks de ML et d'IA : comment un framework se comporte réellement, des revues de plan,
des plages d'hyperparamètres documentées, des diagnostics de pannes. Rien à installer ;
un nouveau compte sur le [tableau de bord Leeroopedia](https://app.leeroopedia.com/dashboard)
est livré avec 20 $ de crédit gratuit.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Installez le paquet** (Python 3.10+) :

```bash
pip install leeroo-kapso
```

Le paquet s'appelle `leeroo-kapso`. Le paquet PyPI nommé `kapso` est un outil WhatsApp
sans rapport qui installe sa propre commande `kapso` et masque celle-ci ;
s'il est présent, exécutez d'abord `pip uninstall kapso`.

**4. Vérifiez l'installation :**

```bash
kapso doctor
```

`doctor` indique ce dont **votre config** a réellement besoin, et nomme le correctif
exact pour tout ce qui manque. Les prérequis suivent la config, de sorte qu'une installation
tout-codex ne se voit jamais demander `claude`. Restreignez-le à un seul verbe pour ne voir
que les prérequis de ce verbe :

```bash
kapso doctor build             # ou learn / deploy / experience
```

Ce sont les mêmes vérifications que le verbe lui-même exécute avant tout travail :
`kapso.build(...)` échoue en quelques secondes sur une CLI manquante plutôt qu'au fin fond
d'une session. Les éléments marqués `[-- ]` sont facultatifs ; ils limitent des fonctionnalités
dont vous n'aurez peut-être pas besoin (docker, Weaviate, Neo4j, les cibles de déploiement).

**Ou laissez votre agent de codage faire tout cela :** collez le
[prompt d'installation](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
dans Claude Code, Codex ou OpenCode.

**Les stockages dont un index a besoin (facultatif)** : `kapso experience add --repo`
et `--research` ingèrent un corpus dans des instances locales de Weaviate + Neo4j. Depuis une
copie des sources :

```bash
bash scripts/start_infra.sh   # démarre les deux via docker
```

**Depuis les sources (pour le développement)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

L'adaptateur aider hérité est un extra (`pip install "leeroo-kapso[aider]"`,
Python <3.13) ; les agents claude/codex par défaut n'ont besoin d'aucun extra.

### Utilisation de base

Les verbes de base n'ont besoin de rien au-delà des prérequis ci-dessus :

```python
from kapso import Kapso, Budget

kapso = Kapso()   # aucun graphe de connaissances nécessaire pour démarrer

# Build : une solution par l'expérimentation. La campagne affiche
# `status: <path>` au lancement ; suivez-la en direct depuis un autre terminal avec
#     kapso watch ./campaign
# Si une session a besoin de quelque chose que vous seul pouvez fournir (une clé d'API, un fichier),
# la campagne se met en pause et `kapso inbox reply` la reprend :
#     https://docs.leeroo.com/docs/build/inbox
# Partagez le résultat en Markdown pour une pull request ou une issue avec
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # ou omettez pour partir de zéro
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Apprenez de la campagne que vous venez d'exécuter : extrayez la trajectoire, notez les
# leçons et versez à la banque des cartes de connaissance valorisées selon leurs preuves. La banque (un dépôt
# git local) est créée automatiquement à la première utilisation ; les leçons restent sur votre
# machine jusqu'à ce que vous les partagiez :
#     kapso experience share <git-url>
# après quoi chaque learn() y pousse la banque.
lesson = kapso.learn(solution)
print(lesson.explain())

# Lancez un nouveau build, servi des cartes qu'il vient de gagner : un build reçoit exactement
# l'expérience qu'il charge.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

Avec les stockages en marche, vous pouvez aussi ingérer des connaissances extérieures dans un
index et les servir de la même façon :

```python
from kapso import Kapso

kapso = Kapso()

# Une seule fois : un dépôt, ou le web exploré pour une question, dans un index
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Chaque build qui charge l'index y effectue des recherches ; Leeroopedia s'appuie sur sa clé
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Une remarque sur le budget : une ingestion extrait tout ce que le matériau permet d'extraire.
Une seule question peut devenir des dizaines de pages liées et une ingestion de plusieurs
heures ; le temps d'ingestion croît avec la substance extractible, ajoutez donc les
sources que vous allez interroger, pas toutes celles que vous possédez.

Et pour transformer une solution en logiciel en fonctionnement :

```python
software = kapso.deploy(solution, target="local")   # ou docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # l'appel que le déploiement a prouvé
software.stop()
```

### Comment borner une campagne ?

Utilisez `max_iterations`, `time_budget_minutes` et `cost_budget` pour borner les itérations, le temps écoulé et la dépense de modèle estimée au mieux, en USD.
Pour les coûts de plateforme, activez la [sélection sensible aux coûts](https://docs.leeroo.com/docs/evolve/cost-aware-selection) dans `search_strategy.params.cost` avec votre estimateur, `cap_per_candidate` et `campaign_budget` ; son `weight` permet aussi à la sélection de préférer les candidats les moins chers.

### Choisir les modèles

Chaque modèle utilisé par Kapso est nommé dans un seul fichier de config. La configuration par défaut
fournie exécute les sessions de build sur `claude-opus-5`, les équipes d'apprentissage sur
`claude-fable-5` et les rôles codex sur `gpt-5.6-sol`, mais l'accès aux modèles dépend de
l'abonnement (un forfait peut plafonner un modèle tout en en servant un autre).
Pour utiliser d'autres modèles, copiez la config fournie, modifiez-la et pointez
Kapso vers la vôtre :

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# p. ex. exécuter les équipes d'apprentissage sur opus plutôt que sur fable :
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Les rôles d'inférence purement textuels peuvent opter pour les Chat Completions `openai_compatible`.
Définissez le modèle et le point de terminaison dans votre config ; les identifiants restent dans l'environnement.
Chargez la config avec `Kapso(config=...)` ou `kapso build --config ...`.
Le codage et la recherche web nécessitent toujours des agents dotés des outils correspondants.
Voir l'[exemple de configuration et les limites](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Avant une longue exécution, sondez chaque modèle nommé dans votre config contre vos
abonnements réels : un modèle que votre compte ne peut pas servir échoue ici en quelques
secondes au lieu d'échouer des heures après le début d'une exécution. Un plafond d'utilisation
sur un modèle que vous pouvez servir n'est pas visible pour une sonde d'un seul token :

```bash
kapso doctor --models                            # config fournie
kapso doctor --models --config kapso-config.yaml # la vôtre
kapso doctor learn --models                      # seulement les équipes d'apprentissage
```

Pour faire de cette sonde une partie de chaque appel plutôt qu'une chose dont il faut se
souvenir, activez-la dans votre config. Elle coûte un token jetable par paire
`{cli, model}` distincte, ce qui en vaut la peine avant un `learn()` de plusieurs heures
sans surveillance :

```yaml
preflight:
  enabled: true            # vérifications statiques avant chaque verbe (par défaut)
  live_model_probe: true   # + sondes de modèle d'un token (par défaut : false)
```

Changer de modèle modifie aussi le *rythme* : les plafonds `timeout_minutes` des équipes dans la
config ont été calibrés sur les modèles par défaut, et un modèle de remplacement qui
raisonne plus longtemps peut nécessiter de les relever.

Pour les étapes d'intégration détaillées, voir les guides [Démarrage rapide](https://docs.leeroo.com/docs/first-build) et [Installation](https://docs.leeroo.com/docs/installation).

## Benchmarks pris en charge

| Benchmark | Description |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | Le benchmark d'ingénierie ML d'OpenAI : des compétitions complètes en tabulaire, vision, texte et audio, des données brutes à la soumission notée |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Le benchmark d'optimisation algorithmique de Sakana AI : concevoir, implémenter et itérer des heuristiques de concours au fil de recherches de plusieurs heures |
| [**RelBench**](../../benchmarks/relbench/README.md) | Le benchmark de Stanford pour le ML prédictif sur données relationnelles : classification, régression et recommandation directement depuis les bases de données multi-tables d'Amazon, H&M, Stack Exchange, d'essais cliniques et plus encore |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | Du ML d'olympiade chronométré en vision, langage et optimisation : des tâches conçues par des experts, le matériel du concours, zéro aide humaine |

Chaque benchmark dispose aussi d'une page de documentation décrivant comment l'exécuter, ses options de CLI et ce que contient la sortie :
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## Exemples

Chaque exemple est une tâche : un brief, un dépôt de départ, les données ou la façon de les obtenir, et un script pilote d'au plus douze lignes ; [examples/README.md](../../examples/README.md) donne la durée en minutes, la commande unique et indique si un résultat est enregistré.

| Exemple | Ce qu'il montre |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | Votre premier build : un évaluateur présent, une base de référence, une cible, le compte rendu |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | Pas d'évaluateur : le premier nœud de la campagne en construit un à partir du protocole du brief et chaque candidat est mesuré par rapport à lui |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | Un pipeline dbt sur DuckDB dont les exports comportent des défauts : le contrat de données rendu contraignant |
| [**trading**](../../examples/trading/README.md) | Un harness sur le build : une session d'installation, des pages et des hooks dans chaque session, une vérification par tentative |
| [**end to end**](../../examples/end-to-end/README.md) | Build, la revue, learn, build à nouveau servi de la leçon, deploy : un modèle tabulaire Kaggle de bout en bout |
| [**prompt harness**](../../examples/prompt-harness/README.md) | Le prompt comme objet du test, noté sur des problèmes AIME |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | Un extracteur vision-langage, noté par rapport à des tables de vérité terrain |
| [**torch performance**](../../examples/torch-performance/README.md) | Un module plus rapide sur CUDA sous une tolérance numérique |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | Attention multi-têtes sur CUDA : flash attention ou Triton |

## 📚 Documentation & support

- **Documentation complète** : [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Installation](https://docs.leeroo.com/docs/installation) : les CLI d'agents de codage, le paquet et `kapso doctor`
  - [Démarrage rapide](https://docs.leeroo.com/docs/first-build) : votre première campagne
  - [Référence CLI](https://docs.leeroo.com/docs/reference/cli) : chaque commande, option et valeur par défaut
  - [API Python](https://docs.leeroo.com/docs/reference/kapso-api) : chaque méthode publique de `Kapso`
  - [Configuration](https://docs.leeroo.com/docs/reference/configuration) : chaque clé de `config.yaml`
  - [Ce que produit learn](https://docs.leeroo.com/docs/learn/what-learn-produces) : la banque de leçons, chacune avec un score de crédence, une frontière et ses preuves, et le [mode développement](https://docs.leeroo.com/docs/learn/development-mode)
  - [Expérimentation intelligente](https://docs.leeroo.com/docs/evolve/overview) : l'arbre d'expériences et ses modules, et [ce qui se passe au sein d'une expérience](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Expérience](https://docs.leeroo.com/docs/experience) : votre banque, Leeroopedia, des dépôts et des articles transformés en connaissances interrogeables
  - [Déploiement](https://docs.leeroo.com/docs/deploy/targets) : local, Docker, Modal et les autres cibles
  - [Le plugin Kapso](https://docs.leeroo.com/docs/coding-agent-skills) : les commandes, les compétences et le serveur MCP pour Claude Code, Codex et Cursor
- **Depuis votre agent de codage** : la documentation expose un serveur MCP à `https://docs.leeroo.com/mcp`, et chaque page est servie en Markdown ; voir [llms.txt](https://docs.leeroo.com/llms.txt) et [Kapso dans votre agent de codage](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Laissez votre agent de codage piloter Kapso** : le plugin sous [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) contient dix commandes, six compétences et le serveur MCP, une ligne d'installation par hôte (étape 0 ci-dessus) ; OpenCode charge les compétences sous forme de dossier (`ln -s <clone>/plugin/skills .opencode/skills`). Voir [Le plugin Kapso](https://docs.leeroo.com/docs/coding-agent-skills).
- **Donnez à votre agent de codage les outils de connaissance de Kapso** : `kapso-mcp`, installé avec le paquet, sert la recherche web, la mémoire du dépôt, l'historique des expériences de la campagne et la banque de leçons à la session dans laquelle vous travaillez ; voir [Portes MCP](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI ; puis `startup_timeout_sec = 60` sous [mcp_servers.kapso] dans ~/.codex/config.toml (les dix secondes de Codex sont courtes sur une machine chargée)
  ```
- **Communauté** : [Discord](https://discord.gg/hqVbPNNEZM)
- **Site web** : [leeroo.com](https://leeroo.com)


## Kapso pour l'entreprise

En entreprise, les agents Kapso sont livrés avec des harness initiaux d'ingénierie des données et de l'IA et une banque de leçons, et transforment par eux-mêmes des objectifs métier en systèmes d'IA et de données opérationnels, en explorant les artefacts de l'entreprise, dépôts, pipelines, sources de données et plus encore, pour bâtir au fil de l'eau le patrimoine de données de l'entreprise. Pour adopter Kapso pour vos tâches d'entreprise, [contactez-nous](https://leeroo.com/contact-us).

## Contribuer

Les contributions sont les bienvenues ! Consultez notre [guide de contribution](../../CONTRIBUTING.md) pour savoir comment démarrer.

## Contributeurs

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

Réalisé avec [contrib.rocks](https://contrib.rocks).
