<!-- French translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Un moteur qui mène des expériences, déploie la meilleure solution et apprend de chaque exécution</h4>
<p align="center"><sub>expérimentation intelligente · boucle d'apprentissage · déploiement · adaptation au domaine</sub></p>

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
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Si vous êtes un agent d'IA qui consulte ce dépôt :** le plugin Kapso, avec les skills pour
> l'installation, la formulation d'un objectif, la rédaction d'un harness et la revue d'une exécution, s'installe
> en une ligne : `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex : `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Actualités

- **IOAI 2026 · le volet AI Model Track.** L'[Olympiade internationale d'IA](https://ioai-official.org) est l'IMO de l'ère de l'IA : 471 concurrents venus de 108 pays et territoires, six tâches conçues par des experts avec un budget d'un seul GPU. En 2026, elle a ouvert [IOAI²](https://ioai-official.org/ai-model-track/), où des systèmes d'IA passent le *même* examen en deux sessions de 6 heures sans intervention humaine. Une fois le chronomètre lancé, aucun humain ne peut résoudre, corriger ni améliorer quoi que ce soit. Kapso y a pris part en tant que l'un des 14 participants IA fondateurs.
  - 🎓 **A devancé les 471 humains** : 536.07 au total, au-dessus de tous les concurrents présents dans la salle.
  - 🏆 **Trophée IOAI² Grand Master** : dans le top 3 de tous les systèmes d'IA engagés. [Résultats complets →](../../benchmarks/ioai2026/README.md) · [Le bilan de l'épreuve par l'IOAI →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Premier sur les trois classements [RelBench](../../benchmarks/relbench/README.md)** : sur le benchmark de Stanford pour le ML prédictif sur données relationnelles, Kapso est en tête du classement officiel en classification (81.2 AUROC sur 12 tâches, contre 79.3 pour [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) et 79.0 pour [RT-J](https://star-project.stanford.edu/rt-j) de Stanford STAR), en régression (0.2476 NMAE sur 9 tâches, contre 0.2711 pour RT-J et 0.2847 pour KumoRFM-v2, soit 8.7% d'erreur en moins) et en recommandation (18.4 MAP sur 10 tâches, contre 9.3 pour ID-GNN, la meilleure des autres entrées). Chaque chiffre correspond à une entrée du [classement officiel RelBench](https://star-project.stanford.edu/relbench/leaderboard/).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)** : un wiki de nos leçons dans les domaines des données et du ML, environ 100k pages extraites en expérimentant sur des articles et des dépôts, qui sert environ 6M de requêtes par mois, dont environ 28% proviennent d'agents d'IA. Kapso s'y connecte via MCP : ses sessions l'utilisent pendant l'idéation et l'implémentation pour rechercher des connaissances, élaborer des plans et diagnostiquer des échecs.
- **[Agents Moltbook 🦞](https://www.moltbook.com/)** : créez des agents d'IA qui optimisent d'autres agents et débattent sur Moltbook ! [Commencer →](../../moltbook_bot/README.md)
- **Rapport technique** : notre rapport technique est désormais disponible ! [Lire l'article](https://arxiv.org/abs/2601.21526)
- **#1 sur [MLE-Bench](../../benchmarks/mle/README.md)** : KAPSO a obtenu la première place parmi les systèmes open source sur les compétitions de ML de Kaggle (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **#1 sur [ALE-Bench](../../benchmarks/ale/README.md)** : KAPSO a obtenu la première place sur des problèmes de découverte algorithmique à long horizon (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## Qu'est-ce que Kapso ?

Kapso est un moteur qui mène des expériences, déploie la meilleure solution et apprend de chaque exécution. À partir d'un objectif, de critères de réussite initiaux et de métriques d'évaluation, il enquête, mène des expériences en choisissant les meilleures idées, fournit parfois des retours pour améliorer les métriques d'évaluation et les critères de réussite, déploie la meilleure solution sur la pile souhaitée et apprend de chaque exécution. Il tient une banque de leçons, chacune dotée d'un score de crédibilité et d'une limite qui précise où elle s'applique, de sorte que la connaissance fructifie au fil du temps au lieu de seulement s'accumuler.

Nous avons construit des systèmes d'IA au sein de grandes entreprises, dont LinkedIn et Meta AI, et les avons déployés pour des produits comptant des millions d'utilisateurs. Kapso, c'est ce flux de travail sous forme de moteur : open source, par [Leeroo](https://leeroo.com), publié sur PyPI sous le nom [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### Les quatre parties

| Partie | Verbe | Ce qu'elle fait |
|------|------|--------------|
| [**Expérimentation intelligente**](https://docs.leeroo.com/docs/build/brief) | `.build()` | La boucle d'expérimentation est un ensemble de modules spécialisés qui partagent une même structure de données, un arbre d'expériences : chaque nœud consigne l'hypothèse, l'implémentation (branche git et difficultés techniques), le budget consommé et la distance à l'objectif. L'idéation s'appuie sur les expériences précédentes et sur la banque de leçons pour proposer des idées, chacune indiquant en quoi elle diffère d'expériences antérieures précises et quelles leçons elle adopte ; un module de reclassement les ordonne selon le rendement attendu par rapport à la cible, mis en balance avec le temps et le budget. L'implémenteur construit sur une branche dérivée du nœud parent, et un vérificateur contrôle le nœud au regard des règles de sécurité et des critères de réussite, décide si le score est valide et si l'objectif est atteint. Les critères de réussite et les tests peuvent évoluer au cours d'une exécution : les modules proposent des mises à jour justifiées, et le vérificateur les approuve ou les rejette. [Comment ça marche →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Boucle d'apprentissage**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Chaque leçon consigne une connaissance, un score de crédibilité, des conditions aux limites qui précisent où elle s'applique et les preuves à l'origine de ses mises à jour. Après chaque exécution, `learn()` met à jour la crédibilité et les limites des leçons utilisées d'après leur effet sur le résultat, induit des leçons à partir de traces similaires ou en déduit une à partir d'une seule preuve et de son mécanisme, fusionne les leçons qui se recoupent, et soumet les leçons mises à jour à un test A/B en exécutant le même build avec et sans elles, de sorte que les mises à jour sont mesurées plutôt que devinées. L'idéation et l'implémentation du build suivant lisent la crédibilité et les limites de chaque leçon pour décider quand s'y fier, au lieu de se contenter d'un RAG sur la connaissance. [Experience →](https://docs.leeroo.com/docs/experience) désigne ce qui est servi à un build : votre banque, une banque partagée par votre équipe, Leeroopedia, ou un corpus ingéré à partir d'un dépôt ou d'une question ayant fait l'objet d'une recherche. |
| [**Déploiement**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Transforme la meilleure solution de l'arbre en logiciel en fonctionnement sur la pile cible, au moyen d'une version raccourcie de la boucle d'expérimentation : une session adapte une copie du nœud, la déploie, puis fait la démonstration des contrôles de santé et d'un appel fonctionnel. Docker, Modal, BentoCloud et LangGraph Platform sont fournis d'origine, et vous pouvez ajouter votre propre pile cible au harness. |
| [**Adaptation au domaine**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Les harness sont les paquets de domaine de Kapso : des connaissances et des flux de travail pour un domaine tel que la modélisation prédictive, afin qu'un build n'ait pas à expérimenter et à apprendre en partant de zéro ; la modélisation prédictive et la programmation compétitive sont fournies avec Kapso. Pour apporter votre propre connaissance du domaine, écrivez un harness via l'interface que définit Kapso : vos instructions, votre code, vos garde-fous et vos contrôles, rédigés avec `/kapso:harness` dans le plugin ou `kapso harness` dans la CLI. |

## 🚀 Démarrage rapide

### Installation

**0. Dans un agent de programmation ?** Installez le plugin Kapso et laissez-le parcourir
les étapes ci-dessous avec vous :

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Ensuite, `/kapso:setup` installe et vérifie, et `/kapso:build <what you want>`
lance la première campagne (sur Codex : `$kapso-setup`, `$kapso-build <what you want>`). Voir [Le plugin Kapso](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Prérequis.** Kapso effectue son inférence par l'intermédiaire des CLI d'agents de programmation
(il n'existe aucune solution de repli par API directe), il vous faut donc Node.js et les deux CLI
d'agents connectées à leur compte avant que quoi que ce soit ne fonctionne :

```bash
# Node.js 18+ (https://nodejs.org), puis :
npm install -g @openai/codex            # recherche, jugement, utilitaires
codex login

npm install -g @anthropic-ai/claude-code  # idéation + implémentation (l'hôte par défaut)
claude auth login
```

Une clé OpenAI est facultative : seuls le graphe de connaissances et la recherche sémantique
sur les expériences passées (`models.embedding` dans votre configuration) calculent des embeddings de texte, et
la génération ne l'utilise jamais. Lorsque vous activez l'un ou l'autre :

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Connecter Leeroopedia (recommandé).** Une seule clé donne à chaque session de
campagne l'accès à [Leeroopedia](https://leeroopedia.com), une base de connaissances hébergée sur les
frameworks de ML et d'IA : le comportement réel d'un framework, des revues de plans,
des plages d'hyperparamètres documentées, des diagnostics d'échec. Rien à installer ;
un nouveau compte sur le [tableau de bord Leeroopedia](https://app.leeroopedia.com/dashboard)
est fourni avec $20 de crédit gratuit.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Installer le paquet** (Python 3.10+) :

```bash
pip install leeroo-kapso
```

Le paquet s'appelle `leeroo-kapso`. Le paquet PyPI nommé `kapso` est un outil
WhatsApp sans rapport, qui installe sa propre commande `kapso` et masque celle-ci ;
s'il est présent, lancez d'abord `pip uninstall kapso`.

**4. Vérifier l'installation :**

```bash
kapso doctor
```

`doctor` indique ce dont **votre configuration** a réellement besoin et nomme le correctif
exact pour tout ce qui manque. Les exigences suivent la configuration : une installation
entièrement codex ne se voit donc jamais réclamer `claude`. Restreignez-le à un seul verbe pour ne voir que
les exigences de ce verbe :

```bash
kapso doctor build             # ou learn / deploy / experience
```

Ce sont les mêmes contrôles que le verbe exécute lui-même avant d'entreprendre le moindre travail :
`kapso.build(...)` échoue en quelques secondes sur une CLI manquante plutôt qu'au beau milieu
d'une session. Les éléments marqués `[-- ]` sont facultatifs ; ils restreignent des fonctionnalités
dont vous n'avez peut-être pas besoin (docker, Weaviate, Neo4j, les cibles de déploiement).

**Ou laissez votre agent de programmation faire tout cela :** collez le
[prompt d'installation](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
dans Claude Code, Codex ou OpenCode.

**Les bases de données dont un index a besoin (facultatif)** : `kapso experience add --repo`
et `--research` ingèrent un corpus dans des instances locales de Weaviate + Neo4j. Depuis une
copie de travail des sources :

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

L'ancien adaptateur aider est un extra (`pip install "leeroo-kapso[aider]"`,
Python <3.13) ; les agents claude/codex par défaut ne nécessitent aucun extra.

### Utilisation de base

La boucle principale ne nécessite rien d'autre que les prérequis ci-dessus :

```python
from kapso import Kapso, Budget

kapso = Kapso()   # aucun graphe de connaissances n'est nécessaire pour commencer

# Build : une solution obtenue par l'expérimentation. La campagne affiche
# `status: <path>` au lancement ; suivez-la en direct depuis un autre terminal avec
#     kapso watch ./campaign
# Si une session a besoin de quelque chose que vous seul pouvez fournir (une clé d'API, un fichier),
# la campagne se met en pause et `kapso inbox reply` la relance :
#     https://docs.leeroo.com/docs/build/inbox
# Partagez le résultat en Markdown pour une pull request ou une issue avec
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # ou omettez-le pour partir de zéro
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Learn : tirer les leçons de la campagne que vous venez d'exécuter, en explorant la trajectoire, en notant les
# leçons et en versant à la banque des fiches de connaissance valorisées selon leurs preuves. La banque (un dépôt
# git local) est créée automatiquement à la première utilisation ; les leçons restent sur votre
# machine jusqu'à ce que vous les partagiez :
#     kapso experience share <git-url>
# après quoi chaque learn() pousse la banque à cet endroit.
lesson = kapso.learn(solution)
print(lesson.explain())

# Build à nouveau, cette fois servi par les fiches qu'il vient de gagner : un build se voit servir exactement
# l'expérience acquise qu'il charge.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

Une fois les bases de données démarrées, vous pouvez aussi ingérer des connaissances externes dans un
index et les servir de la même manière :

```python
from kapso import Kapso

kapso = Kapso()

# Une seule fois : un dépôt, ou le web exploré pour une question, versé dans un index
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Chaque build qui charge l'index y effectue ses recherches ; Leeroopedia passe par sa propre clé
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Une remarque sur le budget : une ingestion extrait tout ce que le matériau permet d'extraire.
Une seule question peut donner des dizaines de pages liées et une ingestion
de plusieurs heures ; la durée d'ingestion croît avec la substance extractible, ajoutez donc les
sources dans lesquelles vous chercherez, et non toutes les sources dont vous disposez.

Et pour transformer une solution en logiciel en fonctionnement :

```python
software = kapso.deploy(solution, target="local")   # ou docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # l'appel que le déploiement a prouvé
software.stop()
```

### Comment borner une campagne ?

Utilisez `max_iterations`, `time_budget_minutes` et `cost_budget` pour borner les itérations, le temps écoulé et la dépense de modèles en USD, estimée au mieux.
Pour les coûts de plateforme, activez la [sélection tenant compte des coûts](https://docs.leeroo.com/docs/evolve/cost-aware-selection) dans `search_strategy.params.cost` avec votre estimateur, `cap_per_candidate` et `campaign_budget` ; son `weight` permet aussi à la sélection de privilégier les candidats moins coûteux.

### Choisir les modèles

Chaque modèle utilisé par Kapso est nommé dans un seul fichier de configuration. La configuration par défaut fournie
exécute les sessions de build sur `claude-opus-5`, les équipes d'apprentissage sur
`claude-fable-5` et les rôles codex sur `gpt-5.6-sol`, mais l'accès aux modèles dépend
de l'abonnement (une formule peut plafonner un modèle tout en en servant un autre).
Pour utiliser d'autres modèles, copiez la configuration fournie, modifiez-la et indiquez
la vôtre à Kapso :

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# p. ex. exécuter les équipes d'apprentissage sur opus au lieu de fable :
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Les rôles d'inférence purement textuels peuvent opter pour les Chat Completions `openai_compatible`.
Définissez le modèle et le point de terminaison dans votre configuration ; les identifiants restent dans l'environnement.
Chargez la configuration avec `Kapso(config=...)` ou `kapso build --config ...`.
La programmation et la recherche web nécessitent toujours des agents dotés des outils correspondants.
Voir l'[exemple de configuration et ses limitations](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Avant une longue exécution, sondez chaque modèle nommé dans votre configuration au regard de vos
abonnements réels : un modèle que votre compte ne peut pas servir échoue ici en
quelques secondes plutôt qu'après des heures d'exécution. Un plafond d'utilisation sur un modèle que vous pouvez
servir n'est pas visible pour une sonde d'un seul token :

```bash
kapso doctor --models                            # configuration fournie
kapso doctor --models --config kapso-config.yaml # la vôtre
kapso doctor learn --models                      # uniquement les équipes d'apprentissage
```

Pour que cette sonde fasse partie de chaque appel au lieu d'être une chose qu'il faut penser à
lancer, activez-la dans votre configuration. Elle coûte un token jetable par paire
`{cli, model}` distincte, ce qui en vaut la peine avant un appel de plusieurs heures sans surveillance à
`learn()` :

```yaml
preflight:
  enabled: true            # contrôles statiques avant chaque verbe (par défaut)
  live_model_probe: true   # + sondes de modèle à un token (par défaut : false)
```

Changer de modèle modifie aussi le *rythme* : les plafonds `timeout_minutes` des équipes dans la
configuration ont été calibrés sur les modèles par défaut, et un modèle de remplacement qui
raisonne plus longtemps peut nécessiter de les relever.

Pour les étapes d'intégration détaillées, consultez les guides [Démarrage rapide](https://docs.leeroo.com/docs/first-build) et [Installation](https://docs.leeroo.com/docs/installation).

## Exemples

Chaque exemple est une tâche : un brief, un dépôt de départ, les données ou la manière de les obtenir, et un script pilote de douze lignes au maximum ; [examples/README.md](../../examples/README.md) indique la durée en minutes, la commande unique et si un résultat est enregistré.

| Exemple | Ce qu'il montre |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | Votre premier build : un évaluateur présent, une base de référence, une cible, le compte rendu |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | Aucun évaluateur : le premier nœud de la campagne en construit un à partir du protocole du brief et chaque candidat est mesuré par rapport à lui |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | Un pipeline dbt sur DuckDB dont les exports comportent des défauts : le contrat de données que l'on fait respecter |
| [**trading**](../../examples/trading/README.md) | Un harness sur le build : une session de mise en place, des pages et des hooks dans chaque session, une vérification à chaque tentative |
| [**full loop**](../../examples/full-loop/README.md) | Build, la revue, learn, build à nouveau avec la leçon servie, deploy : un modèle tabulaire Kaggle de bout en bout |
| [**prompt harness**](../../examples/prompt-harness/README.md) | Le prompt comme objet testé, noté sur des problèmes AIME |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | Un extracteur vision-langage, noté par rapport à des tables de vérité terrain |
| [**torch performance**](../../examples/torch-performance/README.md) | Un module plus rapide sur CUDA dans le respect d'une tolérance numérique |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | Attention multi-têtes sur CUDA : flash attention ou Triton |

## Benchmarks pris en charge

| Benchmark | Description |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | Le benchmark d'ingénierie ML d'OpenAI : des compétitions complètes couvrant le tabulaire, la vision, le texte et l'audio, des données brutes à la soumission notée |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Le benchmark d'optimisation algorithmique de Sakana AI : concevoir, implémenter et itérer des heuristiques de concours au fil de recherches de plusieurs heures |
| [**RelBench**](../../benchmarks/relbench/README.md) | Le benchmark de Stanford pour le ML prédictif sur données relationnelles : classification, régression et recommandation directement à partir des bases de données multi-tables d'Amazon, de H&M, de Stack Exchange, d'essais cliniques, et d'autres encore |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | Du ML d'olympiade en temps limité, couvrant la vision, le langage et l'optimisation : des tâches établies par des experts, le matériel du concours, zéro aide humaine |

Chaque benchmark dispose aussi d'une page de documentation expliquant comment l'exécuter, ses options de CLI et ce que contient la sortie :
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 Documentation et support

- **Documentation complète** : [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Installation](https://docs.leeroo.com/docs/installation) : les CLI des agents de programmation, le paquet et `kapso doctor`
  - [Démarrage rapide](https://docs.leeroo.com/docs/first-build) : votre première campagne
  - [Référence de la CLI](https://docs.leeroo.com/docs/reference/cli) : chaque commande, chaque option et chaque valeur par défaut
  - [API Python](https://docs.leeroo.com/docs/reference/kapso-api) : chaque méthode publique de `Kapso`
  - [Configuration](https://docs.leeroo.com/docs/reference/configuration) : chaque clé de `config.yaml`
  - [Ce que produit learn](https://docs.leeroo.com/docs/learn/what-learn-produces) : la banque de leçons, chacune avec un score de crédibilité, une limite et ses preuves, ainsi que le [mode développement](https://docs.leeroo.com/docs/learn/development-mode)
  - [La boucle d'expérimentation](https://docs.leeroo.com/docs/evolve/overview) : l'arbre d'expériences et ses modules, ainsi que [ce qui se passe à l'intérieur d'une expérience](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience) : votre banque, Leeroopedia, des dépôts et des articles transformés en connaissances interrogeables
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets) : local, Docker, Modal et les autres cibles
  - [Le plugin Kapso](https://docs.leeroo.com/docs/coding-agent-skills) : les commandes, les skills et le serveur MCP pour Claude Code, Codex et Cursor
- **Depuis votre agent de programmation** : la documentation fait tourner un serveur MCP à l'adresse `https://docs.leeroo.com/mcp`, et chaque page est servie en Markdown ; voir [llms.txt](https://docs.leeroo.com/llms.txt) et [Kapso dans votre agent de programmation](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Laissez votre agent de programmation piloter Kapso** : le plugin situé dans [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) contient dix commandes, six skills et le serveur MCP, avec une ligne d'installation par hôte (étape 0 ci-dessus) ; OpenCode charge les skills sous forme de dossier (`ln -s <clone>/plugin/skills .opencode/skills`). Voir [Le plugin Kapso](https://docs.leeroo.com/docs/coding-agent-skills).
- **Donnez à votre agent de programmation les outils de connaissance de Kapso** : `kapso-mcp`, installé avec le paquet, sert la recherche web, la mémoire du dépôt, l'historique des expériences de la campagne et la banque de leçons à la session dans laquelle vous travaillez ; voir [Portes MCP](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI ; puis `startup_timeout_sec = 60` sous [mcp_servers.kapso] dans ~/.codex/config.toml (les dix secondes de Codex sont courtes sur une machine chargée)
  ```
- **Communauté** : [Discord](https://discord.gg/hqVbPNNEZM)
- **Site web** : [leeroo.com](https://leeroo.com)


## Kapso pour les entreprises

Notre produit pour les entreprises est construit sur ce moteur et a été déployé au sein d'entreprises à l'environnement désordonné, où aucun humain ne peut examiner et comprendre l'ensemble du patrimoine de données : nous avons migré une entreprise de son ancienne plateforme de données vers un nouveau lakehouse, à travers plus de 30 familles de systèmes sources, des milliers de tables et de transformations, et des centaines de rapports BI. Chaque tâche qu'il traite alimente la banque de leçons sur vos systèmes, vos données et vos règles durement acquises, de sorte qu'il connaît de mieux en mieux votre entreprise à mesure qu'il y travaille. Pour adopter Kapso pour vos tâches d'entreprise les plus exigeantes, [contactez-nous](https://leeroo.com/contact-us).

## Traductions

Le README est disponible dans d'autres langues : utilisez le sélecteur en haut de la page, ou parcourez [`docs/translations/`](../../docs/translations/). Chaque traduction est générée à partir de README.md avec `scripts/translate_readme.py` et maintenue à jour par `scripts/check_readme_translations.py`, qui échoue lorsqu'une traduction diverge de README.md sur un lien, un nombre ou une ligne de code. Pour en améliorer une, ouvrez une pull request sur le fichier concerné dans ce dossier.

## Contribuer

Les contributions sont les bienvenues ! Consultez notre [guide de contribution](../../CONTRIBUTING.md) pour savoir comment démarrer.

## Citation

Si vous utilisez Kapso dans vos travaux, merci de citer le rapport technique :

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
