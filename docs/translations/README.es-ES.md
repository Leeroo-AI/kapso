<!-- Spanish translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Un motor que ejecuta experimentos, despliega la mejor solución y aprende de cada ejecución</h4>
<p align="center"><sub>experimentación inteligente · aprendizaje · despliegue · adaptación al dominio</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <b>Español</b> | <a href="README.hi-IN.md">हिन्दी</a> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">Más información</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Únete a Discord</a> ·
  <a href="https://leeroo.com">Sitio web</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  Si te gusta este proyecto, apóyanos dándole una estrella ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Si eres un agente de IA que está mirando este repositorio:** el plugin de Kapso, con las skills para
> configurar, enunciar un objetivo, escribir un harness y revisar una ejecución, se instala
> en una línea: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Noticias

- **IOAI 2026 · la pista AI Model Track.** La [Olimpiada Internacional de IA](https://ioai-official.org) es la IMO de la era de la IA: 471 concursantes, seis tareas diseñadas por expertos, un presupuesto de una sola GPU. Su pista [IOAI²](https://ioai-official.org/ai-model-track/) de 2026 puso a los sistemas de IA ante el mismo examen en una única sesión autónoma. Kapso, uno de los 14 participantes fundadores de IA, obtuvo 536.07 según la calificación de los organizadores, por encima de todos los concursantes humanos, y se llevó el Trofeo Grand Master de IOAI², top 3 entre los sistemas de IA inscritos. [Resultados completos →](../../benchmarks/ioai2026/README.md) · [La reseña de la IOAI sobre la pista →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Primero en los tres rankings de [RelBench](../../benchmarks/relbench/README.md).** En el benchmark de Stanford para ML predictivo sobre datos relacionales, Kapso lidera el [ranking oficial](https://star-project.stanford.edu/relbench/leaderboard/) en clasificación (81.2 de AUROC frente a 79.3 de [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) y 79.0 de [RT-J](https://star-project.stanford.edu/rt-j) de Stanford STAR), regresión (0.2476 de NMAE frente a 0.2711 y 0.2847) y recomendaciones (18.4 de MAP frente a 9.3 de ID-GNN, la mejor de las demás entradas).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0 in classification; 0.2476 NMAE against 0.2711 and 0.2847 in regression, lower being better; 18.4 MAP against ID-GNN's 9.3 in recommendations. The first two panels use a truncated axis, the third starts at zero" width="574"></p>

- **[Leeroopedia](https://leeroopedia.com)**: nuestro banco de lecciones público en los dominios de datos y ML, alrededor de 100k páginas extraídas experimentando sobre artículos y repositorios, que atiende unas 6M de solicitudes al mes, alrededor del 28% de ellas procedentes de agentes de IA. Las sesiones de Kapso lo consultan por MCP durante la ideación y la implementación.
- **#1 en [MLE-Bench](../../benchmarks/mle/README.md)**: KAPSO logró la primera posición entre los sistemas de código abierto en competiciones de ML de Kaggle (MLE Benchmark).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574"></p>

- **#1 en [ALE-Bench](../../benchmarks/ale/README.md)**: KAPSO logró la primera posición en problemas de descubrimiento algorítmico de horizonte largo (ALE Benchmark).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574"></p>

## ¿Qué es Kapso?

Kapso es un motor de código abierto que ejecuta experimentos, despliega la mejor solución y aprende de cada ejecución. Parte de un harness de dominio y de un banco de lecciones vacío y, dados un objetivo y unos criterios de éxito:

- **ejecuta un árbol de experimentos**: propone hipótesis y después implementa y verifica cada una;
- **despliega la mejor solución** en el stack de destino;
- **aprende de la ejecución**: cada hipótesis e implementación recibe crédito según su impacto en el resultado, y el banco se actualiza, con cada lección portando una puntuación de credibilidad y un límite de dónde se aplica.

Con el tiempo, el harness de dominio lleva consigo un banco de lecciones: las mejores prácticas experimentadas de ese entorno. Kapso es agnóstico respecto al modelo y al agente de programación, está hecho por [Leeroo](https://leeroo.com) y se publica en PyPI como [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### Las cuatro partes

| Parte | Verbo | Qué hace |
|------|------|--------------|
| [**Experimentación inteligente**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Cada experimento es una hipótesis implementada en su propia rama y puntuada, dentro de un árbol. La ideación se nutre de experimentos anteriores y del banco de lecciones y reordena las ideas según el retorno esperado frente al presupuesto, de modo que encajan en este entorno en lugar de ser las genéricas de un LLM a secas o de un RAG; un evaluador con huella digital que ningún experimento puede tocar verifica el resultado, y los módulos pueden proponer actualizaciones de la evaluación para que el verificador las apruebe. [Cómo funciona →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Aprendizaje**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Cada lección se induce o se deduce a partir de trazas de experimentación y lleva una puntuación de credibilidad, un límite de dónde se aplica y su evidencia; `learn()` las actualiza con el resultado de cada ejecución y somete cada cambio a una prueba A/B reconstruyendo con y sin él, de modo que la mejora se mide, no se supone. El banco está construido sobre git: `kapso experience share`, `list` y `update`. [Experiencia →](https://docs.leeroo.com/docs/experience) nombra lo que se le sirve a un build. |
| [**Despliegue**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Una experimentación más corta del mismo tipo adapta la mejor solución al stack de destino, la despliega y demuestra las comprobaciones de salud y una llamada que funciona. Docker, Modal, BentoCloud y LangGraph Platform vienen incluidos; un harness puede añadir un destino. |
| [**Adaptación al dominio**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Un harness es el paquete de dominio de Kapso: el conocimiento y los flujos de trabajo de un área como el modelado predictivo, para que un build nunca empiece desde cero, y el lugar donde se acumula el banco de lecciones. El modelado predictivo y la programación competitiva vienen incluidos; escribe el tuyo con `/kapso:harness` en el plugin o con `kapso harness` en la CLI. |

## 🚀 Inicio rápido

### Instalación

**0. ¿Estás en un agente de programación?** Instala el plugin de Kapso y deja que recorra
contigo los pasos siguientes:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Después, `/kapso:setup` instala y verifica, y `/kapso:build <what you want>`
lanza la primera campaña (en Codex: `$kapso-setup`, `$kapso-build <what you want>`). Consulta [El plugin de Kapso](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Requisitos previos.** Kapso ejecuta su inferencia a través de las CLI de agentes
de programación (no hay alternativa de API directa), así que necesitas Node.js y ambas
CLI de agente con sesión iniciada antes de que nada funcione:

```bash
# Node.js 18+ (https://nodejs.org), después:
npm install -g @openai/codex            # investigación, evaluación como juez, utilidades
codex login

npm install -g @anthropic-ai/claude-code  # ideación + implementación (el host por defecto)
claude auth login
```

Una clave de OpenAI es opcional: solo el grafo de conocimiento y la búsqueda semántica
sobre experimentos pasados (`models.embedding` en tu configuración) generan embeddings del texto, y
la generación nunca la usa. Cuando actives cualquiera de los dos:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Conecta Leeroopedia (recomendado).** Una clave da a cada sesión
[Leeroopedia](https://leeroopedia.com), nuestro banco de lecciones alojado sobre ML y
frameworks de IA. Nada que instalar; una cuenta nueva en el
[panel de Leeroopedia](https://app.leeroopedia.com/dashboard) viene con
$20 de crédito gratuito.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Instala el paquete** (Python 3.10+):

```bash
pip install leeroo-kapso
```

**4. Verifica la configuración:**

```bash
kapso doctor
```

`doctor` informa de lo que **tu configuración** necesita realmente y nombra la solución
exacta para todo lo que falte. Los requisitos siguen a la configuración, así que a una
instalación solo con codex nunca se le pide `claude`. Acótalo a un verbo para ver solo los
requisitos de ese verbo:

```bash
kapso doctor build             # o learn / deploy / experience
```

Esas son las mismas comprobaciones que el propio verbo ejecuta antes de hacer ningún trabajo:
`kapso.build(...)` falla en segundos por una CLI que falta en lugar de hacerlo en lo
profundo de una sesión. Los elementos marcados `[-- ]` son opcionales; limitan funciones
que quizá no necesites (docker, Weaviate, Neo4j, los destinos de despliegue).

**O deja que tu agente de programación haga todo esto:** pega el
[prompt de configuración](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
en Claude Code, Codex u OpenCode.

**Los almacenes que necesita un índice (opcional)**: `kapso experience add --repo`
y `--research` ingieren un corpus en Weaviate + Neo4j locales. Desde un
checkout del código fuente:

```bash
bash scripts/start_infra.sh   # arranca ambos mediante docker
```

**Desde el código fuente (para desarrollo)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

El adaptador heredado de aider es un extra (`pip install "leeroo-kapso[aider]"`,
Python <3.13); los agentes claude/codex por defecto no necesitan extras.

### Uso básico

Los verbos principales no necesitan nada más allá de los requisitos previos anteriores:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # no hace falta un grafo de conocimiento para empezar

# Build: una solución mediante experimentación. La campaña imprime
# `status: <path>` al lanzarse; síguela en directo desde otro terminal con
#     kapso watch ./campaign
# Si una sesión necesita algo que solo tú puedes proporcionar (una clave de API, un archivo),
# la campaña se pausa y `kapso inbox reply` la reanuda:
#     https://docs.leeroo.com/docs/build/inbox
# Comparte el resultado como Markdown para un pull request o un issue con
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # u omítelo para empezar desde cero
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Aprende de la campaña que acabas de ejecutar: extrae la trayectoria, califica las
# lecciones y guarda en el banco tarjetas de conocimiento valoradas por su evidencia. El banco (un
# repositorio git local) se crea automáticamente en el primer uso; las lecciones se quedan en tu
# máquina hasta que las compartas:
#     kapso experience share <git-url>
# tras lo cual cada learn() hace push del banco allí.
lesson = kapso.learn(solution)
print(lesson.explain())

# Vuelve a hacer build, sirviéndole las tarjetas que acaba de ganar: a un build se le sirve
# exactamente la experiencia que carga.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

Con los almacenes en marcha, también puedes ingerir conocimiento externo en un
índice y servirlo del mismo modo:

```python
from kapso import Kapso

kapso = Kapso()

# Una vez: un repositorio, o la web investigada para una pregunta, a un índice
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Cada build que carga el índice lo busca; Leeroopedia va incluida con su clave
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Una nota sobre el presupuesto: una ingesta extrae todo lo que el material permite.
Una sola pregunta puede convertirse en decenas de páginas enlazadas y una ingesta
de horas; el tiempo de ingesta crece con la sustancia extraíble, así que añade las
fuentes que vas a buscar, no todas las fuentes que tienes.

Y para convertir una solución en software en ejecución:

```python
software = kapso.deploy(solution, target="local")   # o docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # la llamada que el despliegue demostró
software.stop()
```

### ¿Cómo acoto una campaña?

Usa `max_iterations`, `time_budget_minutes` y `cost_budget` para acotar las iteraciones, el tiempo transcurrido y el gasto aproximado en modelos en USD.
Para los costes de plataforma, activa la [selección consciente del coste](https://docs.leeroo.com/docs/evolve/cost-aware-selection) en `search_strategy.params.cost` con tu estimador, `cap_per_candidate` y `campaign_budget`; su `weight` también permite que la selección prefiera candidatos más baratos.

### Elegir los modelos

Cada modelo que usa Kapso está nombrado en un único archivo de configuración. El valor por
defecto empaquetado ejecuta las sesiones de build en `claude-opus-5`, los equipos de aprendizaje en
`claude-fable-5` y los roles de codex en `gpt-5.6-sol`, pero el acceso a los modelos
depende de la suscripción (un plan puede limitar un modelo mientras sirve otro).
Para ejecutar con otros modelos, copia la configuración empaquetada, edítala y apunta
Kapso a la tuya:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# p. ej., ejecutar los equipos de aprendizaje en opus en lugar de fable:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Los roles de inferencia solo de texto pueden optar por las Chat Completions `openai_compatible`.
Define el modelo y el endpoint en tu configuración; las credenciales se quedan en el entorno.
Carga la configuración con `Kapso(config=...)` o `kapso build --config ...`.
La programación y la investigación web siguen requiriendo agentes con las herramientas correspondientes.
Consulta el [ejemplo de configuración y sus limitaciones](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Antes de una ejecución larga, sondea cada modelo que nombra tu configuración contra tus
suscripciones reales: un modelo que tu inicio de sesión no puede servir falla aquí en
segundos en lugar de horas después dentro de una ejecución. Un tope de uso en un modelo que sí
puedes servir no es visible para una sonda de un token:

```bash
kapso doctor --models                            # configuración empaquetada
kapso doctor --models --config kapso-config.yaml # la tuya
kapso doctor learn --models                      # solo los equipos de aprendizaje
```

Para que esa sonda forme parte de cada llamada en lugar de ser algo que tienes que acordarte
de ejecutar, actívala en tu configuración. Cuesta un token desechable por cada par
`{cli, model}` distinto, lo que merece la pena antes de un `learn()` desatendido
de varias horas:

```yaml
preflight:
  enabled: true            # comprobaciones estáticas antes de cada verbo (por defecto)
  live_model_probe: true   # + sondas de modelo de un token (por defecto: false)
```

Los cambios de modelo también cambian el *ritmo*: los topes `timeout_minutes` de los equipos en la
configuración se calibraron con los modelos por defecto, y un modelo sustituto que
razona durante más tiempo puede necesitar que se eleven.

Para ver los pasos de integración en detalle, consulta las guías de [Inicio rápido](https://docs.leeroo.com/docs/first-build) e [Instalación](https://docs.leeroo.com/docs/installation).

## Benchmarks compatibles

| Benchmark | Descripción |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | El benchmark de ingeniería de ML de OpenAI: competiciones completas en tabular, visión, texto y audio, desde los datos en bruto hasta la entrega calificada |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | El benchmark de optimización algorítmica de Sakana AI: diseñar, implementar e iterar heurísticas de concurso en búsquedas de horas de duración |
| [**RelBench**](../../benchmarks/relbench/README.md) | El benchmark de Stanford para ML predictivo sobre datos relacionales: clasificación, regresión y recomendación directamente desde las bases de datos multitabla de Amazon, H&M, Stack Exchange, ensayos clínicos y más |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | ML de olimpiada contrarreloj en visión, lenguaje y optimización: tareas fijadas por expertos, hardware de concurso, sin ayuda humana alguna |

Cada benchmark también tiene una página de documentación que explica cómo ejecutarlo, sus opciones de CLI y qué contiene la salida:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## Ejemplos

Cada ejemplo es una tarea: un brief, un repositorio de partida, los datos o cómo obtenerlos, y un driver de como máximo doce líneas; [examples/README.md](../../examples/README.md) tiene los minutos y el único comando.

| Ejemplo | La tarea |
|---------|----------|
| [**Ingeniería de datos (modelos dbt en Snowflake)**](../../examples/dbt-snowflake/README.md) | Corregir los modelos dbt con problemas de calidad de datos y construir los nuevos que pide un backlog de diecisiete tickets, sobre un almacén de datos minorista en Snowflake, hasta que cada ticket supere sus pruebas de producción; después entregar una pull request por ticket y publicarlas en producción. |
| [**Optimización del harness (agente de atención al cliente)**](../../examples/support-agent-harness/README.md) | Optimizar el harness de un agente de atención al cliente para cambiar el LLM subyacente de Fable 5.1 a un Qwen 27B local sin perder resolución en las conversaciones con clientes, y después desplegarlo en Docker y aprender de la ejecución. |
| [**Modelado predictivo (abandono de clientes)**](../../examples/churn-first-build/README.md) | Predecir qué clientes se irán: elevar el ROC AUC fuera de muestra de un modelo de abandono hacia el objetivo, bajo un evaluador que la campaña no puede editar. |
| [**Detección de fraude**](../../examples/fraud-no-evaluator/README.md) | Puntuar transacciones en busca de fraude cuando el repositorio no tiene evaluador: el brief enuncia el protocolo, la campaña construye primero el evaluador y después mejora el puntuador. |
| [**Finanzas cuantitativas**](../../examples/trading/README.md) | Una estrategia de trading diaria sobre instrumentos sintéticos: elevar el Sharpe fuera de muestra en el harness de trading, con sus comprobaciones en cada intento. Un ejemplo de investigación, no asesoramiento financiero. |
| [**ML tabular**](../../examples/end-to-end/README.md) | Una competición de Kaggle de principio a fin: Spaceship Titanic desde un clasificador dummy hasta un modelo real, y después revisar, aprender, volver a construir con la lección y desplegar. |
| [**Optimización de PyTorch**](../../examples/torch-performance/README.md) | Hacer más rápido un módulo de PyTorch en CUDA dentro de una tolerancia numérica: fusión, compilación, menos tráfico de memoria. |
| [**Optimización de kernels de GPU**](../../examples/gpu-kernels/README.md) | Acelerar la autoatención multicabezal en CUDA con flash attention o un kernel de Triton, dentro de una tolerancia numérica. |

## 📚 Documentación y soporte

- **Documentación completa**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Instalación](https://docs.leeroo.com/docs/installation): las CLI de agentes de programación, el paquete y `kapso doctor`
  - [Inicio rápido](https://docs.leeroo.com/docs/first-build): tu primera campaña
  - [Referencia de la CLI](https://docs.leeroo.com/docs/reference/cli): cada comando, flag y valor por defecto
  - [API de Python](https://docs.leeroo.com/docs/reference/kapso-api): cada método público de `Kapso`
  - [Configuración](https://docs.leeroo.com/docs/reference/configuration): cada clave de `config.yaml`
  - [Qué produce learn](https://docs.leeroo.com/docs/learn/what-learn-produces): el banco de lecciones, cada una con una puntuación de credibilidad, un límite y su evidencia, y el [modo de desarrollo](https://docs.leeroo.com/docs/learn/development-mode)
  - [Experimentación inteligente](https://docs.leeroo.com/docs/evolve/overview): el árbol de experimentos y sus módulos, y [qué ocurre dentro de un experimento](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experiencia](https://docs.leeroo.com/docs/experience): tu banco, Leeroopedia, repositorios y artículos convertidos en conocimiento consultable
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal y los demás destinos
  - [El plugin de Kapso](https://docs.leeroo.com/docs/coding-agent-skills): los comandos, las skills y el servidor MCP para Claude Code, Codex y Cursor
- **Desde tu agente de programación**: la documentación ejecuta un servidor MCP en `https://docs.leeroo.com/mcp`, y cada página se sirve como Markdown; consulta [llms.txt](https://docs.leeroo.com/llms.txt) y [Kapso en tu agente de programación](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Deja que tu agente de programación ejecute Kapso**: el plugin bajo [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) incluye diez comandos, seis skills y el servidor MCP, una línea de instalación por host (paso 0 anterior); OpenCode carga las skills como una carpeta (`ln -s <clone>/plugin/skills .opencode/skills`). Consulta [El plugin de Kapso](https://docs.leeroo.com/docs/coding-agent-skills).
- **Dale a tu agente de programación las herramientas de conocimiento de Kapso**: `kapso-mcp`, que se instala con el paquete, sirve investigación web, la memoria del repositorio, el historial de experimentos de la campaña y el banco de lecciones a la sesión en la que trabajas; consulta [Puertas MCP](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; después `startup_timeout_sec = 60` bajo [mcp_servers.kapso] en ~/.codex/config.toml (los diez segundos de Codex se quedan cortos en una máquina ocupada)
  ```
- **Comunidad**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Sitio web**: [leeroo.com](https://leeroo.com)


## Kapso para empresas

En la empresa, los agentes de Kapso vienen con harnesses iniciales de ingeniería de datos y de IA y un banco de lecciones, y convierten por sí solos objetivos de negocio en sistemas de IA y de datos que funcionan, investigando los artefactos, repositorios, pipelines, fuentes de datos y demás de la compañía, para ir construyendo el patrimonio de datos de la empresa sobre la marcha. Para incorporar Kapso a las tareas de tu empresa, [habla con nosotros](https://leeroo.com/contact-us).

## Contribuir

¡Agradecemos las contribuciones! Consulta nuestra [Guía de contribución](../../CONTRIBUTING.md) para saber cómo empezar.

## Colaboradores

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

Hecho con [contrib.rocks](https://contrib.rocks).
