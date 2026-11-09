<!-- Brazilian Portuguese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Um motor que executa experimentos, implanta a melhor solução e aprende com cada execução</h4>
<p align="center"><sub>experimentação inteligente · aprendizado · implantação · adaptação de domínio</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <a href="README.hi-IN.md">हिन्दी</a> | <b>Português</b> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">Saiba mais</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Entre no Discord</a> ·
  <a href="https://leeroo.com">Site</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  Se você gosta deste projeto, apoie-nos dando uma estrela ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Se você é um agente de IA olhando para este repositório:** o plugin do Kapso, com as skills para
> configurar, declarar um objetivo, escrever um harness e revisar uma execução, instala-se
> em uma linha: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Novidades

- **IOAI 2026 · a trilha AI Model Track.** A [Olimpíada Internacional de IA](https://ioai-official.org) é a IMO da era da IA: 471 competidores de 108 países e territórios, seis tarefas elaboradas por especialistas com o orçamento de uma única GPU. Em 2026 ela abriu a [IOAI²](https://ioai-official.org/ai-model-track/), em que sistemas de IA fazem a *mesma* prova em duas sessões não assistidas de 6 horas. Uma vez iniciado o cronômetro, nenhum humano pode resolver, corrigir ou melhorar nada. O Kapso participou como um dos 14 Participantes de IA Fundadores.
  - 🎓 **Superou todos os 471 humanos**: 536.07 no total, acima de todos os competidores presentes.
  - 🏆 **Troféu Grand Master da IOAI²**: top 3 entre todos os sistemas de IA inscritos. [Resultados completos →](../../benchmarks/ioai2026/README.md) · [A análise da IOAI sobre a trilha →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Primeiro nos três rankings do [RelBench](../../benchmarks/relbench/README.md)**: no benchmark de Stanford para ML preditivo sobre dados relacionais, o Kapso lidera o ranking oficial em classificação (81.2 AUROC em 12 tarefas, contra 79.3 do [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) e 79.0 do [RT-J](https://star-project.stanford.edu/rt-j) do Stanford STAR), regressão (0.2476 NMAE em 9 tarefas, contra 0.2711 do RT-J e 0.2847 do KumoRFM-v2, 8.7% menos erro) e recomendações (18.4 MAP em 10 tarefas, contra 9.3 do ID-GNN, a melhor outra entrada). Cada número é uma entrada no [ranking oficial do RelBench](https://star-project.stanford.edu/relbench/leaderboard/). O aprendizado mantido no harness superou o aprendizado embutido nos pesos do modelo, e atualizar um banco de lições custa muito menos do que retreinar um modelo de fundação tabular.

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)**: uma wiki das nossas lições nos domínios de dados e ML, cerca de 100k páginas extraídas por meio de experimentação em artigos e repositórios, atendendo a cerca de 6M de requisições por mês, cerca de 28% delas vindas de agentes de IA. O Kapso se conecta a ela via MCP: suas sessões a usam durante a ideação e a implementação para buscar conhecimento, montar planos e diagnosticar falhas.
- **#1 no [MLE-Bench](../../benchmarks/mle/README.md)**: o KAPSO alcançou a primeira colocação entre os sistemas de código aberto em competições de ML do Kaggle (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **#1 no [ALE-Bench](../../benchmarks/ale/README.md)**: o KAPSO alcançou a primeira colocação em problemas de descoberta algorítmica de longo horizonte (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## O que é o Kapso?

Entregar um resultado de IA ou de dados dentro de uma empresa significa aprender duas coisas por experimento. Nenhum indivíduo ou agente de IA sabe como um patrimônio de dados está conectado e é operado; a única forma de aprender isso é investigar os artefatos da empresa (repositórios, pipelines, documentação, fontes de dados), propor hipóteses, executar experimentos e aprender com eles, até que o comportamento de cada sistema e suas condições sejam compreendidos. E, uma vez que essa compreensão exista, a melhor solução ainda não pode ser escolhida de antemão, seja a tarefa otimizar um warehouse ou prever churn, então você experimenta e aprende até que uma se ajuste ao ambiente. O know-how que se ajusta ao ambiente se acumula, de modo muito parecido com o RL. O RAG faz o oposto: alimenta o agente com conhecimento geral que nunca foi testado nesse ambiente, todo ele com o mesmo peso, e esse conhecimento tem pouco valor até ter sido experimentado ali.

O Kapso é o motor que construímos para isso. Ele parte de um harness de domínio e de um banco de lições vazio; dados um objetivo e critérios de sucesso, executa uma árvore de experimentos, propondo hipóteses e depois implementando e verificando cada uma, e implanta a melhor solução na stack alvo. Em seguida aprende com a execução, atribuindo crédito a cada hipótese e implementação pelo seu impacto no resultado e atualizando o banco, onde cada lição carrega uma pontuação de credibilidade e uma fronteira de onde se aplica. Com o tempo, o harness de domínio passa a carregar um banco de lições que acumulou as melhores práticas experimentadas naquele ambiente. O sistema é agnóstico quanto ao modelo e quanto ao agente de codificação: código aberto, da [Leeroo](https://leeroo.com), publicado no PyPI como [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### As quatro partes

| Parte | Verbo | O que faz |
|------|------|--------------|
| [**Experimentação inteligente**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Módulos especializados compartilham uma única estrutura de dados, uma árvore de experimentos: cada nó registra a hipótese, a implementação (branch git e dificuldades técnicas), o orçamento usado e a distância até o objetivo. Para cada experimento, um módulo de ideação propõe ideias a partir de experimentos anteriores e do banco de lições, reclassifica-as pelo retorno esperado frente ao orçamento que cada uma usaria e envia as mais bem classificadas ao implementador, que constrói em uma branch bifurcada a partir do nó pai. Como o banco continua aprendendo o que funciona naquele ambiente exato, as ideias se tornam específicas a ele, em vez das genéricas que um LLM puro ou um RAG ofereceriam. Os resultados são verificados por um avaliador com fingerprint ao qual nenhum experimento tem acesso, e a avaliação também evolui: os módulos de experimentação podem propor atualizações às métricas e aos critérios para o verificador aprovar. [Como funciona →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Aprendizado**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Tudo no banco vem de experimentos: cada lição é induzida ou deduzida a partir dos rastros das execuções de experimentação, carrega uma pontuação de credibilidade, condições de fronteira sobre onde se aplica e a evidência por trás dela, e, após cada execução, `learn()` as atualiza a partir do efeito da lição no resultado final. Mudanças no banco também são experimentos: cada lição atualizada passa por um teste A/B executando o mesmo build com e sem ela, de modo que uma melhoria é medida, não adivinhada. O banco é construído sobre git, então você pode compartilhar sua experiência com outros (`kapso experience share`), listar os bancos disponíveis para você (`kapso experience list`) e atualizá-los (`kapso experience update`). [Experiência →](https://docs.leeroo.com/docs/experience) nomeia o que é servido a um build: seu banco, um banco que sua equipe compartilha, a Leeroopedia ou um corpus ingerido de um repositório ou de uma pergunta pesquisada. |
| [**Implantação**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Executa uma experimentação mais curta do mesmo tipo para adaptar a melhor solução à stack alvo: uma sessão adapta uma cópia do nó, a implanta e demonstra verificações de integridade e uma chamada funcionando. Docker, Modal, BentoCloud e LangGraph Platform vêm incluídos, e você pode adicionar sua própria stack alvo ao harness. |
| [**Adaptação de domínio**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harnesses são os pacotes de domínio do Kapso: conhecimento e fluxos de trabalho para uma área como modelagem preditiva, para que um build não experimente e aprenda do zero; modelagem preditiva e programação competitiva vêm incluídas com o Kapso. Uma execução parte de um harness de domínio e de um banco de lições vazio, e com o tempo o harness passa a carregar o banco que acumulou o que funcionou naquele ambiente. Escreva seu harness de domínio inicial, suas instruções, código, guardrails e verificações, com `/kapso:harness` no plugin ou `kapso harness` na CLI. |

## 🚀 Início rápido

### Instalação

**0. Em um agente de codificação?** Instale o plugin do Kapso e deixe-o percorrer os
passos abaixo com você:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Então `/kapso:setup` instala e verifica, e `/kapso:build <what you want>`
lança a primeira campanha (no Codex: `$kapso-setup`, `$kapso-build <what you want>`). Veja [O plugin do Kapso](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Pré-requisitos.** O Kapso executa sua inferência por meio das CLIs de agentes de codificação
(não há fallback para API direta), então você precisa do Node.js e de ambas as CLIs
de agente logadas antes que qualquer coisa funcione:

```bash
# Node.js 18+ (https://nodejs.org), depois:
npm install -g @openai/codex            # pesquisa, julgamento, utilitários
codex login

npm install -g @anthropic-ai/claude-code  # ideação + implementação (o host padrão)
claude auth login
```

Uma chave da OpenAI é opcional: apenas o grafo de conhecimento e a busca semântica
sobre experimentos passados (`models.embedding` na sua configuração) geram embeddings de texto, e
a geração nunca a usa. Quando você ativar um dos dois:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Conecte a Leeroopedia (recomendado).** Uma chave dá a cada sessão de campanha
a [Leeroopedia](https://leeroopedia.com), uma base de conhecimento hospedada de
frameworks de ML e IA: como um framework realmente se comporta, revisões de planos,
faixas de hiperparâmetros documentadas, diagnósticos de falhas. Nada a instalar;
uma nova conta no [painel da Leeroopedia](https://app.leeroopedia.com/dashboard)
vem com $20 de crédito gratuito.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Instale o pacote** (Python 3.10+):

```bash
pip install leeroo-kapso
```

O pacote é `leeroo-kapso`. O pacote do PyPI chamado `kapso` é uma ferramenta de
WhatsApp sem relação com este projeto, que instala um comando `kapso` próprio e oculta este aqui;
se ele estiver presente, execute `pip uninstall kapso` primeiro.

**4. Verifique a configuração:**

```bash
kapso doctor
```

`doctor` relata o que **sua configuração** realmente precisa e nomeia a correção
exata para qualquer coisa ausente. Os requisitos seguem a configuração, então de uma
instalação só com codex nunca se exige o `claude`. Restrinja-o a um verbo para ver apenas os
requisitos daquele verbo:

```bash
kapso doctor build             # ou learn / deploy / experience
```

Essas são as mesmas verificações que o próprio verbo executa antes de fazer qualquer trabalho:
`kapso.build(...)` falha em segundos por uma CLI ausente, em vez de falhar no fundo
de uma sessão. Itens marcados com `[-- ]` são opcionais; eles limitam recursos
de que você talvez não precise (docker, Weaviate, Neo4j, os alvos de implantação).

**Ou deixe seu agente de codificação fazer tudo isso:** cole o
[prompt de configuração](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
no Claude Code, no Codex ou no OpenCode.

**Os armazenamentos de que um índice precisa (opcional)**: `kapso experience add --repo`
e `--research` ingerem um corpus em Weaviate + Neo4j locais. A partir de um
checkout do código-fonte:

```bash
bash scripts/start_infra.sh   # inicia ambos via docker
```

**A partir do código-fonte (para desenvolvimento)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

O adaptador legado do aider é um extra (`pip install "leeroo-kapso[aider]"`,
Python <3.13); os agentes padrão claude/codex não precisam de extras.

### Uso básico

Os verbos centrais não precisam de nada além dos pré-requisitos acima:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # nenhum grafo de conhecimento é necessário para começar

# Build: uma solução por meio de experimentação. A campanha imprime
# `status: <path>` no lançamento; acompanhe-a ao vivo de outro terminal com
#     kapso watch ./campaign
# Se uma sessão precisar de algo que só você pode fornecer (uma chave de API, um arquivo),
# a campanha pausa e `kapso inbox reply` a retoma:
#     https://docs.leeroo.com/docs/build/inbox
# Compartilhe o resultado como Markdown para um pull request ou issue com
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # ou omita para começar do zero
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Aprenda com a campanha que você acabou de executar: minere a trajetória, avalie as
# lições e guarde no banco cartões de conhecimento precificados por evidência. O banco (um
# repositório git local) é criado automaticamente no primeiro uso; as lições ficam na sua
# máquina até que você as compartilhe:
#     kapso experience share <git-url>
# e, a partir daí, cada learn() envia o banco para lá.
lesson = kapso.learn(solution)
print(lesson.explain())

# Faça o build de novo, servido com os cartões que acabou de conquistar: a um build é servida
# exatamente a experiência que ele carrega.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

Com os armazenamentos em execução, você também pode ingerir conhecimento externo em um
índice e servi-lo da mesma forma:

```python
from kapso import Kapso

kapso = Kapso()

# Uma vez: um repositório, ou a web pesquisada para uma pergunta, em um índice
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Cada build que carrega o índice o pesquisa; a Leeroopedia vem junto com a própria chave
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Uma observação sobre orçamento: uma ingestão extrai tudo o que o material sustenta.
Uma única pergunta pode se tornar dezenas de páginas interligadas e uma ingestão
de horas; o tempo de ingestão cresce com a substância extraível, então adicione as
fontes que você vai pesquisar, não todas as fontes que você tem.

E para transformar uma solução em software em execução:

```python
software = kapso.deploy(solution, target="local")   # ou docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # a chamada que a implantação provou
software.stop()
```

### Como delimito uma campanha?

Use `max_iterations`, `time_budget_minutes` e `cost_budget` para limitar iterações, tempo decorrido e, em regime de melhor esforço, o gasto com modelos em USD.
Para custos de plataforma, ative a [seleção sensível a custo](https://docs.leeroo.com/docs/evolve/cost-aware-selection) em `search_strategy.params.cost` com seu estimador, `cap_per_candidate` e `campaign_budget`; seu `weight` também permite que a seleção prefira candidatos mais baratos.

### Escolhendo modelos

Cada modelo que o Kapso usa é nomeado em um único arquivo de configuração. O padrão empacotado
executa as sessões de build em `claude-opus-5`, as equipes de aprendizado em
`claude-fable-5` e os papéis do codex em `gpt-5.6-sol`; mas o acesso aos modelos
depende da assinatura (um plano pode limitar um modelo enquanto serve outro).
Para executar com outros modelos, copie a configuração empacotada, edite e aponte
o Kapso para a sua:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# ex.: execute as equipes de aprendizado no opus em vez do fable:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Papéis de inferência somente texto podem optar pelo Chat Completions `openai_compatible`.
Defina o modelo e o endpoint na sua configuração; as credenciais ficam no ambiente.
Carregue a configuração com `Kapso(config=...)` ou `kapso build --config ...`.
Codificação e pesquisa na web ainda exigem agentes com as ferramentas correspondentes.
Veja o [exemplo de configuração e os limites](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Antes de uma execução longa, sonde cada modelo que sua configuração nomeia contra suas
assinaturas reais: um modelo que seu login não consegue servir falha aqui em
segundos, em vez de horas dentro de uma execução. Um limite de uso em um modelo que você consegue
servir não é visível para uma sondagem de um token:

```bash
kapso doctor --models                            # configuração empacotada
kapso doctor --models --config kapso-config.yaml # a sua
kapso doctor learn --models                      # apenas as equipes de aprendizado
```

Para tornar essa sondagem parte de cada chamada, em vez de algo de que você precisa se lembrar de
executar, ative-a na sua configuração. Ela custa um token descartável por par
`{cli, model}` distinto, o que vale a pena antes de um `learn()` não assistido de várias horas:

```yaml
preflight:
  enabled: true            # verificações estáticas antes de cada verbo (padrão)
  live_model_probe: true   # + sondagens de modelo de um token (padrão: false)
```

Trocas de modelo também mudam o *ritmo*: os limites `timeout_minutes` das equipes na
configuração foram calibrados nos modelos padrão, e um modelo trocado que
raciocina por mais tempo pode exigir que eles sejam aumentados.

Para passos detalhados de integração, veja os guias de [Início rápido](https://docs.leeroo.com/docs/first-build) e [Instalação](https://docs.leeroo.com/docs/installation).

## Benchmarks suportados

| Benchmark | Descrição |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | O benchmark de engenharia de ML da OpenAI: competições completas em tabular, visão, texto e áudio, dos dados brutos à submissão avaliada |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | O benchmark de otimização algorítmica da Sakana AI: projetar, implementar e iterar heurísticas de competição ao longo de buscas de horas |
| [**RelBench**](../../benchmarks/relbench/README.md) | O benchmark de Stanford para ML preditivo sobre dados relacionais: classificação, regressão e recomendação direto dos bancos de dados multitabela da Amazon, H&M, Stack Exchange, ensaios clínicos e mais |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | ML olímpico cronometrado em visão, linguagem e otimização: tarefas definidas por especialistas, hardware de competição, zero ajuda humana |

Cada benchmark também tem uma página de documentação que cobre como executá-lo, suas opções de CLI e o que a saída contém:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## Exemplos

Cada exemplo é uma tarefa: um briefing, um repositório inicial, os dados ou como obtê-los, e um driver de no máximo doze linhas; [examples/README.md](../../examples/README.md) tem os minutos, o comando único e se um resultado está registrado.

| Exemplo | O que mostra |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | Seu primeiro build: um avaliador presente, uma linha de base, uma meta, o registro |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | Sem avaliador: o primeiro nó da campanha constrói um a partir do protocolo do briefing e cada candidato é medido contra ele |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | Um pipeline dbt no DuckDB cujas exportações carregam defeitos: o contrato de dados feito valer |
| [**trading**](../../examples/trading/README.md) | Um harness no build: uma sessão de configuração, páginas e hooks em cada sessão, verificação por tentativa |
| [**end to end**](../../examples/end-to-end/README.md) | Build, a revisão, learn, build de novo servido com a lição, deploy: um modelo tabular do Kaggle de ponta a ponta |
| [**prompt harness**](../../examples/prompt-harness/README.md) | O prompt como o objeto sob teste, pontuado em problemas da AIME |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | Um extrator de visão e linguagem, pontuado contra tabelas de referência |
| [**torch performance**](../../examples/torch-performance/README.md) | Um módulo mais rápido em CUDA sob uma tolerância numérica |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | Atenção multi-cabeça em CUDA: flash attention ou Triton |

## 📚 Documentação e Suporte

- **Documentação completa**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Instalação](https://docs.leeroo.com/docs/installation): as CLIs de agentes de codificação, o pacote e o `kapso doctor`
  - [Início rápido](https://docs.leeroo.com/docs/first-build): sua primeira campanha
  - [Referência da CLI](https://docs.leeroo.com/docs/reference/cli): cada comando, flag e padrão
  - [API Python](https://docs.leeroo.com/docs/reference/kapso-api): cada método público de `Kapso`
  - [Configuração](https://docs.leeroo.com/docs/reference/configuration): cada chave em `config.yaml`
  - [O que o learn produz](https://docs.leeroo.com/docs/learn/what-learn-produces): o banco de lições, cada uma com uma pontuação de credibilidade, uma fronteira e sua evidência, e o [modo de desenvolvimento](https://docs.leeroo.com/docs/learn/development-mode)
  - [Experimentação inteligente](https://docs.leeroo.com/docs/evolve/overview): a árvore de experimentos e seus módulos, e [o que acontece dentro de um experimento](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experiência](https://docs.leeroo.com/docs/experience): seu banco, a Leeroopedia, repositórios e artigos transformados em conhecimento pesquisável
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal e os outros alvos
  - [O plugin do Kapso](https://docs.leeroo.com/docs/coding-agent-skills): os comandos, as skills e o servidor MCP para Claude Code, Codex e Cursor
- **A partir do seu agente de codificação**: a documentação executa um servidor MCP em `https://docs.leeroo.com/mcp`, e cada página é servida como Markdown; veja [llms.txt](https://docs.leeroo.com/llms.txt) e [Kapso no seu agente de codificação](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Deixe seu agente de codificação executar o Kapso**: o plugin em [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) traz dez comandos, seis skills e o servidor MCP, uma linha de instalação por host (passo 0 acima); o OpenCode carrega as skills como uma pasta (`ln -s <clone>/plugin/skills .opencode/skills`). Veja [O plugin do Kapso](https://docs.leeroo.com/docs/coding-agent-skills).
- **Dê ao seu agente de codificação as ferramentas de conhecimento do Kapso**: o `kapso-mcp`, instalado com o pacote, serve pesquisa na web, a memória do repositório, o histórico de experimentos da campanha e o banco de lições à sessão em que você trabalha; veja [Portões MCP](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; depois `startup_timeout_sec = 60` em [mcp_servers.kapso] no ~/.codex/config.toml (os dez segundos do Codex são curtos em uma máquina ocupada)
  ```
- **Comunidade**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Site**: [leeroo.com](https://leeroo.com)


## Kapso para Empresas

Na empresa, os agentes Kapso vêm com harnesses iniciais de engenharia de dados e IA e um banco de lições, e transformam objetivos de negócio em sistemas de IA e dados funcionais por conta própria, investigando os artefatos da empresa, repositórios, pipelines, fontes de dados e mais, para construir o patrimônio de dados da empresa ao longo do caminho. Para adotar o Kapso nas tarefas da sua empresa, [fale conosco](https://leeroo.com/contact-us).

## Contribuindo

Contribuições são bem-vindas! Veja nosso [Guia de Contribuição](../../CONTRIBUTING.md) para detalhes sobre como começar.

## Contribuidores

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

Feito com [contrib.rocks](https://contrib.rocks).
