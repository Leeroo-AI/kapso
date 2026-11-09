<!-- Brazilian Portuguese translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">Um motor que executa experimentos, implanta a melhor solução e aprende com cada execução</h4>
<p align="center"><sub>experimentação inteligente · ciclo de aprendizado · implantação · adaptação de domínio</sub></p>

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
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **Se você é um agente de IA olhando este repositório:** o plugin do Kapso, com as skills para
> configurar, definir um objetivo, escrever um harness e revisar uma execução, é instalado
> em uma linha: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## Novidades

- **IOAI 2026 · a Trilha de Modelos de IA.** A [Olimpíada Internacional de IA](https://ioai-official.org) é a IMO da era da IA: 471 competidores de 108 países e territórios, seis tarefas elaboradas por especialistas com um orçamento de uma única GPU. Em 2026 ela abriu a [IOAI²](https://ioai-official.org/ai-model-track/), em que sistemas de IA fazem a *mesma* prova em duas sessões de 6 horas sem supervisão. Depois que o relógio começa a correr, nenhum humano pode resolver, corrigir ou melhorar nada. O Kapso entrou como um dos 14 Participantes de IA Fundadores.
  - 🎓 **Superou todos os 471 humanos**: 536.07 no total, acima de todos os competidores presentes no salão.
  - 🏆 **Troféu Grand Master da IOAI²**: top 3 entre todos os sistemas de IA participantes. [Resultados completos →](../../benchmarks/ioai2026/README.md) · [A análise da IOAI sobre a trilha →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **Primeiro lugar nos três rankings do [RelBench](../../benchmarks/relbench/README.md)**: no benchmark de Stanford para ML preditivo sobre dados relacionais, o Kapso lidera o ranking oficial em classificação (81.2 AUROC em 12 tarefas, contra 79.3 do [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) e 79.0 do [RT-J](https://star-project.stanford.edu/rt-j) do Stanford STAR), regressão (0.2476 NMAE em 9 tarefas, contra 0.2711 do RT-J e 0.2847 do KumoRFM-v2, 8.7% a menos de erro) e recomendações (18.4 MAP em 10 tarefas, contra 9.3 do ID-GNN, a melhor entre as outras entradas). Cada número é uma entrada no [ranking oficial do RelBench](https://star-project.stanford.edu/relbench/leaderboard/).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)**: uma wiki das nossas lições nos domínios de dados e ML, com cerca de 100k páginas extraídas por experimentação com artigos e repositórios, atendendo cerca de 6M de requisições por mês, cerca de 28% delas vindas de agentes de IA. O Kapso se conecta a ela via MCP: suas sessões a usam durante a ideação e a implementação para buscar conhecimento, montar planos e diagnosticar falhas.
- **[Agentes do Moltbook 🦞](https://www.moltbook.com/)**: Crie agentes de IA que otimizam outros agentes e debatem no Moltbook! [Comece agora →](../../moltbook_bot/README.md)
- **Relatório técnico**: Nosso relatório técnico já está disponível! [Leia o artigo](https://arxiv.org/abs/2601.21526)
- **#1 no [MLE-Bench](../../benchmarks/mle/README.md)**: o KAPSO alcançou a primeira posição entre os sistemas de código aberto em competições de ML do Kaggle (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **#1 no [ALE-Bench](../../benchmarks/ale/README.md)**: o KAPSO alcançou a primeira posição em problemas de descoberta algorítmica de longo horizonte (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## O que é o Kapso?

O Kapso é um motor que executa experimentos, implanta a melhor solução e aprende com cada execução. Dados um objetivo, critérios de sucesso iniciais e métricas de avaliação, ele investiga, executa experimentos escolhendo as melhores ideias, às vezes dá feedback para melhorar as métricas de avaliação e os critérios de sucesso, implanta a melhor solução na stack desejada e aprende com cada execução. Ele mantém um banco de lições, cada uma com uma pontuação de credibilidade e uma fronteira que indica onde ela se aplica, de modo que o conhecimento se potencializa com o tempo, em vez de apenas se acumular.

Construímos sistemas de IA dentro de grandes empresas, incluindo LinkedIn e Meta AI, e os implantamos em produtos com milhões de usuários. O Kapso é esse fluxo de trabalho na forma de um motor: de código aberto, da [Leeroo](https://leeroo.com), publicado no PyPI como [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### As quatro partes

| Parte | Verbo | O que faz |
|------|------|--------------|
| [**Experimentação inteligente**](https://docs.leeroo.com/docs/build/brief) | `.build()` | O ciclo de experimentação é um conjunto de módulos especializados que compartilham uma única estrutura de dados, uma árvore de experimentos: cada nó registra a hipótese, a implementação (branch do git e dificuldades técnicas), o orçamento usado e a distância até o objetivo. A ideação se apoia nos experimentos anteriores e no banco de lições para propor ideias, cada uma dizendo em que difere de experimentos anteriores específicos e quais lições adota; um reranker as ordena pelo retorno esperado em relação à meta, ponderado pelo tempo e pelo orçamento. O implementador constrói em uma branch criada a partir do nó pai, e um verificador confere o nó em relação às regras de segurança e aos critérios de sucesso, decide se a pontuação é válida e se o objetivo foi atingido. Os critérios de sucesso e os testes podem evoluir durante uma execução: os módulos propõem atualizações justificadas, e o verificador as aprova ou rejeita. [Como funciona →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Ciclo de aprendizado**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Cada lição registra um conhecimento, uma pontuação de credibilidade, condições de fronteira que indicam onde ela se aplica e as evidências por trás de suas atualizações. Após cada execução, `learn()` atualiza a credibilidade e as fronteiras das lições usadas a partir do efeito delas no resultado, induz lições a partir de traces semelhantes ou deduz uma a partir de uma única evidência e de seu mecanismo, mescla lições sobrepostas e faz testes A/B das lições atualizadas executando o mesmo build com e sem elas, de modo que as atualizações são medidas em vez de adivinhadas. A ideação e a implementação do build seguinte leem a credibilidade e as fronteiras de cada lição para decidir quando confiar nela, em vez de simplesmente fazer RAG do conhecimento. [Experiência →](https://docs.leeroo.com/docs/experience) define o que é servido a um build: o seu banco, um banco compartilhado pela sua equipe, a Leeroopedia ou um corpus ingerido de um repositório ou de uma pergunta pesquisada. |
| [**Implantação**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Transforma a melhor solução da árvore em software em execução na stack de destino, por meio de uma versão mais curta do ciclo de experimentação: uma sessão adapta uma cópia do nó, faz a implantação e demonstra verificações de saúde e uma chamada funcionando. Docker, Modal, BentoCloud e LangGraph Platform já vêm incluídos, e você pode adicionar sua própria stack de destino ao harness. |
| [**Adaptação de domínio**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Os harnesses são os pacotes de domínio do Kapso: conhecimento e fluxos de trabalho para uma área como modelagem preditiva, para que um build não experimente e aprenda do zero; modelagem preditiva e programação competitiva já vêm com o Kapso. Para trazer seu próprio conhecimento de domínio, escreva um harness pela interface que o Kapso define: suas instruções, código, guardrails e verificações, criados com `/kapso:harness` no plugin ou `kapso harness` na CLI. |

## 🚀 Início rápido

### Instalação

**0. Está em um agente de codificação?** Instale o plugin do Kapso e deixe que ele percorra os
passos abaixo com você:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Depois, `/kapso:setup` instala e verifica, e `/kapso:build <what you want>`
lança a primeira campanha (no Codex: `$kapso-setup`, `$kapso-build <what you want>`). Veja [O plugin do Kapso](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Pré-requisitos.** O Kapso faz sua inferência por meio das CLIs de agentes de codificação
(não há fallback para API direta), então você precisa do Node.js e das duas CLIs de
agente autenticadas antes que qualquer coisa funcione:

```bash
# Node.js 18+ (https://nodejs.org), depois:
npm install -g @openai/codex            # pesquisa, julgamento, utilitários
codex login

npm install -g @anthropic-ai/claude-code  # ideação + implementação (o host padrão)
claude auth login
```

Uma chave da OpenAI é opcional: apenas o grafo de conhecimento e a busca semântica
sobre experimentos passados (`models.embedding` na sua config) geram embeddings de texto, e
a geração nunca a usa. Quando você ativar qualquer um dos dois:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Conecte a Leeroopedia (recomendado).** Uma chave dá a cada sessão de
campanha a [Leeroopedia](https://leeroopedia.com), uma base de conhecimento hospedada sobre
frameworks de ML e IA: como um framework realmente se comporta, revisões de planos,
faixas de hiperparâmetros documentadas, diagnósticos de falhas. Nada para instalar;
uma conta nova no [painel da Leeroopedia](https://app.leeroopedia.com/dashboard)
vem com $20 de crédito gratuito.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Instale o pacote** (Python 3.10+):

```bash
pip install leeroo-kapso
```

O pacote é `leeroo-kapso`. O pacote do PyPI chamado `kapso` é uma ferramenta de
WhatsApp sem relação com este projeto, que instala um comando `kapso` próprio e encobre este;
se ele estiver presente, execute `pip uninstall kapso` primeiro.

**4. Verifique a instalação:**

```bash
kapso doctor
```

`doctor` informa o que **a sua config** realmente precisa e indica a correção
exata para o que estiver faltando. Os requisitos seguem a config, então uma configuração
só com codex nunca é cobrada pelo `claude`. Restrinja a um verbo para ver apenas os
requisitos desse verbo:

```bash
kapso doctor build             # ou learn / deploy / experience
```

Essas são as mesmas verificações que o próprio verbo executa antes de fazer qualquer trabalho:
`kapso.build(...)` falha em segundos por uma CLI ausente, e não lá no fundo
de uma sessão. Os itens marcados com `[-- ]` são opcionais; eles limitam recursos
de que você talvez não precise (docker, Weaviate, Neo4j, os destinos de implantação).

**Ou deixe seu agente de codificação fazer tudo isso:** cole o
[prompt de configuração](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
no Claude Code, Codex ou OpenCode.

**Os armazenamentos de dados de que um índice precisa (opcional)**: `kapso experience add --repo`
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

O ciclo principal não precisa de nada além dos pré-requisitos acima:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # nenhum grafo de conhecimento é necessário para começar

# Build: uma solução por meio de experimentação. A campanha imprime
# `status: <path>` ao iniciar; acompanhe ao vivo de outro terminal com
#     kapso watch ./campaign
# Se uma sessão precisar de algo que só você pode fornecer (uma chave de API, um arquivo),
# a campanha pausa e `kapso inbox reply` a retoma:
#     https://docs.leeroo.com/docs/build/inbox
# Compartilhe o resultado em Markdown para um pull request ou issue com
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # ou omita para começar do zero
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Aprenda com a campanha que você acabou de rodar: minere a trajetória, dê notas às
# lições e guarde no banco cartões de conhecimento precificados pela evidência. O banco (um
# repositório git local) é criado automaticamente no primeiro uso; as lições ficam na sua
# máquina até você compartilhá-las:
#     kapso experience share <git-url>
# depois disso, cada learn() envia o banco para lá.
lesson = kapso.learn(solution)
print(lesson.explain())

# Faça o build de novo, servido com os cartões que acabou de conquistar: um build recebe
# exatamente a experiência que carrega.
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

# Todo build que carrega o índice faz buscas nele; a Leeroopedia funciona pela chave dela
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

Uma observação sobre orçamento: uma ingestão extrai tudo o que o material comporta.
Uma única pergunta pode virar dezenas de páginas interligadas e uma ingestão
de horas; o tempo de ingestão cresce com o conteúdo extraível, então adicione as
fontes em que você vai buscar, não todas as fontes que você tem.

E para transformar uma solução em software em execução:

```python
software = kapso.deploy(solution, target="local")   # ou docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # a chamada que a implantação comprovou
software.stop()
```

### Como limito uma campanha?

Use `max_iterations`, `time_budget_minutes` e `cost_budget` para limitar as iterações, o tempo decorrido e o gasto com modelos em USD, este último em regime de melhor esforço.
Para custos de plataforma, ative a [seleção sensível a custo](https://docs.leeroo.com/docs/evolve/cost-aware-selection) em `search_strategy.params.cost` com o seu estimador, `cap_per_candidate` e `campaign_budget`; o `weight` dela também permite que a seleção prefira candidatos mais baratos.

### Escolhendo os modelos

Todo modelo que o Kapso usa é nomeado em um único arquivo de config. O padrão empacotado
roda as sessões de build no `claude-opus-5`, as equipes de aprendizado no
`claude-fable-5` e os papéis do codex no `gpt-5.6-sol`, mas o acesso aos modelos
depende da assinatura (um plano pode limitar um modelo enquanto atende outro).
Para rodar com modelos diferentes, copie a config empacotada, edite e aponte
o Kapso para a sua:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# p. ex., rodar as equipes de aprendizado no opus em vez do fable:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Os papéis de inferência somente texto podem optar pelo Chat Completions `openai_compatible`.
Defina o modelo e o endpoint na sua config; as credenciais ficam no ambiente.
Carregue a config com `Kapso(config=...)` ou `kapso build --config ...`.
Codificação e pesquisa na web ainda exigem agentes com as ferramentas correspondentes.
Veja o [exemplo de configuração e os limites](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Antes de uma execução longa, sonde cada modelo que sua config nomeia contra as suas
assinaturas reais: um modelo que o seu login não consegue atender falha aqui em
segundos, em vez de horas depois do início de uma execução. Um limite de uso em um modelo que você
consegue atender não é visível para uma sondagem de um token:

```bash
kapso doctor --models                            # config empacotada
kapso doctor --models --config kapso-config.yaml # a sua
kapso doctor learn --models                      # só as equipes de aprendizado
```

Para que essa sondagem faça parte de toda chamada, em vez de ser algo que você precisa lembrar de
rodar, ative-a na sua config. Ela custa um token descartável por par
`{cli, model}` distinto, o que vale a pena antes de um `learn()` de várias horas
sem supervisão:

```yaml
preflight:
  enabled: true            # verificações estáticas antes de cada verbo (padrão)
  live_model_probe: true   # + sondagens de modelo de um token (padrão: false)
```

Trocar de modelo também muda o *ritmo*: os tetos de `timeout_minutes` das equipes na
config foram calibrados com os modelos padrão, e um modelo substituto que
raciocina por mais tempo pode exigir que eles sejam aumentados.

Para os passos detalhados de integração, veja os guias de [Início rápido](https://docs.leeroo.com/docs/first-build) e [Instalação](https://docs.leeroo.com/docs/installation).

## Exemplos

Cada exemplo é uma tarefa: um brief, um repositório inicial, os dados ou como obtê-los, e um driver de no máximo doze linhas; [examples/README.md](../../examples/README.md) traz os minutos, o comando único e se há um resultado registrado.

| Exemplo | O que mostra |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | Seu primeiro build: um avaliador presente, uma baseline, uma meta, o registro |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | Sem avaliador: o primeiro nó da campanha constrói um a partir do protocolo do brief e cada candidato é medido em relação a ele |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | Um pipeline dbt no DuckDB cujas exportações trazem defeitos: o contrato de dados passa a ser cumprido |
| [**trading**](../../examples/trading/README.md) | Um harness no build: uma sessão de configuração, páginas e hooks em cada sessão, verificação por tentativa |
| [**full loop**](../../examples/full-loop/README.md) | Build, a revisão, learn, build de novo servido com a lição, deploy: um modelo tabular do Kaggle de ponta a ponta |
| [**prompt harness**](../../examples/prompt-harness/README.md) | O prompt como o objeto em teste, pontuado em problemas da AIME |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | Um extrator de visão e linguagem, pontuado em relação a tabelas de referência |
| [**torch performance**](../../examples/torch-performance/README.md) | Um módulo mais rápido em CUDA dentro de uma tolerância numérica |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | Atenção multi-head em CUDA: flash attention ou Triton |

## Benchmarks suportados

| Benchmark | Descrição |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | O benchmark de engenharia de ML da OpenAI: competições completas de dados tabulares, visão, texto e áudio, dos dados brutos à submissão avaliada |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | O benchmark de otimização algorítmica da Sakana AI: projetar, implementar e iterar heurísticas de competição em buscas de várias horas |
| [**RelBench**](../../benchmarks/relbench/README.md) | O benchmark de Stanford para ML preditivo sobre dados relacionais: classificação, regressão e recomendação direto dos bancos de dados multitabela da Amazon, H&M, Stack Exchange, ensaios clínicos e outros |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | ML de olimpíada com tempo cronometrado em visão, linguagem e otimização: tarefas definidas por especialistas, hardware de competição, zero ajuda humana |

Cada benchmark também tem uma página de documentação que cobre como executá-lo, suas opções de CLI e o que a saída contém:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 Documentação e suporte

- **Documentação completa**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Instalação](https://docs.leeroo.com/docs/installation): as CLIs de agentes de codificação, o pacote e `kapso doctor`
  - [Início rápido](https://docs.leeroo.com/docs/first-build): sua primeira campanha
  - [Referência da CLI](https://docs.leeroo.com/docs/reference/cli): cada comando, flag e valor padrão
  - [API Python](https://docs.leeroo.com/docs/reference/kapso-api): cada método público de `Kapso`
  - [Configuração](https://docs.leeroo.com/docs/reference/configuration): cada chave em `config.yaml`
  - [O que o learn produz](https://docs.leeroo.com/docs/learn/what-learn-produces): o banco de lições, cada uma com uma pontuação de credibilidade, uma fronteira e suas evidências, e o [modo de desenvolvimento](https://docs.leeroo.com/docs/learn/development-mode)
  - [O ciclo de experimentação](https://docs.leeroo.com/docs/evolve/overview): a árvore de experimentos e seus módulos, e [o que acontece dentro de um experimento](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experiência](https://docs.leeroo.com/docs/experience): o seu banco, a Leeroopedia, repositórios e artigos transformados em conhecimento pesquisável
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal e os outros destinos
  - [O plugin do Kapso](https://docs.leeroo.com/docs/coding-agent-skills): os comandos, as skills e o servidor MCP para Claude Code, Codex e Cursor
- **A partir do seu agente de codificação**: a documentação roda um servidor MCP em `https://docs.leeroo.com/mcp`, e cada página é servida como Markdown; veja [llms.txt](https://docs.leeroo.com/llms.txt) e [Kapso no seu agente de codificação](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Deixe seu agente de codificação rodar o Kapso**: o plugin em [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) traz dez comandos, seis skills e o servidor MCP, uma linha de instalação por host (passo 0 acima); o OpenCode carrega as skills como uma pasta (`ln -s <clone>/plugin/skills .opencode/skills`). Veja [O plugin do Kapso](https://docs.leeroo.com/docs/coding-agent-skills).
- **Dê ao seu agente de codificação as ferramentas de conhecimento do Kapso**: `kapso-mcp`, instalado com o pacote, serve pesquisa na web, a memória do repositório, o histórico de experimentos da campanha e o banco de lições para a sessão em que você trabalha; veja [gates MCP](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; depois `startup_timeout_sec = 60` em [mcp_servers.kapso] no ~/.codex/config.toml (os dez segundos do Codex são curtos em uma máquina ocupada)
  ```
- **Comunidade**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Site**: [leeroo.com](https://leeroo.com)


## Kapso para empresas

Nosso produto empresarial é construído sobre este motor e já foi implantado dentro de empresas complexas e desorganizadas, onde nenhum humano consegue investigar e entender todo o patrimônio de dados: migramos uma empresa de sua antiga plataforma de dados para um novo lakehouse, abrangendo mais de 30 famílias de sistemas de origem, milhares de tabelas e transformações e centenas de relatórios de BI. Cada tarefa em que ele trabalha alimenta o banco de lições sobre os seus sistemas, os seus dados e as suas regras conquistadas a duras penas, de modo que ele fica melhor na sua empresa quanto mais tempo trabalha. Para adotar o Kapso nas suas tarefas empresariais desafiadoras, [fale com a gente](https://leeroo.com/contact-us).

## Traduções

O README está disponível em outros idiomas: use o seletor no topo ou navegue por [`docs/translations/`](../../docs/translations/). Cada tradução é gerada a partir do README.md com `scripts/translate_readme.py` e mantida em sincronia por `scripts/check_readme_translations.py`, que falha quando uma tradução diverge do README.md em um link, um número ou uma linha de código. Para melhorar uma delas, abra um pull request para o arquivo nessa pasta.

## Contribuindo

Contribuições são bem-vindas! Consulte nosso [Guia de contribuição](../../CONTRIBUTING.md) para detalhes sobre como começar.

## Citação

Se você usar o Kapso no seu trabalho, cite o relatório técnico:

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
