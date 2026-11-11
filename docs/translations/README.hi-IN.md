<!-- Hindi translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">एक इंजन जो प्रयोग चलाता है, सबसे अच्छे समाधान को परिनियोजित करता है, और हर रन से सीखता है</h4>
<p align="center"><sub>स्मार्ट प्रयोग · सीखना · परिनियोजन · डोमेन अनुकूलन</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<a href="../../README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja-JP.md">日本語</a> | <a href="README.ko-KR.md">한국어</a> | <a href="README.de-DE.md">Deutsch</a> | <a href="README.fr-FR.md">Français</a> | <a href="README.es-ES.md">Español</a> | <b>हिन्दी</b> | <a href="README.pt-BR.md">Português</a> | <a href="README.ru-RU.md">Русский</a> | <a href="README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">और जानें</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Discord से जुड़ें</a> ·
  <a href="https://leeroo.com">वेबसाइट</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  अगर आपको यह प्रोजेक्ट पसंद है, तो कृपया इसे एक स्टार ⭐ देकर हमारा समर्थन करें
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **अगर आप इस रिपो को देख रहे एक AI एजेंट हैं:** Kapso प्लगइन, जिसमें सेटअप करने, उद्देश्य बताने,
> harness लिखने और रन की समीक्षा करने के स्किल शामिल हैं, एक पंक्ति में इंस्टॉल
> होता है: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`)।

---

## समाचार

- **IOAI 2026 · AI मॉडल ट्रैक।** [अंतर्राष्ट्रीय AI ओलंपियाड](https://ioai-official.org) AI युग का IMO है: 471 प्रतियोगी, छह विशेषज्ञ-निर्मित कार्य, एकल-GPU बजट। इसके 2026 के [IOAI²](https://ioai-official.org/ai-model-track/) ट्रैक ने AI सिस्टमों को एक स्वायत्त सत्र में वही परीक्षा दी। Kapso ने, 14 संस्थापक AI प्रतिभागियों में से एक के रूप में, आयोजकों की ग्रेडिंग के अनुसार 536.07 स्कोर किया, जो हर मानव प्रतियोगी से ऊपर है, और IOAI² ग्रैंड मास्टर ट्रॉफी जीती, यानी प्रवेश किए गए AI सिस्टमों में शीर्ष 3 में। [पूरे परिणाम →](../../benchmarks/ioai2026/README.md) · [ट्रैक पर IOAI की समीक्षा →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **तीनों [RelBench](../../benchmarks/relbench/README.md) लीडरबोर्ड पर पहले स्थान पर।** संबंधपरक डेटा पर भविष्यसूचक ML के लिए Stanford के बेंचमार्क पर, Kapso [आधिकारिक लीडरबोर्ड](https://star-project.stanford.edu/relbench/leaderboard/) में वर्गीकरण (81.2 AUROC, जबकि [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) का 79.3 और Stanford STAR के [RT-J](https://star-project.stanford.edu/rt-j) का 79.0), प्रतिगमन (0.2476 NMAE, जबकि 0.2711 और 0.2847) और सिफ़ारिशों (18.4 MAP, जबकि ID-GNN, अन्य सबसे अच्छी प्रविष्टि, का 9.3) में आगे है।

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0 in classification; 0.2476 NMAE against 0.2711 and 0.2847 in regression, lower being better; 18.4 MAP against ID-GNN's 9.3 in recommendations. The first two panels use a truncated axis, the third starts at zero" width="574"></p>

- **[Leeroopedia](https://leeroopedia.com)**: डेटा और ML डोमेन में सबकों का हमारा सार्वजनिक बैंक, लगभग 100k पृष्ठ जो पेपरों और रिपॉजिटरियों पर प्रयोग करके निकाले गए हैं, जो महीने में लगभग 6M अनुरोध परोसता है, जिनमें से लगभग 28% AI एजेंटों से आते हैं। Kapso के सत्र विचार-निर्माण और कार्यान्वयन के दौरान MCP पर इससे क्वेरी करते हैं।
- **[MLE-Bench](../../benchmarks/mle/README.md) पर #1**: KAPSO ने Kaggle ML प्रतियोगिताओं (MLE बेंचमार्क) पर ओपन-सोर्स सिस्टमों में शीर्ष रैंकिंग हासिल की।

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574"></p>

- **[ALE-Bench](../../benchmarks/ale/README.md) पर #1**: KAPSO ने लंबी अवधि की एल्गोरिदमिक खोज समस्याओं (ALE बेंचमार्क) पर शीर्ष रैंकिंग हासिल की।

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574"></p>

## Kapso क्या है?

Kapso एक ओपन-सोर्स इंजन है जो प्रयोग चलाता है, सबसे अच्छे समाधान को परिनियोजित करता है, और हर रन से सीखता है। यह एक डोमेन harness और सबकों के एक खाली बैंक से शुरू होता है, और एक उद्देश्य तथा सफलता मानदंड दिए जाने पर यह:

- **प्रयोगों का एक वृक्ष चलाता है**: परिकल्पनाएँ प्रस्तावित करता है, फिर हर एक को लागू और सत्यापित करता है;
- **सबसे अच्छे समाधान को लक्ष्य स्टैक पर परिनियोजित करता है**;
- **रन से सीखता है**: हर परिकल्पना और कार्यान्वयन को परिणाम पर उसके प्रभाव के अनुसार श्रेय मिलता है, और बैंक अपडेट होता है, जहाँ हर सबक एक विश्वसनीयता स्कोर और यह सीमा साथ रखता है कि वह कहाँ लागू होता है।

समय के साथ डोमेन harness सबकों का एक बैंक रखता है: उस परिवेश की प्रयोग से सिद्ध सर्वोत्तम प्रथाएँ। Kapso मॉडल-अज्ञेय और कोडिंग-एजेंट-अज्ञेय है, [Leeroo](https://leeroo.com) द्वारा, PyPI पर [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) के रूप में प्रकाशित।

### चार भाग

| भाग | क्रिया | यह क्या करता है |
|------|------|--------------|
| [**स्मार्ट प्रयोग**](https://docs.leeroo.com/docs/build/brief) | `.build()` | हर प्रयोग एक परिकल्पना है जो अपनी ही शाखा पर लागू की जाती है और स्कोर की जाती है, एक वृक्ष में। विचार-निर्माण पहले के प्रयोगों और सबकों के बैंक से प्रेरणा लेता है और बजट के मुकाबले अपेक्षित लाभ के अनुसार विचारों को पुनः क्रमित करता है, ताकि वे किसी नंगे LLM या RAG के सामान्य विचारों के बजाय इसी परिवेश के अनुकूल हों; एक फ़िंगरप्रिंट किया हुआ मूल्यांकनकर्ता, जिसे कोई प्रयोग छू नहीं सकता, परिणाम को सत्यापित करता है, और मॉड्यूल सत्यापनकर्ता की मंज़ूरी के लिए मूल्यांकन अपडेट प्रस्तावित कर सकते हैं। [यह कैसे काम करता है →](https://docs.leeroo.com/docs/evolve/overview) |
| [**सीखना**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | हर सबक प्रयोग के ट्रेस से आगमित या निगमित किया जाता है और एक विश्वसनीयता स्कोर, यह सीमा कि वह कहाँ लागू होता है, और अपना साक्ष्य साथ रखता है; `learn()` हर रन के परिणाम से उन्हें अपडेट करता है और हर बदलाव का A/B परीक्षण उसके साथ और उसके बिना पुनर्निर्माण करके करता है, ताकि सुधार मापा जाए, अनुमान न लगाया जाए। बैंक git पर बना है: `kapso experience share`, `list` और `update`। [अनुभव →](https://docs.leeroo.com/docs/experience) बताता है कि किसी build को क्या परोसा जाता है। |
| [**परिनियोजन**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | उसी तरह का एक छोटा प्रयोग सबसे अच्छे समाधान को लक्ष्य स्टैक के अनुकूल बनाता है, उसे परिनियोजित करता है और स्वास्थ्य जाँचों तथा एक कार्यशील कॉल को सिद्ध करता है। Docker, Modal, BentoCloud और LangGraph Platform साथ आते हैं; कोई harness एक लक्ष्य जोड़ सकता है। |
| [**डोमेन अनुकूलन**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | एक harness Kapso का डोमेन पैकेज है: किसी क्षेत्र, जैसे भविष्यसूचक मॉडलिंग, का ज्ञान और वर्कफ़्लो, ताकि कोई build कभी शून्य से शुरू न हो, और जहाँ सबकों का बैंक जमा होता है। भविष्यसूचक मॉडलिंग और प्रतिस्पर्धी प्रोग्रामिंग साथ आते हैं; अपना harness प्लगइन में `/kapso:harness` या CLI में `kapso harness` से लिखें। |

## 🚀 त्वरित शुरुआत

### इंस्टॉलेशन

**0. किसी कोडिंग एजेंट में हैं?** Kapso प्लगइन इंस्टॉल करें और उसे नीचे दिए
चरणों में आपके साथ चलने दें:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

फिर `/kapso:setup` इंस्टॉल और सत्यापन करता है, और `/kapso:build <what you want>`
पहला अभियान शुरू करता है (Codex पर: `$kapso-setup`, `$kapso-build <what you want>`)। देखें [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills)।

**1. पूर्वापेक्षाएँ।** Kapso अपना इन्फ़रेंस कोडिंग-एजेंट CLI के ज़रिए चलाता है
(कोई प्रत्यक्ष-API फ़ॉलबैक नहीं है), इसलिए कुछ भी काम करने से पहले आपको Node.js और दोनों
एजेंट CLI लॉग इन किए हुए चाहिए:

```bash
# Node.js 18+ (https://nodejs.org), फिर:
npm install -g @openai/codex            # शोध, निर्णय, उपयोगिताएँ
codex login

npm install -g @anthropic-ai/claude-code  # विचार-निर्माण + कार्यान्वयन (डिफ़ॉल्ट होस्ट)
claude auth login
```

एक OpenAI कुंजी वैकल्पिक है: केवल ज्ञान ग्राफ़ और पिछले प्रयोगों पर अर्थपरक खोज
(आपके कॉन्फ़िग में `models.embedding`) ही टेक्स्ट को एम्बेड करते हैं, और
जनरेशन इसका कभी उपयोग नहीं करता। जब आप इनमें से किसी एक को चालू करें:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia जोड़ें (अनुशंसित)।** एक कुंजी हर सत्र को
[Leeroopedia](https://leeroopedia.com) देती है, जो ML और AI फ़्रेमवर्क पर सबकों का हमारा
होस्टेड बैंक है। इंस्टॉल करने को कुछ नहीं;
[Leeroopedia डैशबोर्ड](https://app.leeroopedia.com/dashboard) पर एक नया खाता
$20 के मुफ़्त क्रेडिट के साथ आता है।

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. पैकेज इंस्टॉल करें** (Python 3.10+):

```bash
pip install leeroo-kapso
```

**4. सेटअप सत्यापित करें:**

```bash
kapso doctor
```

`doctor` बताता है कि **आपके कॉन्फ़िग** को वास्तव में क्या चाहिए, और जो कुछ भी गायब है
उसके लिए सटीक समाधान बताता है। आवश्यकताएँ कॉन्फ़िग का अनुसरण करती हैं, इसलिए एक पूरी तरह
codex वाले सेटअप से कभी `claude` नहीं माँगा जाता। केवल उसी क्रिया की आवश्यकताएँ देखने के लिए
इसे एक क्रिया तक सीमित करें:

```bash
kapso doctor build             # या learn / deploy / experience
```

ये वही जाँचें हैं जो क्रिया स्वयं कोई भी काम करने से पहले चलाती है:
`kapso.build(...)` किसी गायब CLI पर सत्र के भीतर गहराई में जाकर नहीं, बल्कि
सेकंडों में विफल हो जाता है। `[-- ]` चिह्नित आइटम वैकल्पिक हैं; वे ऐसी सुविधाओं को
सीमित करते हैं जिनकी आपको शायद ज़रूरत न हो (docker, Weaviate, Neo4j, deploy लक्ष्य)।

**या अपने कोडिंग एजेंट से यह सब करवाएँ:**
[सेटअप प्रॉम्प्ट](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
को Claude Code, Codex या OpenCode में पेस्ट करें।

**वे स्टोर जिनकी एक इंडेक्स को ज़रूरत होती है (वैकल्पिक)**: `kapso experience add --repo`
और `--research` किसी कॉर्पस को स्थानीय Weaviate + Neo4j में इनजेस्ट करते हैं। किसी
स्रोत चेकआउट से:

```bash
bash scripts/start_infra.sh   # docker के ज़रिए दोनों को शुरू करता है
```

**स्रोत से (विकास के लिए)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

पुराना aider एडाप्टर एक एक्स्ट्रा है (`pip install "leeroo-kapso[aider]"`,
Python <3.13); डिफ़ॉल्ट claude/codex एजेंटों को किसी एक्स्ट्रा की ज़रूरत नहीं।

### बुनियादी उपयोग

मुख्य क्रियाओं को ऊपर दी गई पूर्वापेक्षाओं के अलावा कुछ नहीं चाहिए:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # शुरू करने के लिए किसी ज्ञान ग्राफ़ की ज़रूरत नहीं

# Build: प्रयोग के ज़रिए एक समाधान। अभियान लॉन्च पर
# `status: <path>` प्रिंट करता है; इसे किसी दूसरे टर्मिनल से लाइव देखें, इसके साथ
#     kapso watch ./campaign
# अगर किसी सत्र को कुछ ऐसा चाहिए जो केवल आप दे सकते हैं (कोई API कुंजी, कोई फ़ाइल),
# तो अभियान रुक जाता है और `kapso inbox reply` उसे फिर से चालू करता है:
#     https://docs.leeroo.com/docs/build/inbox
# परिणाम को किसी पुल रिक्वेस्ट या इश्यू के लिए Markdown के रूप में साझा करें, इसके साथ
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # या शून्य से शुरू करने के लिए छोड़ दें
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# उस अभियान से सीखें जो आपने अभी चलाया: प्रक्षेपवक्र को खंगालें, सबकों को
# ग्रेड करें, और साक्ष्य-मूल्यांकित ज्ञान कार्ड बैंक में रखें। बैंक (एक स्थानीय
# git रिपो) पहले उपयोग पर अपने आप बन जाता है; सबक आपकी मशीन पर ही रहते हैं
# जब तक आप उन्हें साझा नहीं करते:
#     kapso experience share <git-url>
# जिसके बाद हर learn() बैंक को वहाँ पुश करता है।
lesson = kapso.learn(solution)
print(lesson.explain())

# फिर से build करें, उन कार्डों के साथ जो इसने अभी कमाए: किसी build को ठीक वही
# अनुभव परोसा जाता है जो वह लोड करता है।
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

स्टोर चलते हुए, आप बाहरी ज्ञान को भी किसी इंडेक्स में इनजेस्ट कर सकते हैं
और उसे उसी तरह परोस सकते हैं:

```python
from kapso import Kapso

kapso = Kapso()

# एक बार: कोई रिपॉजिटरी, या किसी प्रश्न के लिए शोध किया गया वेब, एक इंडेक्स में
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# हर build जो इंडेक्स लोड करता है उसमें खोज करता है; Leeroopedia अपनी कुंजी पर चलता है
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

बजट पर एक टिप्पणी: एक इनजेस्ट वह सब कुछ निकालता है जिसका सामग्री समर्थन करती है।
एक अकेला प्रश्न दर्जनों जुड़े हुए पृष्ठ और घंटों लंबा इनजेस्ट बन सकता है;
इनजेस्ट का समय निकालने योग्य सार के साथ बढ़ता है, इसलिए वे स्रोत जोड़ें
जिनमें आप खोज करेंगे, न कि हर वह स्रोत जो आपके पास है।

और किसी समाधान को चलते हुए सॉफ़्टवेयर में बदलने के लिए:

```python
software = kapso.deploy(solution, target="local")   # या docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # वह कॉल जिसे परिनियोजन ने सिद्ध किया
software.stop()
```

### मैं किसी अभियान को कैसे सीमित करूँ?

पुनरावृत्तियों, बीते समय, और USD में सर्वोत्तम-प्रयास मॉडल खर्च को सीमित करने के लिए `max_iterations`, `time_budget_minutes`, और `cost_budget` का उपयोग करें।
प्लेटफ़ॉर्म लागतों के लिए, `search_strategy.params.cost` में अपने अनुमानक, `cap_per_candidate`, और `campaign_budget` के साथ [लागत-जागरूक चयन](https://docs.leeroo.com/docs/evolve/cost-aware-selection) सक्षम करें; इसका `weight` चयन को सस्ते उम्मीदवारों को प्राथमिकता देने की सुविधा भी देता है।

### मॉडल चुनना

Kapso जिस भी मॉडल का उपयोग करता है, उसका नाम एक कॉन्फ़िग फ़ाइल में है। पैकेज्ड डिफ़ॉल्ट
build सत्रों को `claude-opus-5` पर, सीखने की टीमों को
`claude-fable-5` पर, और codex भूमिकाओं को `gpt-5.6-sol` पर चलाता है, लेकिन मॉडल तक पहुँच
सदस्यता पर निर्भर है (कोई योजना एक मॉडल को सीमित कर सकती है जबकि दूसरे को परोसती है)।
अलग मॉडलों पर चलाने के लिए, पैकेज्ड कॉन्फ़िग को कॉपी करें, संपादित करें, और Kapso को
अपने कॉन्फ़िग की ओर इंगित करें:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# उदाहरण के लिए सीखने की टीमों को fable के बजाय opus पर चलाएँ:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

केवल-टेक्स्ट इन्फ़रेंस भूमिकाएँ `openai_compatible` Chat Completions को चुन सकती हैं।
अपने कॉन्फ़िग में मॉडल और एंडपॉइंट सेट करें; क्रेडेंशियल परिवेश में रहते हैं।
कॉन्फ़िग को `Kapso(config=...)` या `kapso build --config ...` से लोड करें।
कोडिंग और वेब शोध के लिए अब भी संबंधित टूल वाले एजेंट चाहिए।
देखें [कॉन्फ़िगरेशन उदाहरण और सीमाएँ](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)।

किसी लंबे रन से पहले, अपने कॉन्फ़िग में नामित हर मॉडल को अपनी
वास्तविक सदस्यताओं के विरुद्ध जाँचें: जिस मॉडल को आपका लॉगिन नहीं परोस सकता, वह रन में
घंटों बाद के बजाय यहाँ सेकंडों में विफल हो जाता है। जिस मॉडल को आप परोस सकते हैं,
उस पर लगा उपयोग कैप एक-टोकन जाँच को दिखाई नहीं देता:

```bash
kapso doctor --models                            # पैकेज्ड कॉन्फ़िग
kapso doctor --models --config kapso-config.yaml # आपका
kapso doctor learn --models                      # केवल सीखने की टीमें
```

उस जाँच को ऐसी चीज़ बनाने के बजाय जिसे चलाना आपको याद रखना पड़े, उसे हर कॉल का हिस्सा
बनाने के लिए, इसे अपने कॉन्फ़िग में चालू करें। इसकी लागत हर अलग
`{cli, model}` जोड़ी के लिए एक फेंकने योग्य टोकन है, जो किसी अनदेखे कई घंटों के
`learn()` से पहले सार्थक है:

```yaml
preflight:
  enabled: true            # हर क्रिया से पहले स्थैतिक जाँचें (डिफ़ॉल्ट)
  live_model_probe: true   # + एक-टोकन मॉडल जाँचें (डिफ़ॉल्ट: false)
```

मॉडल बदलने से *गति* भी बदलती है: कॉन्फ़िग में टीमों के `timeout_minutes` कैप
डिफ़ॉल्ट मॉडलों पर कैलिब्रेट किए गए थे, और एक बदला हुआ मॉडल जो
अधिक देर तक तर्क करता है, उन्हें बढ़ाने की माँग कर सकता है।

विस्तृत एकीकरण चरणों के लिए, [त्वरित शुरुआत](https://docs.leeroo.com/docs/first-build) और [इंस्टॉलेशन](https://docs.leeroo.com/docs/installation) गाइड देखें।

## समर्थित बेंचमार्क

| बेंचमार्क | विवरण |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI का ML-इंजीनियरिंग बेंचमार्क: टेबुलर, विज़न, टेक्स्ट और ऑडियो में पूरी प्रतियोगिताएँ, कच्चे डेटा से ग्रेड की गई सबमिशन तक |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI का एल्गोरिदमिक-अनुकूलन बेंचमार्क: घंटों लंबी खोजों में प्रतियोगिता ह्यूरिस्टिक्स को डिज़ाइन करना, लागू करना और दोहराना |
| [**RelBench**](../../benchmarks/relbench/README.md) | संबंधपरक डेटा पर भविष्यसूचक ML के लिए Stanford का बेंचमार्क: Amazon, H&M, Stack Exchange, क्लिनिकल ट्रायल और अन्य के बहु-तालिका डेटाबेस से सीधे वर्गीकरण, प्रतिगमन और सिफ़ारिश |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | विज़न, भाषा और अनुकूलन में समयबद्ध ओलंपियाड ML: विशेषज्ञ-निर्धारित कार्य, प्रतियोगिता हार्डवेयर, शून्य मानवीय मदद |

हर बेंचमार्क का एक दस्तावेज़ीकरण पृष्ठ भी है जो बताता है कि उसे कैसे चलाएँ, उसके CLI विकल्प क्या हैं और आउटपुट में क्या होता है:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## उदाहरण

हर उदाहरण एक कार्य है: एक ब्रीफ़, एक शुरुआती रिपॉजिटरी, डेटा या उसे पाने का तरीका, और अधिकतम बारह पंक्तियों का एक ड्राइवर; [examples/README.md](../../examples/README.md) में मिनट और एक कमांड दिए गए हैं।

| उदाहरण | कार्य |
|---------|----------|
| [**डेटा इंजीनियरिंग (Snowflake पर dbt मॉडल)**](../../examples/dbt-snowflake/README.md) | Snowflake पर एक खुदरा वेयरहाउस में, डेटा गुणवत्ता की समस्याओं वाले dbt मॉडल ठीक करें और वे नए मॉडल बनाएँ जिनकी माँग सत्रह टिकटों का बैकलॉग करता है, जब तक हर टिकट अपने प्रोडक्शन परीक्षण पास न कर ले; फिर हर टिकट के लिए एक pull request दें और उन्हें प्रोडक्शन में रिलीज़ करें। |
| [**Harness अनुकूलन (ग्राहक सहायता एजेंट)**](../../examples/support-agent-harness/README.md) | एक ग्राहक सहायता एजेंट के harness को अनुकूलित करें ताकि ग्राहक वार्तालापों के समाधान में कमी लाए बिना अंतर्निहित LLM को Fable 5.1 से एक स्थानीय Qwen 27B पर बदला जा सके, फिर उसे Docker में परिनियोजित करें और उस रन से सीखें। |
| [**भविष्यसूचक मॉडलिंग (ग्राहक चर्न)**](../../examples/churn-first-build/README.md) | अनुमान लगाएँ कि कौन से ग्राहक छोड़कर जाएँगे: एक ऐसे मूल्यांकनकर्ता के तहत जिसे अभियान संपादित नहीं कर सकता, चर्न मॉडल के आउट-ऑफ़-सैंपल ROC AUC को लक्ष्य की ओर बढ़ाएँ। |
| [**धोखाधड़ी पहचान**](../../examples/fraud-no-evaluator/README.md) | लेनदेन को धोखाधड़ी के लिए स्कोर करें जब रिपॉजिटरी में कोई मूल्यांकनकर्ता न हो: ब्रीफ़ प्रोटोकॉल बताता है, अभियान पहले मूल्यांकनकर्ता बनाता है, फिर स्कोरर को बेहतर करता है। |
| [**क्वांट फ़ाइनेंस**](../../examples/trading/README.md) | सिंथेटिक उपकरणों पर एक दैनिक ट्रेडिंग रणनीति: ट्रेडिंग harness पर आउट-ऑफ़-सैंपल Sharpe बढ़ाएँ, हर प्रयास पर उसकी जाँचों के साथ। एक शोध उदाहरण, वित्तीय सलाह नहीं। |
| [**टेबुलर ML**](../../examples/end-to-end/README.md) | एक Kaggle प्रतियोगिता शुरू से अंत तक: Spaceship Titanic एक डमी क्लासिफ़ायर से एक वास्तविक मॉडल तक, फिर समीक्षा, सीखना, सबक के साथ फिर से build, और परिनियोजन। |
| [**PyTorch अनुकूलन**](../../examples/torch-performance/README.md) | एक PyTorch मॉड्यूल को CUDA पर एक संख्यात्मक सहनशीलता के भीतर तेज़ बनाएँ: फ़्यूज़न, संकलन, कम मेमोरी ट्रैफ़िक। |
| [**GPU कर्नेल अनुकूलन**](../../examples/gpu-kernels/README.md) | फ़्लैश अटेंशन या एक Triton कर्नेल के साथ, एक संख्यात्मक सहनशीलता के भीतर, CUDA पर मल्टी-हेड सेल्फ़-अटेंशन को तेज़ करें। |

## 📚 दस्तावेज़ीकरण और सहायता

- **पूरा दस्तावेज़ीकरण**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [इंस्टॉलेशन](https://docs.leeroo.com/docs/installation): कोडिंग-एजेंट CLI, पैकेज, और `kapso doctor`
  - [त्वरित शुरुआत](https://docs.leeroo.com/docs/first-build): आपका पहला अभियान
  - [CLI संदर्भ](https://docs.leeroo.com/docs/reference/cli): हर कमांड, फ़्लैग और डिफ़ॉल्ट
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso` पर हर सार्वजनिक मेथड
  - [कॉन्फ़िगरेशन](https://docs.leeroo.com/docs/reference/configuration): `config.yaml` की हर कुंजी
  - [learn क्या उत्पन्न करता है](https://docs.leeroo.com/docs/learn/what-learn-produces): सबकों का बैंक, हर सबक के साथ एक विश्वसनीयता स्कोर, एक सीमा और उसका साक्ष्य, और [विकास मोड](https://docs.leeroo.com/docs/learn/development-mode)
  - [स्मार्ट प्रयोग](https://docs.leeroo.com/docs/evolve/overview): प्रयोगों का वृक्ष और उसके मॉड्यूल, और [एक प्रयोग के भीतर क्या होता है](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [अनुभव](https://docs.leeroo.com/docs/experience): आपका बैंक, Leeroopedia, रिपॉजिटरी और पेपर जो खोजने योग्य ज्ञान में बदले गए
  - [परिनियोजन](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal और अन्य लक्ष्य
  - [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code, Codex और Cursor के लिए कमांड, स्किल और MCP सर्वर
- **अपने कोडिंग एजेंट से**: दस्तावेज़ `https://docs.leeroo.com/mcp` पर एक MCP सर्वर चलाते हैं, और हर पृष्ठ Markdown के रूप में परोसा जाता है; देखें [llms.txt](https://docs.leeroo.com/llms.txt) और [आपके कोडिंग एजेंट में Kapso](https://docs.leeroo.com/docs/coding-agent-skills)।
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **अपने कोडिंग एजेंट को Kapso चलाने दें**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) के अंतर्गत प्लगइन में दस कमांड, छह स्किल और MCP सर्वर हैं, हर होस्ट के लिए एक इंस्टॉल पंक्ति (ऊपर चरण 0); OpenCode स्किलों को एक फ़ोल्डर के रूप में लोड करता है (`ln -s <clone>/plugin/skills .opencode/skills`)। देखें [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills)।
- **अपने कोडिंग एजेंट को Kapso के ज्ञान टूल दें**: `kapso-mcp`, जो पैकेज के साथ इंस्टॉल होता है, उस सत्र को जिसमें आप काम करते हैं, वेब शोध, रिपॉजिटरी की मेमोरी, अभियान का प्रयोग इतिहास और सबकों का बैंक परोसता है; देखें [MCP गेट](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)।
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; फिर ~/.codex/config.toml में [mcp_servers.kapso] के अंतर्गत `startup_timeout_sec = 60` (Codex के दस सेकंड व्यस्त मशीन पर कम पड़ते हैं)
  ```
- **समुदाय**: [Discord](https://discord.gg/hqVbPNNEZM)
- **वेबसाइट**: [leeroo.com](https://leeroo.com)


## एंटरप्राइज़ के लिए Kapso

एंटरप्राइज़ में, Kapso एजेंट शुरुआती डेटा और AI इंजीनियरिंग harness तथा सबकों के एक बैंक के साथ आते हैं, और व्यावसायिक उद्देश्यों को अपने दम पर काम करने वाले AI और डेटा सिस्टमों में बदलते हैं, कंपनी के आर्टिफ़ैक्ट, रिपो, पाइपलाइन, डेटा स्रोतों और बहुत कुछ की जाँच करते हुए, ताकि चलते-चलते एंटरप्राइज़ डेटा एस्टेट को बनाया जा सके। अपने एंटरप्राइज़ कार्यों के लिए Kapso को अपनाने हेतु, [हमसे बात करें](https://leeroo.com/contact-us)।

## योगदान

हम योगदान का स्वागत करते हैं! शुरुआत कैसे करें, इसके विवरण के लिए कृपया हमारी [योगदान गाइड](../../CONTRIBUTING.md) देखें।

## योगदानकर्ता

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

[contrib.rocks](https://contrib.rocks) के साथ बनाया गया।
