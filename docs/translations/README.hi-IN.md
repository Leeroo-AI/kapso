<!-- Hindi translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">एक इंजन जो प्रयोग चलाता है, सर्वश्रेष्ठ समाधान परिनियोजित करता है, और हर रन से सीखता है</h4>
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

> **अगर आप इस रेपो को देख रहे एक AI एजेंट हैं:** Kapso प्लगइन, जिसमें सेटअप करने, उद्देश्य बताने,
> harness लिखने और रन की समीक्षा करने के कौशल शामिल हैं, एक पंक्ति में इंस्टॉल होता है:
> `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`)।

---

## समाचार

- **IOAI 2026 · AI मॉडल ट्रैक।** [अंतर्राष्ट्रीय AI ओलंपियाड](https://ioai-official.org) AI युग का IMO है: 108 देशों और क्षेत्रों से 471 प्रतियोगी, एकल-GPU बजट के तहत विशेषज्ञों द्वारा डिज़ाइन किए गए छह कार्य। 2026 में इसने [IOAI²](https://ioai-official.org/ai-model-track/) शुरू किया, जहाँ AI सिस्टम दो बिना निगरानी वाले 6-घंटे के सत्रों में *वही* परीक्षा देते हैं। घड़ी शुरू होने के बाद कोई भी मनुष्य कुछ भी हल, सही या बेहतर नहीं कर सकता। Kapso ने 14 संस्थापक AI प्रतिभागियों में से एक के रूप में प्रवेश किया।
  - 🎓 **सभी 471 मनुष्यों से अधिक स्कोर किया**: कुल 536.07, हॉल के हर प्रतियोगी से ऊपर।
  - 🏆 **IOAI² ग्रैंड मास्टर ट्रॉफी**: भाग लेने वाले सभी AI सिस्टमों में शीर्ष 3। [पूरे परिणाम →](../../benchmarks/ioai2026/README.md) · [ट्रैक पर IOAI की समीक्षा →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **तीनों [RelBench](../../benchmarks/relbench/README.md) लीडरबोर्ड पर प्रथम**: रिलेशनल डेटा पर प्रेडिक्टिव ML के लिए Stanford के बेंचमार्क पर, Kapso आधिकारिक लीडरबोर्ड में वर्गीकरण (12 कार्यों पर 81.2 AUROC, [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) के 79.3 और Stanford STAR के [RT-J](https://star-project.stanford.edu/rt-j) के 79.0 के मुकाबले), रिग्रेशन (9 कार्यों पर 0.2476 NMAE, RT-J के 0.2711 और KumoRFM-v2 के 0.2847 के मुकाबले, 8.7% कम त्रुटि) और अनुशंसाओं (10 कार्यों पर 18.4 MAP, सर्वश्रेष्ठ अन्य प्रविष्टि ID-GNN के 9.3 के मुकाबले) में आगे है। हर संख्या [आधिकारिक RelBench लीडरबोर्ड](https://star-project.stanford.edu/relbench/leaderboard/) पर एक प्रविष्टि है। harness में रखा गया सीखना मॉडल वेट्स में बेक किए गए सीखने से आगे रहा, और सबकों के बैंक को अपडेट करने की लागत एक टेबुलर फाउंडेशन मॉडल को फिर से प्रशिक्षित करने से कहीं कम है।

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="574">

- **[Leeroopedia](https://leeroopedia.com)**: डेटा और ML डोमेन में हमारे सबकों की एक विकि, पेपरों और रिपॉज़िटरियों पर प्रयोग करके निकाले गए लगभग 100k पृष्ठ, जो महीने में लगभग 6M अनुरोध पूरे करती है, जिनमें से लगभग 28% AI एजेंटों से आते हैं। Kapso इससे MCP के ज़रिए जुड़ता है: इसके सत्र विचार-मंथन और कार्यान्वयन के दौरान ज्ञान खोजने, योजनाएँ बनाने और विफलताओं का निदान करने के लिए इसका उपयोग करते हैं।
- **[MLE-Bench](../../benchmarks/mle/README.md) पर #1**: KAPSO ने Kaggle ML प्रतियोगिताओं (MLE Benchmark) पर ओपन-सोर्स सिस्टमों में शीर्ष रैंकिंग हासिल की।

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574">

- **[ALE-Bench](../../benchmarks/ale/README.md) पर #1**: KAPSO ने लंबी-अवधि की एल्गोरिदमिक खोज समस्याओं (ALE Benchmark) पर शीर्ष रैंकिंग हासिल की।

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574">

## Kapso क्या है?

किसी कंपनी के भीतर AI या डेटा परिणाम देने का मतलब है प्रयोग के ज़रिए दो चीज़ें सीखना। कोई भी व्यक्ति या AI एजेंट नहीं जानता कि कोई डेटा एस्टेट कैसे जुड़ा और संचालित है; इसे सीखने का एकमात्र तरीका है कंपनी के आर्टिफ़ैक्ट्स (रेपो, पाइपलाइन, दस्तावेज़ीकरण, डेटा स्रोत) की जाँच करना, परिकल्पनाएँ प्रस्तावित करना, प्रयोग चलाना और उनसे सीखना, जब तक हर सिस्टम का व्यवहार और उसकी शर्तें समझ में न आ जाएँ। और जब वह समझ बन जाती है, तब भी सर्वश्रेष्ठ समाधान पहले से नहीं चुना जा सकता, चाहे काम किसी वेयरहाउस को ऑप्टिमाइज़ करना हो या चर्न की भविष्यवाणी करना, इसलिए आप तब तक प्रयोग करते और सीखते हैं जब तक कोई समाधान परिवेश के अनुकूल न बैठे। परिवेश के अनुकूल बैठने वाली जानकारी संचित होती जाती है, काफ़ी हद तक RL की तरह। RAG इसका उल्टा करता है: यह एजेंट को ऐसा सामान्य ज्ञान देता है जिसे इस परिवेश में कभी परखा नहीं गया, जिसका हर हिस्सा एक समान वज़न रखता है, और उस ज्ञान का तब तक बहुत कम मूल्य होता है जब तक उसे वहाँ आज़माया न गया हो।

Kapso वह इंजन है जो हमने इसके लिए बनाया है। यह एक डोमेन harness और सबकों के एक खाली बैंक से शुरू होता है; एक उद्देश्य और सफलता मानदंड दिए जाने पर, यह प्रयोगों का एक वृक्ष चलाता है, परिकल्पनाएँ प्रस्तावित करता है, फिर हर एक को कार्यान्वित और सत्यापित करता है, और सर्वश्रेष्ठ समाधान को लक्ष्य स्टैक पर परिनियोजित करता है। फिर यह रन से सीखता है, हर परिकल्पना और कार्यान्वयन को परिणाम पर उसके प्रभाव के अनुसार श्रेय देता है और बैंक को अपडेट करता है, जहाँ हर सबक एक विश्वसनीयता स्कोर और यह सीमा रखता है कि वह कहाँ लागू होता है। समय के साथ डोमेन harness में सबकों का एक ऐसा बैंक बन जाता है जिसमें उस परिवेश में प्रयोग से सिद्ध सर्वोत्तम प्रथाएँ संचित होती हैं। यह सिस्टम मॉडल-अज्ञेय और कोडिंग-एजेंट-अज्ञेय है: ओपन सोर्स, [Leeroo](https://leeroo.com) द्वारा, PyPI पर [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) के रूप में प्रकाशित।

### चार हिस्से

| हिस्सा | क्रिया | यह क्या करता है |
|------|------|--------------|
| [**स्मार्ट प्रयोग**](https://docs.leeroo.com/docs/build/brief) | `.build()` | विशेषीकृत मॉड्यूल एक ही डेटा संरचना साझा करते हैं, प्रयोगों का एक वृक्ष: हर नोड परिकल्पना, कार्यान्वयन (git ब्रांच और तकनीकी कठिनाइयाँ), उपयोग किया गया बजट और उद्देश्य से दूरी दर्ज करता है। हर प्रयोग के लिए, एक विचार-मंथन मॉड्यूल पहले के प्रयोगों और सबकों के बैंक से विचार प्रस्तावित करता है, उन्हें प्रत्येक द्वारा उपयोग किए जाने वाले बजट के मुकाबले अपेक्षित प्रतिफल के अनुसार फिर से रैंक करता है, और शीर्ष-रैंक वाले विचारों को कार्यान्वयनकर्ता को भेजता है, जो पैरेंट नोड से फ़ोर्क की गई ब्रांच पर निर्माण करता है। चूँकि बैंक लगातार सीखता रहता है कि उस सटीक परिवेश में क्या काम करता है, विचार उसके लिए विशिष्ट बन जाते हैं, न कि वे सामान्य विचार जो एक साधारण LLM या RAG देता। परिणामों को एक फ़िंगरप्रिंट किए गए मूल्यांकनकर्ता द्वारा सत्यापित किया जाता है जिस तक कोई प्रयोग नहीं पहुँच सकता, और मूल्यांकन भी विकसित होता है: प्रयोग मॉड्यूल मेट्रिक्स और मानदंडों में अपडेट प्रस्तावित कर सकते हैं जिन्हें सत्यापनकर्ता मंज़ूर करता है। [यह कैसे काम करता है →](https://docs.leeroo.com/docs/evolve/overview) |
| [**सीखना**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | बैंक में सब कुछ प्रयोगों से आता है: हर सबक प्रयोग रनों के ट्रेस से आगमनात्मक या निगमनात्मक रूप से निकाला जाता है, एक विश्वसनीयता स्कोर, यह सीमा शर्तें कि वह कहाँ लागू होता है और उसके पीछे का साक्ष्य रखता है, और हर रन के बाद `learn()` अंतिम परिणाम पर सबक के प्रभाव से इन्हें अपडेट करता है। बैंक में बदलाव भी प्रयोग हैं: हर अपडेट किए गए सबक का A/B परीक्षण उसी build को उसके साथ और उसके बिना चलाकर किया जाता है, ताकि सुधार मापा जाए, अनुमान न लगाया जाए। बैंक git पर बना है, इसलिए आप अपना अनुभव दूसरों के साथ साझा कर सकते हैं (`kapso experience share`), अपने लिए उपलब्ध बैंकों की सूची देख सकते हैं (`kapso experience list`) और उन्हें अपडेट कर सकते हैं (`kapso experience update`)। [अनुभव →](https://docs.leeroo.com/docs/experience) बताता है कि किसी build को क्या परोसा जाता है: आपका बैंक, आपकी टीम द्वारा साझा किया गया बैंक, Leeroopedia, या किसी रिपॉज़िटरी या शोध किए गए प्रश्न से इन्जेस्ट किया गया कॉर्पस। |
| [**परिनियोजन**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | सर्वश्रेष्ठ समाधान को लक्ष्य स्टैक के अनुकूल बनाने के लिए उसी तरह का एक छोटा प्रयोग चलाता है: एक सत्र नोड की एक प्रति को अनुकूलित करता है, उसे परिनियोजित करता है, और हेल्थ चेक तथा एक कार्यशील कॉल प्रदर्शित करता है। Docker, Modal, BentoCloud और LangGraph Platform साथ आते हैं, और आप harness में अपना लक्ष्य स्टैक जोड़ सकते हैं। |
| [**डोमेन अनुकूलन**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harness Kapso के डोमेन पैकेज हैं: प्रेडिक्टिव मॉडलिंग जैसे किसी क्षेत्र के लिए ज्ञान और वर्कफ़्लो, ताकि कोई build शून्य से प्रयोग और सीखना शुरू न करे; प्रेडिक्टिव मॉडलिंग और प्रतिस्पर्धी प्रोग्रामिंग Kapso के साथ आते हैं। एक रन डोमेन harness और सबकों के खाली बैंक से शुरू होता है, और समय के साथ harness में वह बैंक बन जाता है जिसमें उस परिवेश में जो काम आया वह संचित होता है। अपना प्रारंभिक डोमेन harness, अपने निर्देश, कोड, गार्डरेल और जाँचें, प्लगइन में `/kapso:harness` या CLI में `kapso harness` से लिखें। |

## 🚀 क्विकस्टार्ट

### इंस्टॉलेशन

**0. किसी कोडिंग एजेंट में हैं?** Kapso प्लगइन इंस्टॉल करें और इसे नीचे दिए गए
चरणों में आपके साथ चलने दें:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

फिर `/kapso:setup` इंस्टॉल और सत्यापित करता है, और `/kapso:build <what you want>`
पहला अभियान लॉन्च करता है (Codex पर: `$kapso-setup`, `$kapso-build <what you want>`)। देखें [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills)।

**1. पूर्वापेक्षाएँ।** Kapso अपना इन्फ़रेंस कोडिंग-एजेंट CLI के ज़रिए चलाता है
(कोई डायरेक्ट-API फ़ॉलबैक नहीं है), इसलिए कुछ भी काम करने से पहले आपको Node.js और
दोनों एजेंट CLI में लॉग इन होना ज़रूरी है:

```bash
# Node.js 18+ (https://nodejs.org), फिर:
npm install -g @openai/codex            # शोध, निर्णय, यूटिलिटी
codex login

npm install -g @anthropic-ai/claude-code  # विचार-मंथन + कार्यान्वयन (डिफ़ॉल्ट होस्ट)
claude auth login
```

OpenAI कुंजी वैकल्पिक है: केवल नॉलेज ग्राफ़ और पिछले प्रयोगों पर सिमेंटिक खोज
(आपके कॉन्फ़िग में `models.embedding`) टेक्स्ट को एम्बेड करते हैं, और
जनरेशन इसका कभी उपयोग नहीं करता। जब आप इनमें से किसी को चालू करें:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia कनेक्ट करें (अनुशंसित)।** एक कुंजी हर अभियान सत्र को
[Leeroopedia](https://leeroopedia.com) देती है, ML और AI फ़्रेमवर्क का एक होस्टेड
नॉलेज बेस: कोई फ़्रेमवर्क वास्तव में कैसे व्यवहार करता है, योजना समीक्षाएँ,
दस्तावेज़ीकृत हाइपरपैरामीटर रेंज, विफलता निदान। इंस्टॉल करने को कुछ नहीं;
[Leeroopedia डैशबोर्ड](https://app.leeroopedia.com/dashboard) पर एक नया खाता
$20 के मुफ़्त क्रेडिट के साथ आता है।

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. पैकेज इंस्टॉल करें** (Python 3.10+):

```bash
pip install leeroo-kapso
```

पैकेज `leeroo-kapso` है। `kapso` नाम का PyPI पैकेज एक असंबंधित
WhatsApp टूल है जो अपना एक `kapso` कमांड इंस्टॉल करता है और इस वाले को ढक देता है;
अगर वह मौजूद है, तो पहले `pip uninstall kapso` करें।

**4. सेटअप सत्यापित करें:**

```bash
kapso doctor
```

`doctor` बताता है कि **आपके कॉन्फ़िग** को वास्तव में क्या चाहिए, और जो कुछ भी
गायब है उसके लिए सटीक सुधार का नाम देता है। आवश्यकताएँ कॉन्फ़िग का अनुसरण करती हैं,
इसलिए पूरी तरह codex वाले सेटअप से कभी `claude` नहीं माँगा जाता। केवल उस क्रिया की
आवश्यकताएँ देखने के लिए इसे एक क्रिया तक सीमित करें:

```bash
kapso doctor build             # या learn / deploy / experience
```

ये वही जाँचें हैं जो क्रिया स्वयं कोई काम करने से पहले चलाती है:
`kapso.build(...)` किसी गायब CLI पर सत्र के भीतर गहराई में जाने के बजाय
सेकंडों में विफल हो जाता है। `[-- ]` चिह्नित आइटम वैकल्पिक हैं; वे ऐसी सुविधाओं
को सीमित करते हैं जिनकी आपको शायद ज़रूरत न हो (docker, Weaviate, Neo4j, deploy लक्ष्य)।

**या यह सब अपने कोडिंग एजेंट से करवाएँ:**
[सेटअप प्रॉम्प्ट](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
को Claude Code, Codex या OpenCode में पेस्ट करें।

**एक इंडेक्स को जिन स्टोर की ज़रूरत होती है (वैकल्पिक)**: `kapso experience add --repo`
और `--research` एक कॉर्पस को स्थानीय Weaviate + Neo4j में इन्जेस्ट करते हैं। सोर्स
चेकआउट से:

```bash
bash scripts/start_infra.sh   # docker के ज़रिए दोनों शुरू करता है
```

**सोर्स से (विकास के लिए)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

लेगेसी aider एडॉप्टर एक एक्स्ट्रा है (`pip install "leeroo-kapso[aider]"`,
Python <3.13); डिफ़ॉल्ट claude/codex एजेंटों को किसी एक्स्ट्रा की ज़रूरत नहीं है।

### बुनियादी उपयोग

मुख्य क्रियाओं को ऊपर दी गई पूर्वापेक्षाओं के अलावा कुछ नहीं चाहिए:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # शुरू करने के लिए किसी नॉलेज ग्राफ़ की ज़रूरत नहीं

# Build: प्रयोग के ज़रिए एक समाधान। अभियान लॉन्च पर
# `status: <path>` प्रिंट करता है; इसे दूसरे टर्मिनल से लाइव देखें:
#     kapso watch ./campaign
# अगर किसी सत्र को कुछ ऐसा चाहिए जो केवल आप दे सकते हैं (एक API कुंजी, एक फ़ाइल),
# तो अभियान रुक जाता है और `kapso inbox reply` उसे फिर से शुरू करता है:
#     https://docs.leeroo.com/docs/build/inbox
# परिणाम को पुल रिक्वेस्ट या इश्यू के लिए Markdown के रूप में साझा करें:
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # या शून्य से शुरू करने के लिए छोड़ दें
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# अभी चलाए गए अभियान से सीखें: प्रक्षेपपथ का खनन करें, सबकों को
# ग्रेड करें, और साक्ष्य के अनुसार मूल्यांकित ज्ञान कार्ड बैंक में रखें। बैंक (एक स्थानीय
# git रेपो) पहले उपयोग पर अपने आप बन जाता है; सबक आपकी मशीन पर
# तब तक रहते हैं जब तक आप उन्हें साझा न करें:
#     kapso experience share <git-url>
# जिसके बाद हर learn() बैंक को वहाँ पुश करता है।
lesson = kapso.learn(solution)
print(lesson.explain())

# फिर से build करें, उन कार्डों के साथ जो इसने अभी कमाए: एक build को ठीक वही
# अनुभव परोसा जाता है जो वह लोड करता है।
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

स्टोर चलते हुए, आप बाहरी ज्ञान को भी एक इंडेक्स में इन्जेस्ट कर सकते हैं और
उसे उसी तरह परोस सकते हैं:

```python
from kapso import Kapso

kapso = Kapso()

# एक बार: एक रिपॉज़िटरी, या किसी प्रश्न के लिए शोध किया गया वेब, एक इंडेक्स में
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# इंडेक्स लोड करने वाला हर build उसे खोजता है; Leeroopedia अपनी कुंजी पर चलता है
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

बजट पर एक नोट: एक इन्जेस्ट वह सब कुछ निकालता है जिसका सामग्री समर्थन करती है।
एक अकेला प्रश्न दर्जनों लिंक किए गए पृष्ठ और घंटों लंबा इन्जेस्ट बन सकता है;
इन्जेस्ट का समय निकाले जा सकने वाले सार के साथ बढ़ता है, इसलिए वे स्रोत जोड़ें
जिन्हें आप खोजेंगे, न कि आपके पास मौजूद हर स्रोत।

और किसी समाधान को चलते हुए सॉफ़्टवेयर में बदलने के लिए:

```python
software = kapso.deploy(solution, target="local")   # या docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # वह कॉल जिसे परिनियोजन ने सिद्ध किया
software.stop()
```

### मैं किसी अभियान को कैसे सीमित करूँ?

पुनरावृत्तियों, बीते समय और USD में सर्वोत्तम-प्रयास मॉडल खर्च को सीमित करने के लिए `max_iterations`, `time_budget_minutes` और `cost_budget` का उपयोग करें।
प्लेटफ़ॉर्म लागतों के लिए, `search_strategy.params.cost` में अपने एस्टिमेटर, `cap_per_candidate` और `campaign_budget` के साथ [लागत-सजग चयन](https://docs.leeroo.com/docs/evolve/cost-aware-selection) सक्षम करें; इसका `weight` चयन को सस्ते उम्मीदवारों को प्राथमिकता देने की सुविधा भी देता है।

### मॉडल चुनना

Kapso जिस भी मॉडल का उपयोग करता है, उसका नाम एक ही कॉन्फ़िग फ़ाइल में है। पैकेज्ड डिफ़ॉल्ट
build सत्रों को `claude-opus-5` पर, सीखने वाली क्रू को
`claude-fable-5` पर, और codex भूमिकाओं को `gpt-5.6-sol` पर चलाता है, लेकिन मॉडल तक पहुँच
सब्सक्रिप्शन पर निर्भर है (कोई प्लान एक मॉडल को सीमित कर सकता है जबकि दूसरे को परोसता है)।
अलग मॉडलों पर चलाने के लिए, पैकेज्ड कॉन्फ़िग की प्रति बनाएँ, संपादित करें, और
Kapso को अपने कॉन्फ़िग की ओर इंगित करें:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# उदाहरण के लिए, सीखने वाली क्रू को fable के बजाय opus पर चलाएँ:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

केवल-टेक्स्ट इन्फ़रेंस भूमिकाएँ `openai_compatible` Chat Completions को चुन सकती हैं।
अपने कॉन्फ़िग में मॉडल और एंडपॉइंट सेट करें; क्रेडेंशियल परिवेश में रहते हैं।
कॉन्फ़िग को `Kapso(config=...)` या `kapso build --config ...` से लोड करें।
कोडिंग और वेब शोध के लिए अभी भी संबंधित टूल वाले एजेंटों की ज़रूरत होती है।
देखें [कॉन्फ़िगरेशन उदाहरण और सीमाएँ](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)।

किसी लंबे रन से पहले, आपके कॉन्फ़िग में नामित हर मॉडल को अपनी वास्तविक
सब्सक्रिप्शन के विरुद्ध जाँचें: जिस मॉडल को आपका लॉगिन परोस नहीं सकता, वह रन में
घंटों बाद के बजाय यहाँ सेकंडों में विफल हो जाता है। जिस मॉडल को आप परोस सकते हैं
उस पर उपयोग की सीमा एक-टोकन जाँच को दिखाई नहीं देती:

```bash
kapso doctor --models                            # पैकेज्ड कॉन्फ़िग
kapso doctor --models --config kapso-config.yaml # आपका
kapso doctor learn --models                      # केवल सीखने वाली क्रू
```

उस जाँच को कुछ ऐसा बनाने के बजाय जिसे चलाना आपको याद रखना पड़े, हर कॉल का हिस्सा
बनाने के लिए इसे अपने कॉन्फ़िग में चालू करें। इसकी लागत हर अलग
`{cli, model}` जोड़ी के लिए एक फेंकने योग्य टोकन है, जो बिना निगरानी वाले कई घंटों के
`learn()` से पहले सार्थक है:

```yaml
preflight:
  enabled: true            # हर क्रिया से पहले स्थैतिक जाँचें (डिफ़ॉल्ट)
  live_model_probe: true   # + एक-टोकन मॉडल जाँचें (डिफ़ॉल्ट: false)
```

मॉडल बदलने से *गति* भी बदलती है: कॉन्फ़िग में क्रू की `timeout_minutes` सीमाएँ
डिफ़ॉल्ट मॉडलों पर कैलिब्रेट की गई थीं, और अधिक देर तक तर्क करने वाले बदले हुए मॉडल
के लिए उन्हें बढ़ाने की ज़रूरत पड़ सकती है।

विस्तृत एकीकरण चरणों के लिए, [क्विकस्टार्ट](https://docs.leeroo.com/docs/first-build) और [इंस्टॉलेशन](https://docs.leeroo.com/docs/installation) गाइड देखें।

## समर्थित बेंचमार्क

| बेंचमार्क | विवरण |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI का ML-इंजीनियरिंग बेंचमार्क: टेबुलर, विज़न, टेक्स्ट और ऑडियो में पूरी प्रतियोगिताएँ, कच्चे डेटा से ग्रेडेड सबमिशन तक |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI का एल्गोरिदमिक-ऑप्टिमाइज़ेशन बेंचमार्क: घंटों लंबी खोजों में प्रतियोगिता ह्यूरिस्टिक्स को डिज़ाइन, कार्यान्वित और पुनरावृत्त करना |
| [**RelBench**](../../benchmarks/relbench/README.md) | रिलेशनल डेटा पर प्रेडिक्टिव ML के लिए Stanford का बेंचमार्क: Amazon, H&M, Stack Exchange, क्लिनिकल ट्रायल और अन्य के बहु-तालिका डेटाबेस से सीधे वर्गीकरण, रिग्रेशन और अनुशंसा |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | विज़न, भाषा और ऑप्टिमाइज़ेशन में समयबद्ध ओलंपियाड ML: विशेषज्ञों द्वारा निर्धारित कार्य, प्रतियोगिता हार्डवेयर, शून्य मानवीय सहायता |

हर बेंचमार्क का एक दस्तावेज़ीकरण पृष्ठ भी है जो बताता है कि इसे कैसे चलाएँ, इसके CLI विकल्प क्या हैं और आउटपुट में क्या होता है:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## उदाहरण

हर उदाहरण एक कार्य है: एक ब्रीफ़, एक प्रारंभिक रिपॉज़िटरी, डेटा या उसे पाने का तरीका, और अधिकतम बारह पंक्तियों का एक ड्राइवर; [examples/README.md](../../examples/README.md) में मिनट, एक कमांड और यह कि परिणाम दर्ज है या नहीं, दिया गया है।

| उदाहरण | यह क्या दिखाता है |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | आपका पहला build: एक मौजूद मूल्यांकनकर्ता, एक बेसलाइन, एक लक्ष्य, रिकॉर्ड |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | कोई मूल्यांकनकर्ता नहीं: अभियान का पहला नोड ब्रीफ़ के प्रोटोकॉल से एक बनाता है और हर उम्मीदवार को उसके विरुद्ध मापा जाता है |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | DuckDB पर एक dbt पाइपलाइन जिसके एक्सपोर्ट में दोष हैं: डेटा कॉन्ट्रैक्ट को टिकाऊ बनाया गया |
| [**trading**](../../examples/trading/README.md) | build पर एक harness: एक सेटअप सत्र, हर सत्र में पृष्ठ और हुक, हर प्रयास पर सत्यापन |
| [**end to end**](../../examples/end-to-end/README.md) | Build, समीक्षा, learn, सबक के साथ फिर से build, deploy: एक Kaggle टेबुलर मॉडल शुरू से अंत तक |
| [**prompt harness**](../../examples/prompt-harness/README.md) | परीक्षण के विषय के रूप में प्रॉम्प्ट, AIME समस्याओं पर स्कोर किया गया |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | एक विज़न-भाषा एक्सट्रैक्टर, ग्राउंड-ट्रुथ तालिकाओं के विरुद्ध स्कोर किया गया |
| [**torch performance**](../../examples/torch-performance/README.md) | संख्यात्मक सहनशीलता के भीतर CUDA पर एक तेज़ मॉड्यूल |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | CUDA पर मल्टी-हेड अटेंशन: flash attention या Triton |

## 📚 दस्तावेज़ीकरण और सहायता

- **पूरा दस्तावेज़ीकरण**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [इंस्टॉलेशन](https://docs.leeroo.com/docs/installation): कोडिंग-एजेंट CLI, पैकेज, और `kapso doctor`
  - [क्विकस्टार्ट](https://docs.leeroo.com/docs/first-build): आपका पहला अभियान
  - [CLI संदर्भ](https://docs.leeroo.com/docs/reference/cli): हर कमांड, फ़्लैग और डिफ़ॉल्ट
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso` पर हर सार्वजनिक मेथड
  - [कॉन्फ़िगरेशन](https://docs.leeroo.com/docs/reference/configuration): `config.yaml` की हर कुंजी
  - [learn क्या उत्पन्न करता है](https://docs.leeroo.com/docs/learn/what-learn-produces): सबकों का बैंक, हर सबक एक विश्वसनीयता स्कोर, एक सीमा और उसके साक्ष्य के साथ, और [विकास मोड](https://docs.leeroo.com/docs/learn/development-mode)
  - [स्मार्ट प्रयोग](https://docs.leeroo.com/docs/evolve/overview): प्रयोगों का वृक्ष और उसके मॉड्यूल, और [एक प्रयोग के भीतर क्या होता है](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [अनुभव](https://docs.leeroo.com/docs/experience): आपका बैंक, Leeroopedia, खोजने योग्य ज्ञान में बदली गई रिपॉज़िटरियाँ और पेपर
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal और अन्य लक्ष्य
  - [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code, Codex और Cursor के लिए कमांड, कौशल और MCP सर्वर
- **अपने कोडिंग एजेंट से**: दस्तावेज़ `https://docs.leeroo.com/mcp` पर एक MCP सर्वर चलाते हैं, और हर पृष्ठ Markdown के रूप में परोसा जाता है; देखें [llms.txt](https://docs.leeroo.com/llms.txt) और [आपके कोडिंग एजेंट में Kapso](https://docs.leeroo.com/docs/coding-agent-skills)।
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **अपने कोडिंग एजेंट को Kapso चलाने दें**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) के अंतर्गत प्लगइन में दस कमांड, छह कौशल और MCP सर्वर हैं, हर होस्ट के लिए एक इंस्टॉल पंक्ति (ऊपर चरण 0); OpenCode कौशलों को एक फ़ोल्डर के रूप में लोड करता है (`ln -s <clone>/plugin/skills .opencode/skills`)। देखें [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills)।
- **अपने कोडिंग एजेंट को Kapso के ज्ञान टूल दें**: पैकेज के साथ इंस्टॉल होने वाला `kapso-mcp`, वेब शोध, रिपॉज़िटरी की मेमोरी, अभियान के प्रयोग इतिहास और सबकों के बैंक को उस सत्र में परोसता है जिसमें आप काम करते हैं; देखें [MCP गेट](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)।
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; फिर ~/.codex/config.toml में [mcp_servers.kapso] के अंतर्गत `startup_timeout_sec = 60` (व्यस्त मशीन पर Codex के दस सेकंड कम पड़ते हैं)
  ```
- **समुदाय**: [Discord](https://discord.gg/hqVbPNNEZM)
- **वेबसाइट**: [leeroo.com](https://leeroo.com)


## एंटरप्राइज़ के लिए Kapso

एंटरप्राइज़ में, Kapso एजेंट प्रारंभिक डेटा और AI इंजीनियरिंग harness तथा सबकों के एक बैंक के साथ आते हैं, और व्यावसायिक उद्देश्यों को अपने आप काम करने वाले AI और डेटा सिस्टमों में बदलते हैं, कंपनी के आर्टिफ़ैक्ट्स, रेपो, पाइपलाइन, डेटा स्रोतों और अन्य की जाँच करते हुए, ताकि चलते-चलते एंटरप्राइज़ डेटा एस्टेट का निर्माण होता जाए। अपने एंटरप्राइज़ कार्यों के लिए Kapso को ऑनबोर्ड करने हेतु, [हमसे बात करें](https://leeroo.com/contact-us)।

## योगदान

हम योगदान का स्वागत करते हैं! शुरुआत कैसे करें, इसके विवरण के लिए कृपया हमारी [योगदान गाइड](../../CONTRIBUTING.md) देखें।

## योगदानकर्ता

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

[contrib.rocks](https://contrib.rocks) के साथ बनाया गया।
