<!-- Hindi translation of README.md, generated with scripts/translate_readme.py (Claude Code). Keep it in step: README.md and this file change in the same commit, and scripts/check_readme_translations.py fails when a link, a number or a line of code differs. -->
<h1 align="center">Kapso</h1>

<h4 align="center">एक इंजन जो प्रयोग चलाता है, सबसे अच्छे समाधान को परिनियोजित करता है, और हर रन से सीखता है</h4>
<p align="center"><sub>स्मार्ट प्रयोग · सीखने का लूप · परिनियोजन · डोमेन अनुकूलन</sub></p>

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
  अगर आपको यह प्रोजेक्ट पसंद है, तो कृपया इसे स्टार देकर हमारा समर्थन करें ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter loop of the same shape that ends in a deployed solution; every run's trajectories feed the learning loop, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **अगर आप इस रिपॉज़िटरी को देख रहे एक AI एजेंट हैं:** Kapso प्लगइन, जिसमें सेटअप करने,
> उद्देश्य बताने, harness लिखने और किसी रन की समीक्षा करने के स्किल शामिल हैं,
> एक ही पंक्ति में इंस्टॉल हो जाता है: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`)।

---

## समाचार

- **IOAI 2026 · AI मॉडल ट्रैक।** [AI का अंतर्राष्ट्रीय ओलंपियाड](https://ioai-official.org) AI युग का IMO है: 471 प्रतियोगी, 108 देशों और क्षेत्रों से, और एक GPU के बजट में विशेषज्ञों द्वारा तैयार किए गए छह कार्य। 2026 में इसने [IOAI²](https://ioai-official.org/ai-model-track/) शुरू किया, जहाँ AI सिस्टम बिना किसी निगरानी के 6 घंटे के दो सत्रों में *वही* परीक्षा देते हैं। घड़ी शुरू होने के बाद कोई भी इंसान कुछ भी हल, ठीक या बेहतर नहीं कर सकता। Kapso ने 14 संस्थापक AI प्रतिभागियों में से एक के रूप में भाग लिया।
  - 🎓 **सभी 471 मनुष्यों से अधिक अंक**: कुल 536.07, हॉल में मौजूद हर प्रतियोगी से ऊपर।
  - 🏆 **IOAI² ग्रैंड मास्टर ट्रॉफ़ी**: भाग लेने वाले सभी AI सिस्टमों में शीर्ष 3। [पूरे परिणाम →](../../benchmarks/ioai2026/README.md) · [ट्रैक पर IOAI की समीक्षा →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **तीनों [RelBench](../../benchmarks/relbench/README.md) लीडरबोर्ड पर पहला स्थान**: रिलेशनल डेटा पर प्रेडिक्टिव ML के लिए Stanford के बेंचमार्क पर, Kapso आधिकारिक लीडरबोर्ड में आगे है: वर्गीकरण (81.2 AUROC, 12 कार्यों पर, जबकि [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) का 79.3 और Stanford STAR के RT-J का 79.0), रिग्रेशन (0.2476 NMAE, 9 कार्यों पर, जबकि RT-J का 0.2711 और KumoRFM-v2 का 0.2847, यानी 8.7% कम त्रुटि) और अनुशंसाएँ (18.4 MAP, 10 कार्यों पर, जबकि ID-GNN का 9.3, जो बाकी प्रविष्टियों में सर्वश्रेष्ठ है)। हर संख्या [आधिकारिक RelBench लीडरबोर्ड](https://star-project.stanford.edu/relbench/leaderboard/) पर दर्ज एक प्रविष्टि है।

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia](https://leeroopedia.com)**: डेटा और ML क्षेत्रों में हमारे सबकों की एक विकी, जिसमें शोध-पत्रों और रिपॉज़िटरी पर प्रयोग करके निकाले गए लगभग 100k पृष्ठ हैं, जो हर महीने लगभग 6M अनुरोध पूरे करती है, जिनमें से लगभग 28% AI एजेंटों से आते हैं। Kapso इससे MCP के ज़रिए जुड़ता है: इसके सत्र विचार-निर्माण और कार्यान्वयन के दौरान ज्ञान खोजने, योजनाएँ बनाने और विफलताओं का निदान करने के लिए इसका उपयोग करते हैं।
- **[Moltbook एजेंट 🦞](https://www.moltbook.com/)**: ऐसे AI एजेंट बनाएँ जो दूसरे एजेंटों को ऑप्टिमाइज़ करें और Moltbook पर बहस करें! [शुरू करें →](../../moltbook_bot/README.md)
- **तकनीकी रिपोर्ट**: हमारी तकनीकी रिपोर्ट अब उपलब्ध है! [पेपर पढ़ें](https://arxiv.org/abs/2601.21526)
- **[MLE-Bench](../../benchmarks/mle/README.md) पर #1**: KAPSO ने Kaggle ML प्रतियोगिताओं (MLE Benchmark) पर ओपन-सोर्स सिस्टमों में शीर्ष रैंकिंग हासिल की।

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **[ALE-Bench](../../benchmarks/ale/README.md) पर #1**: KAPSO ने लंबी अवधि वाली एल्गोरिदमिक खोज समस्याओं (ALE Benchmark) पर शीर्ष रैंकिंग हासिल की।

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## Kapso क्या है?

Kapso एक इंजन है जो प्रयोग चलाता है, सबसे अच्छे समाधान को परिनियोजित करता है, और हर रन से सीखता है। कोई उद्देश्य, प्रारंभिक सफलता मानदंड और मूल्यांकन मेट्रिक दिए जाने पर, यह जाँच-पड़ताल करता है, सबसे अच्छे विचार चुनकर प्रयोग चलाता है, कभी-कभी मूल्यांकन मेट्रिक और सफलता मानदंड को बेहतर बनाने के लिए फ़ीडबैक देता है, सबसे अच्छे समाधान को वांछित स्टैक पर परिनियोजित करता है, और हर रन से सीखता है। यह सबकों का एक बैंक रखता है, जिसमें हर सबक के साथ एक विश्वसनीयता स्कोर और यह सीमा होती है कि वह कहाँ लागू होता है, ताकि ज्ञान समय के साथ केवल जमा न हो, बल्कि चक्रवृद्धि की तरह बढ़े।

हमने LinkedIn और Meta AI सहित बड़ी कंपनियों के भीतर AI सिस्टम बनाए और उन्हें लाखों उपयोगकर्ताओं वाले उत्पादों के लिए परिनियोजित किया। Kapso वही वर्कफ़्लो एक इंजन के रूप में है: ओपन सोर्स, [Leeroo](https://leeroo.com) द्वारा, और PyPI पर [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/) के नाम से प्रकाशित।

### चार भाग

| भाग | क्रिया | यह क्या करता है |
|------|------|--------------|
| [**स्मार्ट प्रयोग**](https://docs.leeroo.com/docs/build/brief) | `.build()` | प्रयोग लूप विशेषीकृत मॉड्यूलों का एक समूह है जो एक ही डेटा संरचना साझा करते हैं, यानी प्रयोगों का वृक्ष: हर नोड परिकल्पना, कार्यान्वयन (git ब्रांच और तकनीकी कठिनाइयाँ), इस्तेमाल हुआ बजट और उद्देश्य से दूरी दर्ज करता है। विचार-निर्माण पिछले प्रयोगों और सबकों के बैंक के आधार पर विचार प्रस्तावित करता है, और हर विचार बताता है कि वह किन्हीं ख़ास पिछले प्रयोगों से कैसे अलग है और कौन से सबक अपनाता है; एक रीरैंकर उन्हें लक्ष्य के मुक़ाबले अपेक्षित प्रतिफल के आधार पर, समय और बजट को तौलते हुए, क्रम देता है। कार्यान्वयनकर्ता पैरेंट नोड से फ़ोर्क की गई ब्रांच पर निर्माण करता है, और एक सत्यापक नोड को सुरक्षा नियमों और सफलता मानदंड के विरुद्ध जाँचता है, तथा तय करता है कि स्कोर वैध है या नहीं और उद्देश्य पूरा हुआ या नहीं। सफलता मानदंड और टेस्ट किसी रन के दौरान विकसित हो सकते हैं: मॉड्यूल तर्कसंगत अपडेट प्रस्तावित करते हैं, और सत्यापक उन्हें स्वीकृत या अस्वीकृत करता है। [यह कैसे काम करता है →](https://docs.leeroo.com/docs/evolve/overview) |
| [**सीखने का लूप**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | हर सबक ज्ञान का एक अंश, एक विश्वसनीयता स्कोर, वह कहाँ लागू होता है इसकी सीमा-शर्तें और उसके अपडेट के पीछे का साक्ष्य दर्ज करता है। हर रन के बाद, `learn()` इस्तेमाल किए गए सबकों की विश्वसनीयता और सीमाओं को परिणाम पर उनके प्रभाव के आधार पर अपडेट करता है, मिलते-जुलते ट्रेस से आगमन द्वारा सबक निकालता है या किसी एक साक्ष्य और उसकी क्रियाविधि से निगमन द्वारा एक सबक निकालता है, ओवरलैप करने वाले सबकों को मिला देता है, और अपडेट किए गए सबकों का A/B परीक्षण करता है, यानी वही build उनके साथ और उनके बिना चलाता है, ताकि अपडेट अनुमान से नहीं बल्कि माप से तय हों। अगले build का विचार-निर्माण और कार्यान्वयन हर सबक की विश्वसनीयता और सीमाओं को पढ़कर तय करते हैं कि उस पर कब भरोसा करना है, न कि ज्ञान पर बस RAG चला देते हैं। [Experience →](https://docs.leeroo.com/docs/experience) बताता है कि किसी build को क्या परोसा जाता है: आपका बैंक, आपकी टीम का साझा बैंक, Leeroopedia, या किसी रिपॉज़िटरी अथवा शोध किए गए प्रश्न से इनजेस्ट किया गया कॉर्पस। |
| [**परिनियोजन**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | वृक्ष के सबसे अच्छे समाधान को लक्ष्य स्टैक पर चलते हुए सॉफ़्टवेयर में बदलता है, प्रयोग लूप के एक छोटे संस्करण के ज़रिए: एक सत्र नोड की एक प्रति को अनुकूलित करता है, उसे परिनियोजित करता है, और हेल्थ चेक तथा एक काम करती हुई कॉल दिखाता है। Docker, Modal, BentoCloud और LangGraph Platform साथ आते हैं, और आप harness में अपना लक्ष्य स्टैक जोड़ सकते हैं। |
| [**डोमेन अनुकूलन**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | Harness Kapso के डोमेन पैकेज हैं: प्रेडिक्टिव मॉडलिंग जैसे किसी क्षेत्र के लिए ज्ञान और वर्कफ़्लो, ताकि build को शून्य से प्रयोग करना और सीखना न पड़े; प्रेडिक्टिव मॉडलिंग और प्रतिस्पर्धी प्रोग्रामिंग Kapso के साथ आते हैं। अपना डोमेन ज्ञान लाने के लिए, Kapso द्वारा परिभाषित इंटरफ़ेस के ज़रिए एक harness लिखें: आपके निर्देश, कोड, गार्डरेल और जाँचें, जिन्हें प्लगइन में `/kapso:harness` या CLI में `kapso harness` से लिखा जाता है। |

## 🚀 त्वरित शुरुआत

### इंस्टॉलेशन

**0. कोडिंग एजेंट में हैं?** Kapso प्लगइन इंस्टॉल करें और उसे नीचे दिए गए
चरणों में आपका साथ देने दें:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

फिर `/kapso:setup` इंस्टॉल और सत्यापन करता है, और `/kapso:build <what you want>`
पहला अभियान शुरू करता है (Codex पर: `$kapso-setup`, `$kapso-build <what you want>`)। देखें [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills)।

**1. पूर्वापेक्षाएँ।** Kapso अपना इन्फ़रेंस कोडिंग एजेंट CLI के ज़रिए चलाता है
(सीधे API का कोई फ़ॉलबैक नहीं है), इसलिए कुछ भी काम करने से पहले आपको Node.js और
दोनों एजेंट CLI लॉग इन अवस्था में चाहिए:

```bash
# Node.js 18+ (https://nodejs.org), फिर:
npm install -g @openai/codex            # शोध, जजिंग, यूटिलिटी
codex login

npm install -g @anthropic-ai/claude-code  # विचार-निर्माण + कार्यान्वयन (डिफ़ॉल्ट होस्ट)
claude auth login
```

OpenAI की कुंजी वैकल्पिक है: केवल नॉलेज ग्राफ़ और पिछले प्रयोगों पर सिमैंटिक खोज
(आपके कॉन्फ़िग में `models.embedding`) ही टेक्स्ट को एम्बेड करते हैं, और
जनरेशन कभी इसका उपयोग नहीं करता। जब आप इनमें से किसी को चालू करें:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Leeroopedia जोड़ें (अनुशंसित)।** एक कुंजी अभियान के हर सत्र को
[Leeroopedia](https://leeroopedia.com) देती है, जो ML और AI फ़्रेमवर्क का एक
होस्टेड नॉलेज बेस है: कोई फ़्रेमवर्क वास्तव में कैसे व्यवहार करता है, योजनाओं की समीक्षा,
दस्तावेज़ित हाइपरपैरामीटर रेंज, विफलताओं के निदान। कुछ भी इंस्टॉल नहीं करना है;
[Leeroopedia डैशबोर्ड](https://app.leeroopedia.com/dashboard) पर नए खाते के साथ
$20 का मुफ़्त क्रेडिट मिलता है।

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. पैकेज इंस्टॉल करें** (Python 3.10+):

```bash
pip install leeroo-kapso
```

पैकेज का नाम `leeroo-kapso` है। PyPI पर `kapso` नाम का पैकेज एक असंबंधित
WhatsApp टूल है जो अपना ख़ुद का `kapso` कमांड इंस्टॉल करता है और इस कमांड को ढक देता है;
अगर वह मौजूद है, तो पहले `pip uninstall kapso` चलाएँ।

**4. सेटअप सत्यापित करें:**

```bash
kapso doctor
```

`doctor` बताता है कि **आपके कॉन्फ़िग** को वास्तव में क्या चाहिए, और जो कुछ भी गायब है
उसका सटीक समाधान बताता है। आवश्यकताएँ कॉन्फ़िग के अनुसार तय होती हैं, इसलिए पूरी तरह codex वाले
सेटअप से कभी `claude` नहीं माँगा जाता। केवल किसी एक क्रिया की आवश्यकताएँ देखने के लिए
इसे उसी क्रिया तक सीमित करें:

```bash
kapso doctor build             # या learn / deploy / experience
```

ये वही जाँचें हैं जो क्रिया ख़ुद कोई भी काम करने से पहले चलाती है:
CLI गायब होने पर `kapso.build(...)` किसी सत्र के बहुत भीतर जाकर नहीं, बल्कि कुछ ही सेकंड में
विफल हो जाता है। `[-- ]` से चिह्नित आइटम वैकल्पिक हैं; वे उन सुविधाओं को सीमित करते हैं
जिनकी आपको शायद ज़रूरत न हो (docker, Weaviate, Neo4j, deploy के लक्ष्य)।

**या यह सब अपने कोडिंग एजेंट से करवाएँ:**
[सेटअप प्रॉम्प्ट](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
को Claude Code, Codex या OpenCode में पेस्ट करें।

**इंडेक्स के लिए ज़रूरी स्टोर (वैकल्पिक)**: `kapso experience add --repo`
और `--research` एक कॉर्पस को लोकल Weaviate + Neo4j में इनजेस्ट करते हैं। सोर्स
चेकआउट से:

```bash
bash scripts/start_infra.sh   # docker के ज़रिए दोनों को शुरू करता है
```

**सोर्स से (डेवलपमेंट के लिए)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

पुराना aider एडैप्टर एक एक्स्ट्रा है (`pip install "leeroo-kapso[aider]"`,
Python <3.13); डिफ़ॉल्ट claude/codex एजेंटों को किसी एक्स्ट्रा की ज़रूरत नहीं है।

### बुनियादी उपयोग

मुख्य लूप को ऊपर दी गई पूर्वापेक्षाओं के अलावा किसी चीज़ की ज़रूरत नहीं है:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # शुरू करने के लिए किसी नॉलेज ग्राफ़ की ज़रूरत नहीं

# Build: प्रयोग के ज़रिए एक समाधान। अभियान शुरू होते ही
# `status: <path>` प्रिंट करता है; इसे दूसरे टर्मिनल से लाइव देखें:
#     kapso watch ./campaign
# अगर किसी सत्र को कुछ ऐसा चाहिए जो केवल आप दे सकते हैं (कोई API कुंजी, कोई फ़ाइल),
# तो अभियान रुक जाता है और `kapso inbox reply` उसे फिर शुरू करता है:
#     https://docs.leeroo.com/docs/build/inbox
# पुल रिक्वेस्ट या इश्यू के लिए परिणाम को Markdown के रूप में साझा करें:
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # या शून्य से शुरू करने के लिए इसे छोड़ दें
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# अभी चलाए गए अभियान से सीखें: प्रक्षेप पथ को खंगालें, सबकों को
# ग्रेड दें, और साक्ष्य के आधार पर आँके गए नॉलेज कार्ड बैंक में जमा करें। बैंक (एक लोकल
# git रिपॉज़िटरी) पहले उपयोग पर अपने आप बन जाता है; सबक आपकी
# मशीन पर ही रहते हैं जब तक आप उन्हें साझा नहीं करते:
#     kapso experience share <git-url>
# इसके बाद हर learn() बैंक को वहाँ पुश करता है।
lesson = kapso.learn(solution)
print(lesson.explain())

# फिर से build करें, इस बार अभी कमाए गए कार्ड के साथ: build को ठीक वही
# experience परोसा जाता है जिसे वह लोड करता है।
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

स्टोर चालू हों, तो आप बाहरी ज्ञान को भी किसी इंडेक्स में इनजेस्ट कर सकते हैं
और उसे उसी तरह परोस सकते हैं:

```python
from kapso import Kapso

kapso = Kapso()

# एक बार: कोई रिपॉज़िटरी, या किसी प्रश्न के लिए शोध किया गया वेब, एक इंडेक्स में
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# इंडेक्स लोड करने वाला हर build उसमें खोजता है; Leeroopedia अपनी कुंजी के सहारे चलता है
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

बजट के बारे में एक बात: इनजेस्ट वह सब कुछ निकालता है जो सामग्री से निकल सकता है।
एक अकेला प्रश्न दर्जनों जुड़े हुए पृष्ठों और घंटों लंबे इनजेस्ट में
बदल सकता है; इनजेस्ट का समय निकाले जा सकने वाले सार के अनुपात में बढ़ता है, इसलिए वही
स्रोत जोड़ें जिनमें आप खोजेंगे, न कि हर वह स्रोत जो आपके पास है।

और किसी समाधान को चलते हुए सॉफ़्टवेयर में बदलने के लिए:

```python
software = kapso.deploy(solution, target="local")   # या docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # वह कॉल जिसे परिनियोजन ने प्रमाणित किया
software.stop()
```

### किसी अभियान को सीमित कैसे करूँ?

`max_iterations`, `time_budget_minutes` और `cost_budget` का उपयोग करके इटरेशन, बीते समय और USD में मॉडल ख़र्च (यथासंभव प्रयास के आधार पर) को सीमित करें।
प्लैटफ़ॉर्म लागतों के लिए, `search_strategy.params.cost` में अपने एस्टिमेटर, `cap_per_candidate` और `campaign_budget` के साथ [लागत-सचेत चयन](https://docs.leeroo.com/docs/evolve/cost-aware-selection) सक्षम करें; इसका `weight` चयन को सस्ते उम्मीदवारों को वरीयता देने की सुविधा भी देता है।

### मॉडल चुनना

Kapso जिस भी मॉडल का उपयोग करता है, उसका नाम एक ही कॉन्फ़िग फ़ाइल में दर्ज है। पैकेज के साथ आने वाला डिफ़ॉल्ट
build सत्रों को `claude-opus-5` पर, सीखने वाले क्रू को
`claude-fable-5` पर, और codex भूमिकाओं को `gpt-5.6-sol` पर चलाता है, लेकिन मॉडल की उपलब्धता
सब्सक्रिप्शन पर निर्भर करती है (कोई प्लान एक मॉडल पर कैप लगा सकता है जबकि दूसरा उपलब्ध कराता है)।
अलग मॉडलों पर चलाने के लिए, पैकेज वाला कॉन्फ़िग कॉपी करें, उसे संपादित करें, और
Kapso को अपने कॉन्फ़िग की ओर इंगित करें:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# उदाहरण: सीखने वाले क्रू को fable की जगह opus पर चलाएँ:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

केवल टेक्स्ट वाली इन्फ़रेंस भूमिकाएँ `openai_compatible` Chat Completions को चुन सकती हैं।
मॉडल और एंडपॉइंट अपने कॉन्फ़िग में सेट करें; क्रेडेंशियल एनवायरनमेंट में ही रहते हैं।
कॉन्फ़िग को `Kapso(config=...)` या `kapso build --config ...` से लोड करें।
कोडिंग और वेब शोध के लिए अब भी संबंधित टूल वाले एजेंट ज़रूरी हैं।
देखें [कॉन्फ़िगरेशन का उदाहरण और लिमिट](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api)।

लंबे रन से पहले, अपने कॉन्फ़िग में दर्ज हर मॉडल को अपने
वास्तविक सब्सक्रिप्शन के विरुद्ध प्रोब करें: जिस मॉडल को आपका लॉगिन उपलब्ध नहीं करा सकता, वह यहीं
कुछ सेकंड में विफल हो जाता है, न कि रन शुरू होने के घंटों बाद। जिस मॉडल को आप उपलब्ध करा सकते हैं
उस पर लगा उपयोग कैप एक टोकन वाले प्रोब को दिखाई नहीं देता:

```bash
kapso doctor --models                            # पैकेज वाला कॉन्फ़िग
kapso doctor --models --config kapso-config.yaml # आपका
kapso doctor learn --models                      # केवल सीखने वाले क्रू
```

उस प्रोब को हर कॉल का हिस्सा बनाने के लिए, ताकि उसे चलाना याद न रखना पड़े,
इसे अपने कॉन्फ़िग में चालू करें। इसकी लागत हर अलग
`{cli, model}` जोड़ी के लिए एक फेंकने लायक टोकन है, जो बिना निगरानी के कई घंटे चलने वाले
`learn()` से पहले वसूल है:

```yaml
preflight:
  enabled: true            # हर क्रिया से पहले स्टैटिक जाँचें (डिफ़ॉल्ट)
  live_model_probe: true   # + एक टोकन वाले मॉडल प्रोब (डिफ़ॉल्ट: false)
```

मॉडल बदलने से *गति* भी बदलती है: कॉन्फ़िग में क्रू के `timeout_minutes` कैप
डिफ़ॉल्ट मॉडलों पर कैलिब्रेट किए गए थे, और बदला हुआ मॉडल जो
ज़्यादा देर तक रीज़निंग करता है, उसके लिए इन्हें बढ़ाना पड़ सकता है।

विस्तृत इंटीग्रेशन चरणों के लिए, [त्वरित शुरुआत](https://docs.leeroo.com/docs/first-build) और [इंस्टॉलेशन](https://docs.leeroo.com/docs/installation) गाइड देखें।

## उदाहरण

हर उदाहरण एक कार्य है: एक ब्रीफ़, एक शुरुआती रिपॉज़िटरी, डेटा या उसे पाने का तरीक़ा, और अधिकतम बारह पंक्तियों का एक ड्राइवर; [examples/README.md](../../examples/README.md) में मिनट, वह एक कमांड और यह जानकारी है कि कोई परिणाम दर्ज है या नहीं।

| उदाहरण | यह क्या दिखाता है |
|---------|---------------|
| [**churn first build**](../../examples/churn-first-build/README.md) | आपका पहला build: एक मौजूद मूल्यांकक, एक बेसलाइन, एक लक्ष्य, और रिकॉर्ड |
| [**fraud no evaluator**](../../examples/fraud-no-evaluator/README.md) | कोई मूल्यांकक नहीं: अभियान का पहला नोड ब्रीफ़ के प्रोटोकॉल से एक मूल्यांकक बनाता है और हर उम्मीदवार को उसी के विरुद्ध मापा जाता है |
| [**pipeline dbt**](../../examples/pipeline-dbt/README.md) | DuckDB पर एक dbt पाइपलाइन जिसके एक्सपोर्ट में दोष हैं: डेटा कॉन्ट्रैक्ट का पालन सुनिश्चित किया गया |
| [**trading**](../../examples/trading/README.md) | build पर एक harness: एक सेटअप सत्र, हर सत्र में पेज और हुक, हर प्रयास पर सत्यापन |
| [**full loop**](../../examples/full-loop/README.md) | Build, समीक्षा, learn, सबक के साथ फिर से build, deploy: एक Kaggle टैबुलर मॉडल शुरू से अंत तक |
| [**prompt harness**](../../examples/prompt-harness/README.md) | प्रॉम्प्ट ही परीक्षण के अधीन वस्तु है, जिसे AIME समस्याओं पर स्कोर किया जाता है |
| [**agent scaffold**](../../examples/agent-scaffold/README.md) | एक विज़न-लैंग्वेज एक्सट्रैक्टर, जिसे ग्राउंड-ट्रुथ तालिकाओं के विरुद्ध स्कोर किया जाता है |
| [**torch performance**](../../examples/torch-performance/README.md) | संख्यात्मक टॉलरेंस के भीतर CUDA पर एक तेज़ मॉड्यूल |
| [**gpu kernels**](../../examples/gpu-kernels/README.md) | CUDA पर मल्टी-हेड अटेंशन: फ़्लैश अटेंशन या Triton |

## समर्थित बेंचमार्क

| बेंचमार्क | विवरण |
|-----------|-------------|
| [**MLE-Bench**](../../benchmarks/mle/README.md) | OpenAI का ML-इंजीनियरिंग बेंचमार्क: टैबुलर, विज़न, टेक्स्ट और ऑडियो में पूरी प्रतियोगिताएँ, कच्चे डेटा से लेकर ग्रेड किए गए सबमिशन तक |
| [**ALE-Bench**](../../benchmarks/ale/README.md) | Sakana AI का एल्गोरिदमिक-ऑप्टिमाइज़ेशन बेंचमार्क: घंटों लंबी खोजों में कॉन्टेस्ट ह्यूरिस्टिक डिज़ाइन करना, लागू करना और उन पर इटरेट करना |
| [**RelBench**](../../benchmarks/relbench/README.md) | रिलेशनल डेटा पर प्रेडिक्टिव ML के लिए Stanford का बेंचमार्क: वर्गीकरण, रिग्रेशन और अनुशंसा, सीधे Amazon, H&M, Stack Exchange, क्लिनिकल ट्रायल और अन्य के मल्टी-टेबल डेटाबेस से |
| [**IOAI 2026**](../../benchmarks/ioai2026/README.md) | विज़न, भाषा और ऑप्टिमाइज़ेशन में समयबद्ध ओलंपियाड ML: विशेषज्ञों द्वारा तय किए गए कार्य, प्रतियोगिता का हार्डवेयर, शून्य मानवीय मदद |

हर बेंचमार्क का एक दस्तावेज़ पृष्ठ भी है, जिसमें बताया गया है कि उसे कैसे चलाएँ, उसके CLI विकल्प क्या हैं और आउटपुट में क्या होता है:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 दस्तावेज़ और सहायता

- **पूरा दस्तावेज़**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [इंस्टॉलेशन](https://docs.leeroo.com/docs/installation): कोडिंग एजेंट CLI, पैकेज, और `kapso doctor`
  - [त्वरित शुरुआत](https://docs.leeroo.com/docs/first-build): आपका पहला अभियान
  - [CLI संदर्भ](https://docs.leeroo.com/docs/reference/cli): हर कमांड, फ़्लैग और डिफ़ॉल्ट
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api): `Kapso` का हर सार्वजनिक मेथड
  - [कॉन्फ़िगरेशन](https://docs.leeroo.com/docs/reference/configuration): `config.yaml` की हर कुंजी
  - [learn क्या बनाता है](https://docs.leeroo.com/docs/learn/what-learn-produces): सबकों का बैंक, जिसमें हर सबक के साथ एक विश्वसनीयता स्कोर, एक सीमा और उसका साक्ष्य होता है, और [डेवलपमेंट मोड](https://docs.leeroo.com/docs/learn/development-mode)
  - [प्रयोग लूप](https://docs.leeroo.com/docs/evolve/overview): प्रयोगों का वृक्ष और उसके मॉड्यूल, और [एक प्रयोग के भीतर क्या होता है](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience): आपका बैंक, Leeroopedia, और खोजने योग्य ज्ञान में बदले गए रिपॉज़िटरी और शोध-पत्र
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets): local, Docker, Modal और अन्य लक्ष्य
  - [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills): Claude Code, Codex और Cursor के लिए कमांड, स्किल और MCP सर्वर
- **अपने कोडिंग एजेंट से**: दस्तावेज़ `https://docs.leeroo.com/mcp` पर एक MCP सर्वर चलाते हैं, और हर पृष्ठ Markdown के रूप में परोसा जाता है; देखें [llms.txt](https://docs.leeroo.com/llms.txt) और [आपके कोडिंग एजेंट में Kapso](https://docs.leeroo.com/docs/coding-agent-skills)।
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **अपने कोडिंग एजेंट से Kapso चलवाएँ**: [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) के अंतर्गत मौजूद प्लगइन में दस कमांड, छह स्किल और MCP सर्वर हैं, हर होस्ट के लिए एक इंस्टॉल पंक्ति (ऊपर चरण 0); OpenCode स्किल को एक फ़ोल्डर के रूप में लोड करता है (`ln -s <clone>/plugin/skills .opencode/skills`)। देखें [Kapso प्लगइन](https://docs.leeroo.com/docs/coding-agent-skills)।
- **अपने कोडिंग एजेंट को Kapso के नॉलेज टूल दें**: पैकेज के साथ इंस्टॉल होने वाला `kapso-mcp` उस सत्र को, जिसमें आप काम करते हैं, वेब शोध, रिपॉज़िटरी की मेमोरी, अभियान का प्रयोग इतिहास और सबकों का बैंक उपलब्ध कराता है; देखें [MCP गेट](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent)।
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; फिर ~/.codex/config.toml में [mcp_servers.kapso] के नीचे `startup_timeout_sec = 60` (व्यस्त मशीन पर Codex के दस सेकंड कम पड़ते हैं)
  ```
- **समुदाय**: [Discord](https://discord.gg/hqVbPNNEZM)
- **वेबसाइट**: [leeroo.com](https://leeroo.com)


## एंटरप्राइज़ के लिए Kapso

हमारा एंटरप्राइज़ उत्पाद इसी इंजन पर बना है और उन उलझे हुए एंटरप्राइज़ों के भीतर परिनियोजित किया जा चुका है, जहाँ कोई भी इंसान पूरे डेटा एस्टेट की जाँच करके उसे समझ नहीं सकता: हमने एक कंपनी को उसके पुराने डेटा प्लैटफ़ॉर्म से एक नए लेकहाउस पर स्थानांतरित किया, जिसमें 30 से अधिक सोर्स-सिस्टम परिवार, हज़ारों तालिकाएँ और ट्रांसफ़ॉर्मेशन, और सैकड़ों BI रिपोर्ट शामिल थीं। यह जिस भी कार्य पर काम करता है, वह आपके सिस्टम, आपके डेटा और मुश्किल से सीखे गए आपके नियमों के बारे में सबकों के बैंक को समृद्ध करता है, इसलिए यह जितना लंबा काम करता है, आपकी कंपनी के लिए उतना ही बेहतर होता जाता है। अपने चुनौतीपूर्ण एंटरप्राइज़ कार्यों के लिए Kapso को अपनाने के लिए, [हमसे बात करें](https://leeroo.com/contact-us)।

## अनुवाद

README अन्य भाषाओं में उपलब्ध है: सबसे ऊपर दिए गए स्विचर का उपयोग करें, या [`docs/translations/`](../../docs/translations/) ब्राउज़ करें। हर अनुवाद README.md से `scripts/translate_readme.py` द्वारा तैयार किया जाता है और `scripts/check_readme_translations.py` उसे तालमेल में रखता है, जो तब विफल हो जाता है जब कोई अनुवाद किसी लिंक, संख्या या कोड की पंक्ति पर README.md से मेल नहीं खाता। किसी अनुवाद को बेहतर बनाने के लिए, उस फ़ोल्डर की फ़ाइल पर एक पुल रिक्वेस्ट खोलें।

## योगदान

हम योगदान का स्वागत करते हैं! शुरुआत कैसे करें, इसकी जानकारी के लिए कृपया हमारी [योगदान मार्गदर्शिका](../../CONTRIBUTING.md) देखें।

## उद्धरण

अगर आप अपने काम में Kapso का उपयोग करते हैं, तो कृपया तकनीकी रिपोर्ट को उद्धृत करें:

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```
