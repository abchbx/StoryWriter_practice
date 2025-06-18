# StoryWriter

**StoryWriter** is a multi-agent framework for generating high-quality **long stories** with logical coherence and engaging plots—two major challenges that remain unsolved for most current large language models (LLMs).

---

## ✨ Highlights

Long story generation is hard due to:

1. **Discourse Coherence**
   Maintaining consistency, logic, and completeness throughout the story.

2. **Narrative Complexity**
   Crafting engaging, interwoven plots across characters and events.

To tackle these, we introduce **`StoryWriter`**, a **multi-agent story generation framework** with the following components:

* **🧠 Outline Agent**
  Produces event-based outlines rich in plots, characters, and inter-event relationships.

* **🗂️ Planning Agent**
  Breaks down the outline into chapter-wise plans, ensuring an engaging, interwoven narrative.

* **✍️ Writing Agent**
  Dynamically compresses the story history to generate coherent new content aligned with the current event.

---

## 📊 Results

We conduct both human and automatic evaluations, and **StoryWriter** significantly **outperforms** existing baselines in:
![image](https://github.com/user-attachments/assets/a23eb560-1870-4661-ad10-27d378ca470f)



---

## 📚 Dataset

We use StoryWriter to generate a large-scale long story dataset:

* **\~5,000 stories**
* **Average length: 8,000 words/story**

---

## 📥 Get Started

* **📖 Read Sample Stories**
  Download generated stories here:
  👉 [Tsinghua huggingface Link]([https://cloud.tsinghua.edu.cn/f/6173850b58114951ab7e/](https://huggingface.co/datasets/THU-KEG/LongStory))

* **🛠️ Train Your Own Model**
  Use [LongWriter](https://github.com/THUDM/LongWriter/tree/main) to train on our dataset.
  Replace the original raw file with our training JSON.
