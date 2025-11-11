# 🧠 SmartTemp: Teaching LLMs to Adjust Their Own Creativity  

**SmartTemp** is an experimental engine that lets **Large Language Models (LLMs)** *self-adjust their temperature dynamically* based on the intent and semantics of the prompt.  
Instead of manually tuning creativity, SmartTemp interprets the prompt context, estimates confidence, and adapts temperature intelligently — blending *precision* and *imagination* on its own.  

---

## 🔍 Why SmartTemp?

Most LLM applications rely on a **static temperature value** — set once, used everywhere.  
But not every prompt deserves the same level of randomness.  

- “Summarize this paper” → needs *low temperature (factual)*  
- “Write a poem about entropy” → needs *high temperature (creative)*  

SmartTemp automates that decision by reading the *prompt itself*, determining its nature, and **adapting the model’s temperature in real time**.

---

## ⚙️ How It Works

SmartTemp combines:
1. **Prompt Analysis Engine** – Classifies the intent (`creative`, `factual`, `analytical`, `instructional`)  
2. **Confidence Scoring** – Estimates how confidently it recognized the intent  
3. **Dynamic Temperature Scaling** – Computes  

temperature = base_temp + (1 - confidence) * scale_factor

yaml
Copy code

so the less certain SmartTemp is, the more exploratory the LLM becomes.  
4. **LLM Integration Layer** – Connects to **Ollama** or any OpenAI-compatible API  
5. **Gradio Interface** – Displays real-time temperature/confidence graphs, prompt summaries, and responses  

---

## 🧩 System Architecture

┌──────────────────────────┐
│ User Prompt │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ SmartTemp Analyzer │
│ (Categorization + Conf.) │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ Dynamic Temperature │
│ Adjustment Formula │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ LLM Integration (Ollama │
│ or OpenAI-compatible) │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ Gradio Visualization UI │
└──────────────────────────┘

yaml
Copy code

---

## 💡 Key Features

- 🧠 **Intent-Aware Temperature Tuning** – Adjusts creativity based on prompt semantics  
- 📊 **Real-Time Visualization** – Temperature & confidence graphs update per response  
- 🔄 **Ollama-Compatible** – Runs locally or with any OpenAI-style API  
- 🎨 **Mock Mode** – Simulates creative, factual, and analytical tones even without a live model  
- 🪄 **Fully Modular** – Each component (analyzer, integration, interface) can be reused independently  

---

## 🧠 Example Interactions

| Prompt | Detected Type | Confidence | Assigned Temp | Behavior |
|--------|----------------|-------------|----------------|-----------|
| “What is the capital of Brazil?” | factual | 0.95 | 0.2 | Precise, single factual answer |
| “Write a story about a robot in love.” | creative | 0.90 | 0.9 | Vivid, expressive storytelling |
| “Explain quantum computing in simple terms.” | analytical | 0.70 | 0.5 | Balanced technical clarity |

---

## 🚀 Quick Setup

> 🧩 SmartTemp is designed as a concept prototype.  
> You can run it locally to explore adaptive temperature logic — no hosting or credentials needed.

**Requirements**
```bash
pip install gradio openai
Run the Portal

bash
Copy code
python smarttemp.py
If you have Ollama installed and running, SmartTemp will automatically use it as the LLM backend.
Otherwise, it runs in mock mode, generating sample outputs for visualization.

🔬 Example Output
vbnet
Copy code
Prompt: Explain gravity like I’m five
Category: Instructional
Confidence: 0.83
Assigned Temperature: 0.43
Response: Imagine you drop your toy—it falls because the Earth is giving it a gentle hug called gravity.
🧭 Future Extensions
🔍 Replace rule-based classification with embedding-based semantic clustering

🔁 Add feedback loops for adaptive learning based on user rating

🧩 Integrate with LangChain / LangGraph for dynamic reasoning chains

☁️ Deploy as an API layer for existing chat or agent systems

---

-Suhruth Krishna Yalamanchili
Data Scientist | AI Engineer | Writer
Exploring intersections of cognitive science and computational intelligence.



