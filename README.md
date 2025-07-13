
# 📄 ResumeRanker

A smart AI-powered tool that ranks resumes based on their relevance to a given Job Description (JD).  
Built with **Python**, **Streamlit**, and **Sentence-BERT** for semantic matching.


## 🚀 Features

- Upload multiple resumes (`.pdf`, `.docx`, `.txt`)
- Paste or type a custom job description (JD)
- Automatically extracts skills, experience, education from each resume
- Computes:
  - ✅ Keyword Match Score
  - 🧠 Skills Match Score
  - 💬 Semantic Similarity
- Ranks resumes from best to least relevant

---

## 🖥️ Demo

Try the app locally:

```bash
streamlit run app.py

## 📦 Installation

### Clone the repository:

```bash
git clone https://github.com/sidaksodhi2/ResumeRanker.git
cd ResumeRanker

## 🛠️ Setup Instructions

### 🔹 Create a virtual environment (Python 3.10 recommended)

```bash
python -m venv resume-env
resume-env\Scripts\activate      # On Windows
# source resume-env/bin/activate  # On Mac/Linux

