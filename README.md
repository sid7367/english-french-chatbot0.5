# 🗣️ English to French Translation Chatbot

An AI-powered chatbot that translates English text into French using Hugging Face’s **Helsinki-NLP/opus-mt-en-fr** model.  
Built with **Streamlit** for a simple, interactive, and user-friendly interface.

## 🚀 Features
- Real-time English to French translation using state-of-the-art Transformer models  
- Typing animation for natural, conversational feel  
- Mobile-friendly Streamlit interface for smooth demos  
- Easy deployment via Streamlit Cloud or local run  

## 🧠 Model & Dataset
The chatbot leverages Hugging Face’s **MarianMT** model trained on the **OPUS dataset**, a large-scale multilingual parallel corpus used for translation tasks.  
Model: `Helsinki-NLP/opus-mt-en-fr`  
Dataset: **OPUS (Open Parallel Corpus)** — pairs English and French sentences for high-quality neural machine translation.

## 🧩 Installation

Clone the repository:
```bash
git clone https://github.com/<your-username>/english-french-chatbot.git
cd english-french-chatbot
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🧪 Run the Application
To launch the chatbot:
```bash
streamlit run app.py
```

Then open the provided URL (usually `http://localhost:8501/`) in your browser.

## 📦 Requirements
```
streamlit
transformers
torch
sentencepiece
```

## 📸 Example
**Input:**  
`Hello, how are you?`

**Output:**  
`Bonjour, comment allez-vous ?`

## 🧑‍💻 Author
Developed by **<Your Name>**  
Project for demonstration of **AI-driven NLP Translation** and interactive chatbot design.

---

💡 *This project showcases the integration of Natural Language Processing and modern AI frameworks to create real-time language translation applications.*
