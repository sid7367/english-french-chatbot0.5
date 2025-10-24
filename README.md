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
Developed by **Sidhaarth Mohandas**  
Project for demonstration of **AI-driven NLP Translation** and interactive chatbot design.

---

💡 *This project showcases the integration of Natural Language Processing and modern AI frameworks to create real-time language translation applications.*

## 🔧 Dependency updates (2025-10-25)

I made a few small, conservative fixes to `requirements.txt` to avoid common resolver conflicts and ensure the environment installs cleanly:

- `fsspec==2025.9.0` → `fsspec==2024.11.0` (2025 release looked unavailable)
- `protobuf==5.29.3` → `protobuf<5,>=3.20.0` (many packages are incompatible with Protobuf 5)
- `psutil==7.0.0` → `psutil==5.9.5` (7.0.0 is not an official release)
- `types-python-dateutil==2.9.0.20241206` → `types-python-dateutil==2.9.0` (normalised anomalous timestamped wheel)

Local validation performed:

```powershell
# create and activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r .\requirements.txt
pip check  # returned: "No broken requirements found."
```

If you see resolver errors on your machine, paste the `pip install` output here and I will adjust specific pins further.
