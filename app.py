import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# ---------- Translation Function ----------
def translate_to_french(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**tokens)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="English–French Chatbot", layout="wide")
st.title("🇬🇧➡️🇫🇷 English to French Translation Chatbot")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            display: flex;
            flex-direction: column-reverse;
            background-color: #f7f9fc;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .user-bubble {
            background-color: #DCF8C6;
            border-radius: 12px;
            padding: 10px 15px;
            margin: 5px;
            text-align: right;
            width: fit-content;
            max-width: 80%;
            float: right;
            clear: both;
        }
        .bot-bubble {
            background-color: #E9EBEE;
            border-radius: 12px;
            padding: 10px 15px;
            margin: 5px;
            text-align: left;
            width: fit-content;
            max-width: 80%;
            float: left;
            clear: both;
        }
        @media (max-width: 600px) {
            .user-bubble, .bot-bubble {
                max-width: 95%;
                font-size: 16px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Chat Session ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in reversed(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Input Section ----------
st.markdown("---")
col1, col2 = st.columns([8, 1])

with col1:
    user_input = st.text_input("Type your message in English:", key="input", label_visibility="collapsed")

with col2:
    if st.button("Translate"):
        if user_input.strip() != "":
            st.session_state.messages.append({"role": "user", "content": user_input})
            translation = translate_to_french(user_input)
            st.session_state.messages.append({"role": "bot", "content": translation})
            st.rerun()
