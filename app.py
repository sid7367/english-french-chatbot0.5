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
    body { background-color: #F5F7FA; }

    .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        max-width: 600px;
        margin: auto;
        height: calc(100vh - 250px); /* Fixed height: 250px accounts for header, title, input bar */
        overflow-y: auto; /* Enable vertical scrolling */
        padding: 10px;
        margin-bottom: 120px; /* space for bottom bar */
    }

    .user-bubble {
        background-color: #0078FF;
        color: white;
        padding: 10px 16px;
        border-radius: 18px;
        margin: 6px;
        text-align: right;
        align-self: flex-end;
        max-width: 80%;
    }

    .bot-bubble {
        background-color: #E5E5EA;
        color: black;
        padding: 10px 16px;
        border-radius: 18px;
        margin: 6px;
        align-self: flex-start;
        max-width: 80%;
    }

    .input-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 1000;
    }

    @media (max-width: 600px) {
        .user-bubble, .bot-bubble {
            max-width: 95%;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Chat Session ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat Messages ---
st.markdown("<div class='chat-container' id='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

# Add an anchor at the bottom for auto-scroll
st.markdown("<div id='bottom-anchor'></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Auto-scroll JavaScript - triggers on each page rerender to show latest messages
st.markdown("""
    <script>
        setTimeout(function() {
            var element = document.getElementById('bottom-anchor');
            if (element) {
                element.scrollIntoView({behavior: 'smooth', block: 'end'});
            }
        }, 100);
    </script>
""", unsafe_allow_html=True)

# --- Input Bar Fixed at Bottom ---
with st.container():
    st.markdown("<div class='input-bar'>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your English text:", key="input", label_visibility="collapsed", placeholder="Type a message…")
        submitted = st.form_submit_button("Send")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Handle Submission ---
if submitted and user_input.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Translate
    inputs = tokenizer(user_input, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

with col1:
    user_input = st.text_input("Type your message in English:", key="input", label_visibility="collapsed")

with col2:
    if st.button("Translate"):
        if user_input.strip() != "":
            st.session_state.messages.append({"role": "user", "content": user_input})
            translation = translate_to_french(user_input)
            st.session_state.messages.append({"role": "bot", "content": translation})
            st.rerun()
