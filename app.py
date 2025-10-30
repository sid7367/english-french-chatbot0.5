import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Load Model ---
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- Streamlit Config ---
st.set_page_config(page_title="Chat Translator 💬", page_icon="🌍", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    body { background-color: #F5F7FA; }

    .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        max-width: 600px;
        margin: auto;
        height: calc(100vh - 250px); /* Fixed height for scrolling */
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
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 English → French Chat Translator")

# --- Initialize Session ---
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

# Auto-scroll JavaScript
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

    # Typing animation (inside chat container)
    st.markdown("<div class='bot-bubble'>", unsafe_allow_html=True)
    display_text = ""
    placeholder = st.empty()
    for ch in translated_text:
        display_text += ch
        placeholder.markdown(f"<div class='bot-bubble'>{display_text}</div>", unsafe_allow_html=True)
        time.sleep(0.03)
    st.markdown("</div>", unsafe_allow_html=True)
    placeholder.empty()

    # Save AI message
    st.session_state.messages.append({"role": "bot", "content": translated_text})
    st.rerun()
