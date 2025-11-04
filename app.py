# app.py
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import Tuple
import html
import io
import time

# ---------------- Page config (must be first Streamlit call) ----------------
st.set_page_config(page_title="English ↔ French Translator", layout="wide", initial_sidebar_state="expanded")

# ---------------- Sidebar / Settings ----------------
st.sidebar.title("Settings")
model_name = st.sidebar.text_input("Hugging Face model name", value="Helsinki-NLP/opus-mt-en-fr",
                                  help="Change to another Marian model if you like (e.g. opus-mt-en-de).")
device_opt = st.sidebar.selectbox("Device", options=["cpu", "cuda" if torch.cuda.is_available() else "cpu"])
max_input_chars = st.sidebar.number_input("Max input characters", min_value=100, max_value=20000, value=4000, step=100)
show_model_info = st.sidebar.checkbox("Show model info after load", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Features:\n- Chat history\n- Clear and Download\n- Copy translation button")

# ----------------- Model loading (cached) -----------------
@st.cache_resource
def load_model_and_tokenizer(name: str, device: str) -> Tuple[MarianMTModel, MarianTokenizer]:
    tokenizer = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    return model, tokenizer

# Show spinner while loading
with st.spinner(f"Loading model {model_name} ({device_opt}) ... this may take a while"):
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, device_opt)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

if show_model_info:
    st.sidebar.success(f"Model loaded: {model_name} on {device_opt}")

# ----------------- CSS -----------------
st.markdown("""
    <style>
    body { background-color: #F5F7FA; }
    .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        max-width: 900px;
        margin: 16px auto;
        height: calc(100vh - 220px);
        overflow-y: auto;
        padding: 12px;
        margin-bottom: 140px;
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
        word-wrap: break-word;
    }
    .bot-bubble {
        background-color: #E5E5EA;
        color: black;
        padding: 10px 16px;
        border-radius: 18px;
        margin: 6px;
        align-self: flex-start;
        max-width: 80%;
        word-wrap: break-word;
    }
    .meta {
        font-size: 12px;
        opacity: 0.7;
        margin-top: 4px;
    }
    .input-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 12px 20px;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.08);
        z-index: 1000;
    }
    @media (max-width: 600px) {
        .user-bubble, .bot-bubble {
            max-width: 95%;
        }
        .chat-container {
            max-width: 98%;
            padding-left: 8px;
            padding-right: 8px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Session state for messages ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # each msg: {"role": "user"|"bot", "content": str, "time": float}

# ---------------- Helper: translation ----------------
def translate_text(text: str, tokenizer: MarianTokenizer, model: MarianMTModel, device: str, max_len=512) -> str:
    # basic safety/length handling
    if len(text) == 0:
        return ""
    # truncate if necessary to avoid OOM
    # (we count characters here; better heuristics possible)
    if len(text) > max_input_chars:
        text = text[:max_input_chars]  # truncate
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    if device == "cuda" and torch.cuda.is_available():
        tokens = {k: v.to("cuda") for k, v in tokens.items()}
    outputs = model.generate(**tokens, max_length=2 * tokens["input_ids"].shape[-1] + 50, num_beams=5, early_stopping=True)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

# ---------------- Main layout ----------------
st.title("🇬🇧 ↔ 🇫🇷 Translation Chat")
st.write("Type English text and get a French translation. Uses a Marian model locally.")

# Chat area (render messages)
st.markdown("<div class='chat-container' id='chat-container'>", unsafe_allow_html=True)
for i, msg in enumerate(st.session_state.messages):
    safe_text = html.escape(msg["content"]).replace("\n", "<br>")
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{safe_text}</div>", unsafe_allow_html=True)
    else:
        # add a small copy-button and timestamp displayed under the bubble via meta text
        st.markdown(f"<div class='bot-bubble'>{safe_text}</div>", unsafe_allow_html=True)
        # Show copy button separately (Streamlit widget) so user can click to copy translation
        # We give each button a unique key
        copy_key = f"copy_{i}"
        st.button("Copy translation", key=copy_key, on_click=st.clipboard_set if hasattr(st, "clipboard_set") else None)
        # (Note: st.clipboard_set may not be available in all Streamlit versions;
        #  the button could instead place text into the text_input, or the user can select text)
st.markdown("<div id='bottom-anchor'></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Auto-scroll JavaScript - ensures chat always shows latest message
st.markdown("""
    <script>
        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Re-run after Streamlit finishes rendering
        window.addEventListener('load', () => {
            const element = document.getElementById('bottom-anchor');
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        });
    </script>
""", unsafe_allow_html=True)


# ---------------- Input bar (fixed) ----------------
with st.form("input_form", clear_on_submit=True):
    cols = st.columns([8, 1, 1])
    with cols[0]:
        user_text = st.text_area("Type your English text here:", height=80, placeholder="Type a message…", key="user_input_field")
        char_count = len(user_text or "")
        st.caption(f"Characters: {char_count}/{int(max_input_chars)}")
    with cols[1]:
        submit = st.form_submit_button("Translate")
    with cols[2]:
        clear = st.form_submit_button("Clear chat")

# Buttons under input for download / extra controls
download_buffer = None
if st.session_state.messages:
    # prepare text for download
    buf = io.StringIO()
    for m in st.session_state.messages:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("time", time.time())))
        role = "User" if m["role"] == "user" else "Bot"
        buf.write(f"[{t}] {role}: {m['content']}\n\n")
    download_buffer = buf.getvalue()

col_dl1, col_dl2 = st.columns([1, 1])
with col_dl1:
    if download_buffer:
        st.download_button("Download chat", data=download_buffer, file_name="chat_history.txt", mime="text/plain")
with col_dl2:
    if st.button("Clear local chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Handle submission & translation ----------------
if submit:
    if (user_text or "").strip() == "":
        st.warning("Please type something to translate.")
    else:
        # add user message
        st.session_state.messages.append({"role": "user", "content": user_text.strip(), "time": time.time()})
        # show immediate rerun so UI shows user bubble quickly
        st.rerun()

# When a new user message exists and no bot reply yet, do translation
# (We check last two messages to detect a user message without a bot response.)
if st.session_state.messages:
    last = st.session_state.messages[-1]
    need_reply = (last["role"] == "user") and (len(st.session_state.messages) == 1 or st.session_state.messages[-1]["role"] == "user")
    # Another safe check: if last user message doesn't already have a bot reply next
    if last["role"] == "user":
        # Translate synchronously (blocking)
        try:
            with st.spinner("Translating..."):
                translation = translate_text(last["content"], tokenizer, model, device_opt)
        except Exception as e:
            st.error(f"Translation failed: {e}")
            translation = "⚠️ Error during translation."
        st.session_state.messages.append({"role": "bot", "content": translation, "time": time.time()})
        # rerun to show bot message (this will trigger auto-scroll)
        st.rerun()
