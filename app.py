# app.py
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import Tuple
import html
import io
import time
import streamlit.components.v1 as components

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
# ----------------- Modern Visual Theme -----------------
st.markdown("""
    <style>
    /* Global background */
    body {
        background-color: #F5F7FA;
        font-family: "Segoe UI", Roboto, sans-serif;
    }

    /* Header bar */
    .header-bar {
        background: linear-gradient(90deg, #0078FF, #00B4FF);
        color: white;
        text-align: center;
        padding: 14px;
        font-size: 22px;
        font-weight: 600;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        letter-spacing: 0.4px;
    }

    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        max-width: 900px;
        margin: 16px auto;
        height: calc(100vh - 250px);
        overflow-y: auto;
        padding: 16px;
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        margin-bottom: 160px;
    }

    /* User message bubble */
    .user-bubble {
        background: linear-gradient(135deg, #0078FF, #00B4FF);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px;
        text-align: right;
        align-self: flex-end;
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.5;
    }

    /* Bot message bubble */
    .bot-bubble {
        background-color: #EAEAEA;
        color: black;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px;
        align-self: flex-start;
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.5;
    }
    
    /* Input bar */
    .input-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #ffffff;
        padding: 14px 24px;
        box-shadow: 0 -2px 15px rgba(0,0,0,0.07);
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 1000;
    }

    /* Input field */
    textarea {
        flex: 1;
        border: 1px solid #d0d0d0 !important;
        border-radius: 24px !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
        resize: none !important;
        transition: all 0.2s ease;
        box-shadow: none !important;
    }
    textarea:focus {
        border-color: #0078FF !important;
        box-shadow: 0 0 4px rgba(0,120,255,0.3) !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #0078FF !important;
        border: none !important;
        border-radius: 20px !important;
        color: white !important;
        font-weight: 500 !important;
        transition: background 0.2s ease;
    }
    button[kind="primary"]:hover {
        background-color: #005ec7 !important;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background-color: #F0F4FA !important;
        border-right: 1px solid #E0E0E0 !important;
    }

    /* Make scroll smooth */
    .chat-container {
        scroll-behavior: smooth;
    }

    @media (max-width: 600px) {
        .user-bubble, .bot-bubble {
            max-width: 95%;
        }
        .chat-container {
            max-width: 98%;
            padding-left: 10px;
            padding-right: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Add custom header bar (before title)
st.markdown("<div class='header-bar' >🇬🇧 English → 🇫🇷 French Translator</div>", unsafe_allow_html=True)


# st.markdown("""
#     <style>
#     body { background-color: #F5F7FA; }
#     .chat-container {
#         display: flex;
#         flex-direction: column;
#         width: 100%;
#         max-width: 900px;
#         margin: 16px auto;
#         height: calc(100vh - 220px);
#         overflow-y: auto;
#         padding: 12px;
#         margin-bottom: 140px;
#     }
#     .user-bubble {
#         background-color: #0078FF;
#         color: white;
#         padding: 10px 16px;
#         border-radius: 18px;
#         margin: 6px;
#         text-align: right;
#         align-self: flex-end;
#         max-width: 80%;
#         word-wrap: break-word;
#     }
#     .bot-bubble {
#         background-color: #E5E5EA;
#         color: black;
#         padding: 10px 16px;
#         border-radius: 18px;
#         margin: 6px;
#         align-self: flex-start;
#         max-width: 80%;
#         word-wrap: break-word;
#     }
#     .meta {
#         font-size: 12px;
#         opacity: 0.7;
#         margin-top: 4px;
#     }
#     .input-bar {
#         position: fixed;
#         bottom: 0;
#         left: 0;
#         right: 0;
#         background-color: white;
#         padding: 12px 20px;
#         box-shadow: 0 -2px 6px rgba(0,0,0,0.08);
#         z-index: 1000;
#     }
#     @media (max-width: 600px) {
#         .user-bubble, .bot-bubble {
#             max-width: 95%;
#         }
#         .chat-container {
#             max-width: 98%;
#             padding-left: 8px;
#             padding-right: 8px;
#         }
#     }
#     </style>
# """, unsafe_allow_html=True)

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
        # Use Streamlit components to safely inject HTML + JS for the copy button
        components.html(f"""
            <div style="display:flex; flex-direction:column; align-items:flex-start; margin-bottom:8px;">
                <div id="bot_msg_{i}" class="bot-bubble" style="
                    background-color:#E5E5EA;
                    color:black;
                    padding:10px 16px;
                    border-radius:18px;
                    max-width:80%;
                    word-wrap:break-word;
                ">{safe_text}</div>
                <button id="copy_btn_{i}" style="
                    margin-top:4px;
                    padding:6px 12px;
                    border:none;
                    border-radius:8px;
                    background-color:#0078FF;
                    color:white;
                    cursor:pointer;
                ">Copy translation</button>
            </div>
            <script>
                const btn = document.getElementById("copy_btn_{i}");
                const textElem = document.getElementById("bot_msg_{i}");
                if (btn && textElem) {{
                    btn.onclick = async () => {{
                        try {{
                            await navigator.clipboard.writeText(textElem.innerText);
                            btn.innerText = "Copied!";
                            setTimeout(() => btn.innerText = "Copy translation", 2000);
                        }} catch (err) {{
                            btn.innerText = "Failed!";
                        }}
                    }};
                }}
            </script>
        """, height=90)

st.markdown("<div id='bottom-anchor'></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Auto-scroll JavaScript - ensures chat always shows latest message
# ---------------- Auto-scroll (improved) ----------------
st.markdown("""
    <script>
        function scrollToBottom() {
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        // Delay scroll slightly after DOM updates
        window.addEventListener('load', () => {
            setTimeout(scrollToBottom, 300);
        });

        // Also scroll when Streamlit updates DOM dynamically
        new MutationObserver(() => setTimeout(scrollToBottom, 150))
            .observe(document.body, { childList: true, subtree: true });
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


