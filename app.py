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
st.set_page_config(
    page_title="English ↔ French Translator | AI-Powered Translation",
    page_icon="🇫🇷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# English-French Translation Chatbot\nPowered by Helsinki-NLP Marian MT model"
    }
)

# ---------------- Sidebar / Settings ----------------
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 🤖 Model Configuration")
model_name = st.sidebar.text_input(
    "Hugging Face model name", 
    value="Helsinki-NLP/opus-mt-en-fr",
    help="Change to another Marian model if you like (e.g. opus-mt-en-de)."
)

device_opt = st.sidebar.selectbox(
    "Computing Device", 
    options=["cpu", "cuda" if torch.cuda.is_available() else "cpu"],
    help="Select CPU or GPU (CUDA) for processing"
)

max_input_chars = st.sidebar.number_input(
    "Max input characters", 
    min_value=100, 
    max_value=20000, 
    value=4000, 
    step=100,
    help="Maximum number of characters allowed in a single translation"
)

show_model_info = st.sidebar.checkbox("Show model info after load", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ✨ Features")
st.sidebar.markdown("""
- 💬 **Chat history** - Keep track of all translations
- 📋 **Copy button** - Easily copy translations
- 💾 **Download** - Save your conversation
- 🗑️ **Clear** - Start fresh anytime
- 📱 **Responsive** - Works on mobile devices
""")

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
with st.spinner(f"🔄 Loading model {model_name} on {device_opt}... This may take a moment."):
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, device_opt)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

if show_model_info:
    st.sidebar.success(f"✅ Model loaded successfully!\n\n**Model:** {model_name}\n\n**Device:** {device_opt.upper()}")

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
    /* Global styles */
    body { 
        background-color: #F5F7FA; 
    }
    
    /* Hide Streamlit default elements for cleaner UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat container with proper scrolling */
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
        margin: 0 auto;
        padding: 16px;
        padding-bottom: 20px;
        min-height: 400px;
        gap: 8px;
    }
    
    /* User message bubble - aligned to right */
    .user-bubble {
        background: linear-gradient(135deg, #0078FF 0%, #0063D1 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 4px 20px;
        margin: 4px 0;
        margin-left: auto;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0, 120, 255, 0.2);
        font-size: 15px;
        line-height: 1.5;
    }
    
    /* Bot message bubble - aligned to left */
    .bot-bubble {
        background-color: #E5E5EA;
        color: #000000;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 4px;
        margin: 4px 0;
        margin-right: auto;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        font-size: 15px;
        line-height: 1.5;
    }
    
    /* Copy button styling */
    .copy-btn {
        margin-top: 6px;
        margin-left: 0;
        padding: 6px 14px;
        border: none;
        border-radius: 10px;
        background-color: #0078FF;
        color: white;
        cursor: pointer;
        font-size: 13px;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 120, 255, 0.2);
    }
    
    .copy-btn:hover {
        background-color: #0063D1;
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0, 120, 255, 0.3);
    }
    
    .copy-btn:active {
        transform: translateY(0);
    }
    
    /* Message wrapper for proper alignment */
    .message-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 8px;
        width: 100%;
    }
    
    .user-message-wrapper {
        align-items: flex-end;
    }
    
    /* Input area improvements */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #E5E5EA;
        padding: 12px;
        font-size: 15px;
    }
    
    .stTextArea textarea:focus {
        border-color: #0078FF;
        box-shadow: 0 0 0 2px rgba(0, 120, 255, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Download button styling */
    .stDownloadButton button {
        border-radius: 12px;
        padding: 10px 20px;
    }
    
    /* Character counter */
    .stCaption {
        font-size: 13px;
        color: #666;
        margin-top: 4px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-bubble, .bot-bubble {
            max-width: 85%;
            font-size: 14px;
            padding: 10px 14px;
        }
        .chat-container {
            max-width: 98%;
            padding-left: 10px;
            padding-right: 10px;
            padding: 12px 8px;
        }
    }
    
    @media (max-width: 480px) {
        .user-bubble, .bot-bubble {
            max-width: 90%;
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
st.title("🇬🇧 ↔ 🇫🇷 English-French Translation Chat")
st.markdown("""
    <p style='font-size: 18px; color: #666; margin-bottom: 24px;'>
        ✨ Type English text and get instant French translation powered by AI
    </p>
""", unsafe_allow_html=True)

# Show helpful tip if no messages yet
if not st.session_state.messages:
    st.info("👋 Welcome! Type your English text below and click 'Translate' to get started.", icon="💡")

# Chat area (render messages)
st.markdown("<div class='chat-container' id='chat-container'>", unsafe_allow_html=True)
for i, msg in enumerate(st.session_state.messages):
    safe_text = html.escape(msg["content"]).replace("\n", "<br>")

    if msg["role"] == "user":
        # User message aligned to the right
        st.markdown(f"""
            <div class="message-wrapper user-message-wrapper">
                <div class='user-bubble'>{safe_text}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Bot message with copy button, aligned to the left
        components.html(f"""
            <div class="message-wrapper" style="display:flex; flex-direction:column; align-items:flex-start; width:100%; margin-bottom:12px;">
                <div id="bot_msg_{i}" style="
                    background-color:#E5E5EA;
                    color:#000000;
                    padding:12px 18px;
                    border-radius:20px 20px 20px 4px;
                    max-width:75%;
                    word-wrap:break-word;
                    box-shadow:0 2px 4px rgba(0,0,0,0.08);
                    font-size:15px;
                    line-height:1.5;
                    margin:4px 0;
                ">{safe_text}</div>
                <button id="copy_btn_{i}" class="copy-btn" style="
                    margin-top:6px;
                    padding:6px 14px;
                    border:none;
                    border-radius:10px;
                    background-color:#0078FF;
                    color:white;
                    cursor:pointer;
                    font-size:13px;
                    font-weight:500;
                    transition:all 0.2s ease;
                    box-shadow:0 2px 4px rgba(0,120,255,0.2);
                " onmouseover="this.style.backgroundColor='#0063D1'; this.style.transform='translateY(-1px)'" 
                   onmouseout="this.style.backgroundColor='#0078FF'; this.style.transform='translateY(0)'">
                    📋 Copy translation
                </button>
            </div>
            <script>
                const btn = document.getElementById("copy_btn_{i}");
                const textElem = document.getElementById("bot_msg_{i}");
                if (btn && textElem) {{
                    btn.onclick = async () => {{
                        try {{
                            const text = textElem.innerText || textElem.textContent;
                            await navigator.clipboard.writeText(text);
                            btn.innerHTML = "✓ Copied!";
                            btn.style.backgroundColor = "#28a745";
                            setTimeout(() => {{
                                btn.innerHTML = "📋 Copy translation";
                                btn.style.backgroundColor = "#0078FF";
                            }}, 2000);
                        }} catch (err) {{
                            btn.innerHTML = "✗ Failed";
                            btn.style.backgroundColor = "#dc3545";
                            setTimeout(() => {{
                                btn.innerHTML = "📋 Copy translation";
                                btn.style.backgroundColor = "#0078FF";
                            }}, 2000);
                        }}
                    }};
                }}
            </script>
        """, height=110)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Input bar ----------------
st.markdown("---")
st.subheader("💬 Enter your message")

with st.form("input_form", clear_on_submit=True):
    user_text = st.text_area(
        "Type your English text here:", 
        height=100, 
        placeholder="Type your message in English...", 
        key="user_input_field",
        label_visibility="collapsed"
    )
    char_count = len(user_text or "")
    st.caption(f"✏️ Characters: {char_count}/{int(max_input_chars)}")
    
    # Button row with better spacing
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit = st.form_submit_button("🚀 Translate", use_container_width=True, type="primary")
    with col2:
        clear = st.form_submit_button("🗑️ Clear chat", use_container_width=True)
    with col3:
        pass  # Empty column for spacing

# Additional controls with better layout
if st.session_state.messages:
    st.markdown("---")
    st.subheader("📥 Download & Manage")
    
    # prepare text for download
    buf = io.StringIO()
    for m in st.session_state.messages:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("time", time.time())))
        role = "User" if m["role"] == "user" else "Bot"
        buf.write(f"[{t}] {role}: {m['content']}\n\n")
    download_buffer = buf.getvalue()
    
    col_dl1, col_dl2, col_dl3 = st.columns([2, 2, 2])
    with col_dl1:
        st.download_button(
            "💾 Download chat history", 
            data=download_buffer, 
            file_name="translation_history.txt", 
            mime="text/plain",
            use_container_width=True
        )
    with col_dl2:
        if st.button("🗑️ Clear all messages", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_dl3:
        st.metric("Messages", len(st.session_state.messages))

# ---------------- Handle submission & translation ----------------
if submit:
    if (user_text or "").strip() == "":
        st.warning("⚠️ Please type something to translate.", icon="⚠️")
    else:
        # add user message
        st.session_state.messages.append({"role": "user", "content": user_text.strip(), "time": time.time()})
        # show immediate rerun so UI shows user bubble quickly
        st.rerun()

# When a new user message exists and no bot reply yet, do translation
if st.session_state.messages:
    last = st.session_state.messages[-1]
    # Check if last message is from user and needs a reply
    if last["role"] == "user":
        # Translate synchronously (blocking)
        try:
            with st.spinner("🔄 Translating your message..."):
                translation = translate_text(last["content"], tokenizer, model, device_opt)
        except Exception as e:
            st.error(f"❌ Translation failed: {e}", icon="❌")
            translation = "⚠️ Error during translation. Please try again."
        st.session_state.messages.append({"role": "bot", "content": translation, "time": time.time()})
        # rerun to show bot message
        st.rerun()

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px 0;'>
        <p style='font-size: 14px;'>
            <strong>English-French Translation Chatbot</strong> | 
            Powered by <a href='https://huggingface.co/Helsinki-NLP/opus-mt-en-fr' target='_blank'>Helsinki-NLP Marian MT</a> | 
            Built with ❤️ using Streamlit
        </p>
        <p style='font-size: 12px; color: #999;'>
            💡 Tip: The model processes text locally for privacy. Longer texts may take more time to translate.
        </p>
    </div>
""", unsafe_allow_html=True)

