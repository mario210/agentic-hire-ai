import streamlit as st
from loguru import logger
import os
import tempfile
import base64
import re
import functools
import html

# --- BACKEND IMPORTS ---
from main import _prepare_cv_data, _initialize_state, _run_graph
from src.config.settings import config
from src.agents.agents import get_agent_factory
from src.graph import build_graph


# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Orbitron AI")


# --- STATE ---
if "running" not in st.session_state:
    st.session_state.running = False

if "logs" not in st.session_state:
    st.session_state.logs = []

if "final_state" not in st.session_state:
    st.session_state.final_state = None

if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False


# --- BASE64 IMAGE ---
@functools.lru_cache(maxsize=10)
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# --- CSS ---
def inject_layout_css(img_base64):
    st.markdown(f"""
    <style>

    .bg-img-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
    }}

    .bg-img-container img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: brightness(0.6);
    }}

    .stApp {{
        background: transparent;
        color: #C0D6E4;
        font-family: monospace;
    }}

    .glass-panel {{
        background: rgba(10, 31, 51, 0.75);
        backdrop-filter: blur(12px);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid rgba(29, 233, 182, 0.3);
        margin-top: 20px;
    }}

    /* ===== TERMINAL ===== */

    .neural-terminal {{
        height: 600px;
        overflow-y: auto;
        background: rgba(8, 24, 40, 0.95);
        border: 1px solid #1de9b6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(29, 233, 182, 0.3);
    }}

    .msg {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 10px;
        opacity: 0;
        animation: fadeIn 0.4s ease forwards;
    }}

    .msg img {{
        width: 28px;
        height: 28px;
    }}

    .msg-text {{
        font-size: 0.8rem;
        line-height: 1.4;
    }}

    .system {{
        color: #9aa7b0;
        font-style: italic;
    }}

    @keyframes fadeIn {{
        to {{
            opacity: 1;
        }}
    }}

    button {{
        border: 1px solid #1de9b6 !important;
        background: transparent !important;
        color: #1de9b6 !important;
    }}

    </style>

    <div class="bg-img-container">
        <img src="data:image/png;base64,{img_base64}">
    </div>
    """, unsafe_allow_html=True)


# --- TERMINAL RENDERER ---
def render_terminal(placeholder, logs):
    """Zero-gap terminal using pure HTML/CSS for log rows."""
    with placeholder.container():
        with st.container(height=600, border=True):
            st.markdown(
                """
                <style>
                .terminal-row {
                    display: flex;
                    align-items: flex-start;
                    gap: 10px; /* Adjust this to 0px if you want them touching */
                    margin-bottom: 15px;
                    width: 100%;
                }

                .terminal-avatar {
                    flex-shrink: 0;
                    border: 1px solid rgba(29, 233, 182, 0.3);
                    border-radius: 4px;
                }

                .log-entry {
                    background: rgba(29, 233, 182, 0.07);
                    border-left: 4px solid #1de9b6;
                    padding: 12px;
                    border-radius: 0px 8px 8px 0px;
                    flex-grow: 1;
                    min-height: 100px;
                }

                .prefix { 
                    color: #1de9b6; 
                    font-weight: bold; 
                    font-family: 'Courier New', monospace; 
                    font-size: 0.9rem;
                    margin-bottom: 4px;
                }

                .msg-content { 
                    color: #C0D6E4; 
                    font-family: monospace; 
                    font-size: 0.95rem;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            if not logs:
                st.write("SYSTEM_IDLE: Awaiting Uplink...")
                return

            for agent, text, img in reversed(logs[-50:]):
                # Convert image to base64 to use inside raw HTML
                img_html = ""
                if img and os.path.exists(img):
                    with open(img, "rb") as f:
                        data = base64.b64encode(f.read()).decode()
                        img_html = f'<img src="data:image/png;base64,{data}" width="100" class="terminal-avatar">'
                else:
                    img_html = '<div style="width:100px; text-align:center; font-size:40px;">⚙️</div>'

                prefix = {
                    "scout": "[SCOUT]",
                    "tailor": "[TAILOR]",
                    "orchestrator": "[ORCHESTRATOR]",
                    "system": "⚙ [SYS_MSG]"
                }.get(agent, "LOG")

                # Single markdown call for the whole row ensures no Streamlit column padding
                st.markdown(
                    f"""
                    <div class="terminal-row">
                        {img_html}
                        <div class="log-entry">
                            <div class="prefix">{prefix}</div>
                            <div class="msg-content">{text}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# --- LOG SINK ---
class StreamlitLogSink:
    def __init__(self, placeholder, log_buffer):
        self.placeholder = placeholder
        self.log_buffer = log_buffer
        self.handler_id = None

    def write(self, message):
        clean = message.strip()

        if "[SCOUT]" in clean:
            agent = "scout"
            img = "ui/images/a.png"
        elif "[TAILOR]" in clean:
            agent = "tailor"
            img = "ui/images/b.png"
        elif "[ORCHESTRATOR]" in clean:
            agent = "orchestrator"
            img = "ui/images/c.png"
        else:
            agent = "system"
            img = None

        # ❌ NO HTML processing anymore
        self.log_buffer.append((agent, clean, img))

        render_terminal(self.placeholder, self.log_buffer)

    def __enter__(self):
        self.handler_id = logger.add(
            self,
            format="{time:HH:mm:ss} | {message}",
            level="INFO"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler_id:
            logger.remove(self.handler_id)


# --- MAIN APP ---
def streamlit_app():

    img_path = "ui/images/bg.png"
    if os.path.exists(img_path):
        inject_layout_css(get_base64_image(img_path))

    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.title("AGENTIC HIRE AI")
        st.caption("HEADHUNTER PROTOCOL v1.0")
        st.write("---")

        uploaded_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])

        criteria = st.text_area(
            "Search Parameters:",
            height=120,
            value="Python Developer or AI Engineer roles"
        )

        c1, c2 = st.columns(2)
        start = c1.button("INITIALIZE", use_container_width=True)
        stop = c2.button("ABORT", use_container_width=True)

    with col_right:
        status_placeholder = st.empty()
        terminal_placeholder = st.empty()

        if stop:
            st.session_state.cancel_requested = True
            st.session_state.running = False
            st.session_state.logs.append(("system", "[SYSTEM] Aborted by user.", None))
            status_placeholder.warning("Process stopped.")
            st.stop()

        if start:
            if not uploaded_file:
                status_placeholder.error("Upload CV first.")
                return

            st.session_state.running = True
            st.session_state.logs = []
            st.session_state.final_state = None
            st.session_state.cancel_requested = False

            st.rerun()

        if st.session_state.running:
            cv_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    cv_path = tmp.name

                with StreamlitLogSink(terminal_placeholder, st.session_state.logs):
                    status_placeholder.info("🧠 Agents are running...")

                    logger.info("[SYSTEM] Preparing CV...")
                    cv_manager = _prepare_cv_data(cv_path, get_agent_factory())

                    logger.info("[TAILOR] Understanding candidate profile...")
                    state = _initialize_state(cv_manager, config)

                    logger.info("[SCOUT] Exploring job market...")
                    final_state = _run_graph(state, build_graph())

                    logger.info("[ORCHESTRATOR] Finalizing results...")

                    st.session_state.final_state = final_state
                    status_placeholder.success("✅ Done")

            except Exception as e:
                logger.error(f"[ERROR] {e}")
                status_placeholder.error(str(e))

            finally:
                if cv_path and os.path.exists(cv_path):
                    os.remove(cv_path)

                st.session_state.running = False

        # This ensures the terminal stays populated with the last logs after a run
        # or shows the idle message on first load.
        else:
            render_terminal(terminal_placeholder, st.session_state.logs)

        if st.session_state.final_state:
            st.markdown("## 🎯 Results")
            apps = st.session_state.final_state.get("applications", {})

            for _, content in apps.items():
                st.markdown(f"""
                <div class="glass-panel">
                    <h3>{content.get('job_title')}</h3>
                    <p>{content.get('company')}</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    streamlit_app()