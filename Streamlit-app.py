import streamlit as st
import os

# Try importing google-generativeai
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ModuleNotFoundError:
    GEMINI_AVAILABLE = False

CACHE_FILE = ".gemini_api_key"

st.set_page_config(page_title="Gemini API Key Demo", layout="centered")
st.title("🔑 Gemini API Key Demo with Key Caching")


# --- Load cached key if it exists ---
cached_key = None
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            cached_key = f.read().strip()
    except Exception:
        cached_key = None


# --- Sidebar ---
st.sidebar.header("API Settings")

# Checkbox to choose caching
cache_choice = st.sidebar.checkbox("Remember my API key")

# API key input: pre-fill if cached
gemini_api_key = st.sidebar.text_input(
    "Enter your Gemini API key:",
    type="password",
    value=cached_key if cached_key else ""
)

# Store key in session_state
if gemini_api_key:
    st.session_state["GEMINI_API_KEY"] = gemini_api_key


# --- Caching logic ---
def save_cached_key(key):
    with open(CACHE_FILE, "w") as f:
        f.write(key)

def delete_cached_key():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)


# If user enables caching, save key
if cache_choice and gemini_api_key:
    save_cached_key(gemini_api_key)
elif not cache_choice:
    delete_cached_key()


# --- Dependency warning ---
if not GEMINI_AVAILABLE:
    st.error(
        "The `google-generativeai` library is not installed.\n\n"
        "Install it with:\n"
        "```\npip install google-generativeai\n```"
    )
else:
    # Validate key button
    if st.sidebar.button("Test API Key"):
        if not gemini_api_key:
            st.sidebar.error("Please enter a Gemini API key.")
        else:
            try:
                genai.configure(api_key=gemini_api_key)
                models = genai.list_models()
                st.sidebar.success("API key is valid! ✔️")
            except Exception as e:
                st.sidebar.error(f"Invalid API key or error: {e}")


# --- Main content ---
if "GEMINI_API_KEY" in st.session_state:
    st.success("Gemini API key is loaded and ready to use.")
    if cache_choice:
        st.info("Your API key is cached locally and will auto-load next time.")
    else:
        st.info("Caching is disabled. Your key will be forgotten on reload.")
else:
    st.info("Please enter your Gemini API key in the sidebar.")
