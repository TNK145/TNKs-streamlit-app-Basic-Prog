import streamlit as st
import pandas as pd
import re
import json

# Safe imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ===============================
# Helpers
# ===============================

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def build_prompt(sentences):
    """
    Prompt instructing LLM to return ONLY raw JSON.
    """
    return f"""
You are a Russian linguistics expert.

For EACH of the following Russian sentences, analyze the grammar cases used.

Return **ONLY raw JSON** — DO NOT include markdown, code fences, or explanations.

The JSON should be a list of objects, each with these keys:
- "sentence": original sentence
- "case": Russian grammatical case(s)
- "altered_words": list of words whose form was changed
- "original_forms": list of base/dictionary forms

Sentences:
{sentences}
"""

def clean_model_json(text):
    """
    Remove code fences, markdown, and leading/trailing spaces.
    """
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()

def parse_json_output(text):
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except:
        return None

def batch_sentences(sentences, batch_size):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i:i+batch_size]

def get_valid_gemini_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        valid = []
        for m in models:
            methods = getattr(m, "supported_generation_methods", [])
            if "generateContent" in methods:
                valid.append(m.name)
        return valid
    except Exception:
        return []

def call_model(provider, api_key, prompt, fallback_model="gpt-4.1-mini"):
    """
    Returns (text, status)
    Status can be: "OK", "NO_GEMINI_MODELS", "GEMINI_ERROR", "OPENAI_ERROR:..."
    """
    if provider == "Gemini":
        try:
            genai.configure(api_key=api_key)
            valid_models = get_valid_gemini_models(api_key)
            if not valid_models:
                return None, "NO_GEMINI_MODELS"
            model_name = valid_models[0]
            model = genai.GenerativeModel(model_name)
            out = model.generate_content(prompt)
            return out.text, "OK"
        except Exception:
            return None, "GEMINI_ERROR"
    # OpenAI path
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=fallback_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content, "OK"
    except Exception as e:
        return None, f"OPENAI_ERROR: {e}"

# ===============================
# Streamlit UI
# ===============================

st.set_page_config(page_title="Russian Grammar Case Analyzer", layout="wide")
st.title("🇷🇺 Russian Grammar Case Analyzer (Gemini + OpenAI Fallback)")

st.write("Analyze Russian text for grammatical cases, altered words, and base forms.")

# Sidebar
st.sidebar.header("API Settings")
provider = st.sidebar.selectbox("Preferred Provider", ["Gemini", "OpenAI"])
api_key = st.sidebar.text_input(f"Enter your {provider} API key:", type="password")
batch_size = st.sidebar.number_input("Batch size (sentences/request):", 1, 20, 5)

if not api_key:
    st.warning("Please enter your API key.")
    st.stop()

text = st.text_area("Enter Russian text:", height=200, placeholder="Paste Russian text here...")

if st.button("Start Analysis"):
    if not text.strip():
        st.error("Please enter some text.")
        st.stop()

    sentences = split_into_sentences(text)
    st.write(f"Detected **{len(sentences)}** sentences.")
    results = []

    for batch in batch_sentences(sentences, batch_size):
        prompt = build_prompt(batch)
        output, status = call_model(provider, api_key, prompt, fallback_model="gpt-4.1-mini")

        if status == "NO_GEMINI_MODELS":
            st.warning("⚠ Gemini has no text-generation models. Falling back to OpenAI.")
            output, status = call_model("OpenAI", api_key, prompt)

        elif status == "GEMINI_ERROR":
            st.warning("⚠ Gemini failed. Falling back to OpenAI.")
            output, status = call_model("OpenAI", api_key, prompt)

        if status.startswith("OPENAI_ERROR"):
            st.error(f"❌ OpenAI failed. Error: {status}")
            st.stop()

        cleaned = clean_model_json(output)
        data = parse_json_output(cleaned)
        if data is None:
            st.error("Model returned invalid JSON:")
            st.code(output)
            st.stop()

        results.extend(data)

    df = pd.DataFrame(results)
    required = ["sentence", "case", "altered_words", "original_forms"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    st.subheader("📊 Analysis Result")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "russian_case_analysis.csv",
        "text/csv"
    )
