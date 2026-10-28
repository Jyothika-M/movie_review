"""
CineSense — Movie Review Sentiment Analyzer
Streamlit Web Application

PCCSL 508 – Machine Learning Lab | Department of CSE, FISAT
"""

import re
import io
import pandas as pd
import streamlit as st
from transformers import pipeline as hf_pipeline, BertTokenizerFast

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineSense | Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling (Cinematic Grade Palette: Teal & Amber on Charcoal) ─────────
st.markdown(
    """
    <style>
    /* Global Base */
    .stApp {
        background-color: #070b0f;
        color: #eaf2f5;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Brand */
    .brand-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0px;
        line-height: 1.1;
    }
    .brand-cine {
        color: #eaf2f5;
    }
    .brand-sense {
        background: linear-gradient(135deg, #19c3b1 0%, #5eead4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-tagline {
        color: #8a9ba5;
        font-size: 1.05rem;
        margin-top: 4px;
        margin-bottom: 24px;
        font-family: monospace;
    }
    
    /* Cards */
    .metric-box {
        background: rgba(13, 20, 26, 0.7);
        border: 1px solid rgba(25, 195, 177, 0.25);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #19c3b1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8a9ba5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Result Card */
    .result-card-pos {
        background: rgba(52, 211, 153, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 0 25px rgba(52, 211, 153, 0.15);
    }
    .result-card-neg {
        background: rgba(255, 93, 93, 0.08);
        border: 1px solid rgba(255, 93, 93, 0.4);
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 0 25px rgba(255, 93, 93, 0.15);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 9999px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Clean input text helper ───────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Model Loader (Cached via st.cache_resource) ──────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading DistilBERT model...")
def load_sentiment_pipeline():
    import os
    MODEL_DIR = "./sentiment_model"
    safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")
    bin_path = os.path.join(MODEL_DIR, "pytorch_model.bin")

    # If local fine-tuned weights exist in ./sentiment_model, load from directory
    if os.path.exists(safetensors_path) or os.path.exists(bin_path):
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        model_source = MODEL_DIR
    else:
        # Streamlit Cloud fallback: load standard DistilBERT fine-tuned on SST-2 / IMDb
        model_source = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = BertTokenizerFast.from_pretrained(model_source)

    classifier = hf_pipeline(
        "text-classification",
        model=model_source,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=256,
        device=-1,  # CPU inference
    )
    return classifier

classifier = load_sentiment_pipeline()

# ── Session State for History ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="brand-title">🎬 <span class="brand-cine">Cine</span><span class="brand-sense">Sense</span></div>
    <div class="brand-tagline">Where every review gets its verdict · AI Sentiment Analyzer</div>
    """,
    unsafe_allow_html=True,
)

# ── Navigation Tabs ───────────────────────────────────────────────────────────
tab_single, tab_batch, tab_about = st.tabs(["🎭 Single Review Analyzer", "🎪 Batch Analysis", "ℹ️ About the Model"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Review Analyzer
# ─────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### Enter a Movie Review")

    col_samples, col_clear = st.columns([4, 1])
    with col_samples:
        st.caption("Quick sample reviews:")
        s_pos, s_neg, s_mix = st.columns(3)
        if s_pos.button("😍 Positive Sample", use_container_width=True):
            st.session_state.sample_input = "An absolute masterpiece! The cinematography, the script, the performances — everything was flawless. One of the best films of the decade!"
        if s_neg.button("😤 Negative Sample", use_container_width=True):
            st.session_state.sample_input = "What a complete waste of two hours. The plot made no sense, the acting was wooden, and the ending felt like the writers just gave up. Avoid at all costs."
        if s_mix.button("🤔 Mixed Sample", use_container_width=True):
            st.session_state.sample_input = "It had some genuinely beautiful moments and good acting, but the pacing was off and the third act was rushed. Worth watching once."

    default_text = st.session_state.get("sample_input", "")
    review_input = st.text_area(
        "Write or paste your review here:",
        value=default_text,
        height=140,
        placeholder="Type your film thoughts, review, or critique...",
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 4])
    analyze_clicked = col_btn.button("🎬 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_clicked:
        cleaned = clean_text(review_input)
        if not cleaned:
            st.warning("⚠️ Please provide a movie review before clicking analyze.")
        else:
            with st.spinner("Analyzing review with DistilBERT..."):
                raw_scores = classifier(cleaned)[0]
                best = max(raw_scores, key=lambda x: x["score"])
                scores_dict = {item["label"]: round(float(item["score"]), 4) for item in raw_scores}
                label = best["label"]
                confidence = float(best["score"])

                # Store in history (max 10)
                st.session_state.history.insert(0, {
                    "review": cleaned,
                    "label": label,
                    "confidence": confidence,
                    "pos_score": scores_dict.get("POSITIVE", 0.0),
                    "neg_score": scores_dict.get("NEGATIVE", 0.0),
                })
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop()

            is_pos = (label == "POSITIVE")
            card_class = "result-card-pos" if is_pos else "result-card-neg"
            verdict_badge = "👍 CERTIFIED FRESH / POSITIVE" if is_pos else "👎 ROTTEN / NEGATIVE"
            emoji = "🌟" if is_pos else "💀"
            accent_color = "#34d399" if is_pos else "#ff5d5d"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 2.2rem;">{emoji}</span>
                            <span style="font-size: 1.8rem; font-weight: 800; color: {accent_color}; margin-left: 12px;">{label}</span>
                            <div style="color: #8a9ba5; font-size: 0.95rem; margin-top: 4px;">{verdict_badge}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.2rem; font-weight: 800; color: {accent_color};">{confidence * 100:.1f}%</div>
                            <div style="color: #8a9ba5; font-size: 0.8rem; text-transform: uppercase;">Confidence</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Sentiment Score Breakdown")
            c1, c2 = st.columns(2)
            c1.metric("Positive Score", f"{scores_dict.get('POSITIVE', 0.0) * 100:.2f}%")
            c1.progress(scores_dict.get("POSITIVE", 0.0))
            c2.metric("Negative Score", f"{scores_dict.get('NEGATIVE', 0.0) * 100:.2f}%")
            c2.progress(scores_dict.get("NEGATIVE", 0.0))

    # Session History
    if st.session_state.history:
        st.markdown("---")
        h_col1, h_col2 = st.columns([4, 1])
        h_col1.markdown(f"#### 🎟️ Recent Screenings ({len(st.session_state.history)}/10)")
        if h_col2.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        for idx, item in enumerate(st.session_state.history, start=1):
            badge = "🟢 POSITIVE" if item["label"] == "POSITIVE" else "🔴 NEGATIVE"
            snippet = item["review"][:120] + ("..." if len(item["review"]) > 120 else "")
            st.markdown(f"**#{idx:02d}** · {badge} · **{item['confidence']*100:.1f}%** — *\"{snippet}\"*")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### 🎪 Batch Review Analysis")
    st.write("Upload a `.csv` (column named `review` or `text`) or a `.txt` file (one review per line, up to 200 reviews).")

    uploaded_file = st.file_uploader("Upload review file", type=["csv", "txt"], label_visibility="collapsed")

    if uploaded_file is not None:
        reviews_to_analyze = []
        if uploaded_file.name.endswith(".csv"):
            df_in = pd.read_csv(uploaded_file)
            col_candidates = [c for c in df_in.columns if c.lower() in ["review", "text", "reviews", "content"]]
            target_col = col_candidates[0] if col_candidates else df_in.columns[0]
            reviews_to_analyze = [str(r) for r in df_in[target_col].dropna().tolist()]
        else:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            reviews_to_analyze = [line.strip() for line in content.splitlines() if line.strip()]

        reviews_to_analyze = reviews_to_analyze[:200]

        if not reviews_to_analyze:
            st.warning("No valid text lines found in the uploaded file.")
        else:
            st.info(f"Loaded **{len(reviews_to_analyze)}** reviews for analysis.")

            if st.button("🚀 Run Batch Prediction", type="primary"):
                progress_bar = st.progress(0)
                batch_results = []

                for i, r in enumerate(reviews_to_analyze):
                    c_text = clean_text(r)
                    if not c_text:
                        continue
                    out = classifier(c_text)[0]
                    best = max(out, key=lambda x: x["score"])
                    s_map = {x["label"]: round(float(x["score"]), 4) for x in out}
                    batch_results.append({
                        "Review": c_text,
                        "Verdict": best["label"],
                        "Confidence (%)": round(float(best["score"]) * 100, 2),
                        "Positive (%)": round(s_map.get("POSITIVE", 0.0) * 100, 2),
                        "Negative (%)": round(s_map.get("NEGATIVE", 0.0) * 100, 2),
                    })
                    progress_bar.progress((i + 1) / len(reviews_to_analyze))

                df_results = pd.DataFrame(batch_results)
                pos_count = (df_results["Verdict"] == "POSITIVE").sum()
                neg_count = (df_results["Verdict"] == "NEGATIVE").sum()

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Reviews", len(df_results))
                m2.metric("Positive Verdicts", pos_count)
                m3.metric("Negative Verdicts", neg_count)

                # Donut / Chart
                chart_data = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative"],
                    "Count": [pos_count, neg_count],
                })
                st.bar_chart(chart_data.set_index("Sentiment"))

                # Results table
                st.dataframe(df_results, use_container_width=True)

                # Download CSV
                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="cinesense_batch_results.csv",
                    mime="text/csv",
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: About the Model
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### 🎥 Behind the Scenes: DistilBERT Sentiment Model")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            """
            **DistilBERT Transformer**  
            CineSense leverages a fine-tuned **DistilBERT** architecture (`distilbert-base-uncased`) — a light, high-performance distilled version of BERT. It retains 97% of BERT's natural language understanding while operating with 40% fewer parameters and 60% faster inference.

            **Large Movie Review Dataset (IMDb)**  
            Trained and evaluated on the benchmark IMDb dataset consisting of **50,000 highly polarized movie reviews** balanced between positive and negative sentiments (25,000 train / 25,000 test).

            **Transfer Learning & Fine-Tuning**  
            The base weights were pre-trained on large English corpora, with classification heads adapted specifically to IMDb text representations.
            """
        )

    with col_b:
        st.markdown("#### Test Set Performance")
        m_col1, m_col2 = st.columns(2)
        m_col1.markdown(
            """
            <div class="metric-box">
                <div class="metric-value">93.21%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <br>
            <div class="metric-box">
                <div class="metric-value">93.08%</div>
                <div class="metric-label">Precision</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m_col2.markdown(
            """
            <div class="metric-box">
                <div class="metric-value">93.37%</div>
                <div class="metric-label">Recall</div>
            </div>
            <br>
            <div class="metric-box">
                <div class="metric-value">93.22%</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #8a9ba5; font-size: 0.85rem;">
        🎬 <strong>CineSense</strong> · Movie Review Sentiment Analyzer<br>
        Department of Computer Science & Engineering · Federal Institute of Science and Technology (FISAT)<br>
        <em>PCCSL 508 — Machine Learning Lab</em>
    </div>
    """,
    unsafe_allow_html=True,
)
