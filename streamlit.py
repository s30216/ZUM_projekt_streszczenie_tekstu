import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from rouge_score import rouge_scorer

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Load test dataset
ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
df_test = ds['test'].to_pandas()

# Summarization function
def summarize(text):
    inputs = tokenizer(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids=inputs['input_ids'], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Compute ROUGE scores
def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        "ROUGE-1": sum(rouge1_scores) / len(rouge1_scores),
        "ROUGE-2": sum(rouge2_scores) / len(rouge2_scores),
        "ROUGE-L": sum(rougeL_scores) / len(rougeL_scores),
    }

# Evaluation function
def evaluate_model():
    predictions = []
    references = df_test["highlights"].tolist()

    for article in df_test["article"]:
        summary = summarize(article)
        predictions.append(summary)

    return predictions, references

# Streamlit app layout
st.title("Summarization Model Dashboard")

# Sidebar for mode selection
mode = st.sidebar.radio(
    "Choose Mode",
    options=["Test Dataset Sample", "Custom Input"]
)

if mode == "Custom Input":
    input_text = st.text_area("Enter text to summarize:")
    if st.button("Generate Summary"):
        if input_text.strip():
            summary = summarize(input_text)
            st.subheader("Generated Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text.")
else:
    sample_id = st.slider("Select a sample from the test dataset:", 0, len(df_test)-1, 0)
    st.subheader("Original Article")
    st.write(df_test.iloc[sample_id]["article"])
    if st.button("Generate Summary"):
        summary = summarize(df_test.iloc[sample_id]["article"])
        st.subheader("Generated Summary")
        st.write(summary)

# Evaluation Metrics Section
st.subheader("Evaluation Metrics")
if st.button("Compute Evaluation Metrics"):
    with st.spinner("Evaluating..."):
        predictions, references = evaluate_model()
        metrics = compute_rouge(predictions, references)
        for metric, value in metrics.items():
            st.metric(label=metric, value=f"{value:.4f}")

# Error Analysis Section
st.subheader("Error Analysis")
if st.button("Show Error Analysis"):
    with st.spinner("Analyzing errors..."):
        mismatches = []
        for i, article in enumerate(df_test["article"]):
            generated_summary = summarize(article)
            actual_summary = df_test.iloc[i]["highlights"]
            if generated_summary != actual_summary:
                mismatches.append((article, generated_summary, actual_summary))
        if mismatches:
            for i, mismatch in enumerate(mismatches[:5]):
                st.write(f"**Mismatch {i+1}**")
                st.text_area("Original Article", mismatch[0][:300] + "...")
                st.text_area("Generated Summary", mismatch[1])
                st.text_area("Actual Summary", mismatch[2])
        else:
            st.success("No mismatches found!")
