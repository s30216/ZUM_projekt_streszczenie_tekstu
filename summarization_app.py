import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
df_test = ds['test'].to_pandas()
df_test = df_test.sample(100, random_state=42)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def summarize(text):
    inputs = tokenizer(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids=inputs['input_ids'], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_batch(texts, batch_size=8):
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Tokenize and move tensors to the correct device
        inputs = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to device

        # Generate summaries
        outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150)
        summaries.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return summaries

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

def evaluate_model(df_test, batch_size=16):
    references = df_test["highlights"].tolist()
    articles = df_test["article"].tolist()

    predictions = []
    progress_bar = st.progress(0)
    total_batches = (len(articles) + batch_size - 1) // batch_size

    def process_batch(start_idx):
        batch = articles[start_idx:start_idx + batch_size]
        return summarize_batch(batch, batch_size)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, i) for i in range(0, len(articles), batch_size)]
        for idx, future in enumerate(futures):
            predictions.extend(future.result())
            progress_bar.progress((idx + 1) / total_batches)

    return predictions, references

st.title("Summarization Model Dashboard")

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

st.subheader("Evaluation Metrics")
if st.button("Compute Evaluation Metrics"):
    with st.spinner("Evaluating..."):
        predictions, references = evaluate_model(df_test)
        metrics = compute_rouge(predictions, references)
        for metric, value in metrics.items():
            st.metric(label=metric, value=f"{value:.4f}")

st.subheader("Error Analysis")
if st.button("Show Error Analysis"):
    with st.spinner("Analyzing errors..."):
        mismatches = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_article = {executor.submit(summarize, article): idx for idx, article in enumerate(df_test["article"])}

            for future in concurrent.futures.as_completed(future_to_article):
                idx = future_to_article[future]
                try:
                    generated_summary = future.result()
                    actual_summary = df_test.iloc[idx]["highlights"]

                    if generated_summary != actual_summary:
                        mismatches.append((df_test.iloc[idx]["article"], generated_summary, actual_summary))
                except Exception as e:
                    st.error(f"Error processing article {idx}: {e}")

        if mismatches:
            for i, mismatch in enumerate(mismatches[:5]):
                st.write(f"**Mismatch {i + 1}**")
                st.text_area("Original Article", mismatch[0][:300] + "...", key=f"article_{i}")
                st.text_area("Generated Summary", mismatch[1], key=f"summary_{i}")
                st.text_area("Actual Summary", mismatch[2], key=f"reference_{i}")
        else:
            st.success("No mismatches found!")
