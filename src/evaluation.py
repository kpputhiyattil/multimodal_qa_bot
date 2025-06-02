import pandas as pd
from time import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from huggingface_implementation import hugging_face_query
from langchain_implementation import langchain_query
from llama_index_implementation import llama_index_query

import nltk

nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key

# Ground truth answer for the question

query = "What is the main contribution of the paper?"
file = "testing.pdf"
reference_answer = "The introduction of the Transformer model, which relies entirely on attention mechanisms."

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load embedding model
def semantic_similarity(a, b):
    embeddings = embedding_model.encode([a, b])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


# Evaluation functions
# Sentence embedding model for semantic similarity


# Example usage for each system
def compute_f1(pred, ref):
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(candidate, reference):
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothing)


def compute_rouge(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return score['rougeL'].fmeasure


model_list = []
f1_list = []
latency_list = []
similarity_list = []
bleu_list = []
rouge_list = []


def evaluate_qa(system_name, answer, time_taken):
    # Semantic similarity
    f1 = compute_f1(answer, reference_answer)
    similarity = semantic_similarity(answer, reference_answer)
    bleu = compute_bleu(answer, reference_answer)
    rouge = compute_rouge(answer, reference_answer)
    model_list.append(system_name)
    f1_list.append(f1)
    latency_list.append(time_taken)
    similarity_list.append(similarity)
    bleu_list.append(bleu)
    rouge_list.append(rouge)
    print(f"\n{system_name} Evaluation:")
    print(f"Answer: {answer}")
    print(f"Latency: {time_taken:.2f} sec")
    print(f"Semantic Similarity: {similarity:.3f}")
    print(f"F1: {f1:.2f}")
    print(f"BLEU: {bleu:.3f}")
    print(f"ROUGE-L F1: {rouge:.3f}")
    # Add more metrics if needed


start = time()
print("langchain_query")
result = langchain_query(file, api_key, query)
end = time()
evaluate_qa("LangChain", result, end - start)

start = time()
print("llama_index_query")
response = llama_index_query(file, query)
end = time()
evaluate_qa("LlamaIndex", str(response), end - start)

start = time()
print("hugging_face_query")
result = hugging_face_query(file, query)
end = time()
evaluate_qa("HuggingFace", result, end - start)

results_df = pd.DataFrame(
    {"model": model_list, "f1 score": f1_list, "bleu": bleu_list, "rogue":rouge_list, "latency": latency_list, "similarity": similarity_list})
results_df.to_excel("model_comparison.xlsx", index=False)
print("Results Exported")
