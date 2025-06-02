import pdfplumber
from transformers import pipeline


def hugging_face_query(file, query):
    # Extract PDF text
    with pdfplumber.open(file) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)

    # QA Pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    result = qa_pipeline(question=query, context=text)
    print(result["answer"])
    return result["answer"]
