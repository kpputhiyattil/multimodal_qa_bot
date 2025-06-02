import os
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI


def llama_index_query(file, query):
    # Load your PDF
    documents = SimpleDirectoryReader(input_files=[file]).load_data()
    Settings.llm = OpenAI(model="gpt-3.5-turbo")

    # Create vector index
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)
    return response
