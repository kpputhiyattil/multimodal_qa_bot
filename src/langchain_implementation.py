from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import langchain_openai
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import CharacterTextSplitter


def langchain_query(file, api_key, query):
    # Load and split PDF
    loader = PyPDFLoader(file)
    documents = loader.load()
    # Better text splitting with overlap for context
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    documents = text_splitter.split_documents(documents)

    # Embed and store with OpenAI
    embedding = langchain_openai.OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(documents, embedding)

    # Create retrieval chain with chat memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=langchain_openai.ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0),
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True
    )

    # Initialize chat history
    chat_history = []

    # Run a query
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    print("Answer:", result["answer"])
    return result["answer"]

