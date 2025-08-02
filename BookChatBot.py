
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableBranch
from dotenv import load_dotenv
from chromadb.config import Settings
import os

import nest_asyncio
nest_asyncio.apply()

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_TOKEN")

def setup(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    client_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=None,  # disables file system writes
    )
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=None,  
        collection_name="pdf-chat"
        client_settings=client_settings
    )
    vector_store.add_documents(chunks)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_store.as_retriever(search_kwargs={"k": 5}), bm25],
        weights=[0.5, 0.5]
    )

    return chunks, hybrid_retriever

def build_chain(chunks, hybrid_retriever):
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.2)
    parser = StrOutputParser()

    rewrite_prompt = PromptTemplate.from_template(
        "Please improve the query to make retrieval more effective:\nOriginal: {query}\nImproved:"
    )
    intent_prompt = PromptTemplate.from_template(
        """Classify this user query into one of the following categories:
- summary
- structure
- factual
- opinion
- general information

Query: {question}
Intent:"""
    )
    summarizing_prompt = PromptTemplate.from_template(
        "You are in a multi-turn conversation. Here is the history:\n\n{chat_history}\n\n"
        "Use the following document to answer the question:\n\n{context}\n\n"
        "User: {question}\nBot:"
    )
    general_info_prompt = PromptTemplate.from_template(
        "You are in a multi-turn conversation. Here is the history:\n\n{chat_history}\n\n"
        "Use the following document to answer the question:\n\n{context}\n\n"
        "For this kind of question you get your information from the starting pages of the document, "
        "right before the introduction of the article. Questions may include the name of the author or title of the article.\n\n"
        "User: {question}\nBot:"
    )
    factual_prompt = PromptTemplate.from_template(
        "You are in a multi-turn conversation. Here is the history:\n\n{chat_history}\n\n"
        "Use the following retrieved context to answer:\n\n{context}\n\n"
        "User: {question}\nBot:"
    )

    rewrite_chain = rewrite_prompt | llm | parser
    intent_chain = intent_prompt | llm | parser

    intent_map = RunnableMap({
        "question": lambda x: x["question"],
        "rewritten": lambda x: rewrite_chain.invoke({"query": x["question"]}),
        "intent": lambda x: intent_chain.invoke({"question": x["question"]}).strip().lower(),
        "chat_history": lambda x: x.get("chat_history", "")
    })

    summarizing_chain = RunnableMap({
        "context": lambda x: "\n\n".join([doc.page_content for doc in chunks]),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }) | summarizing_prompt | llm | parser

    general_info_chain = RunnableMap({
        "context": lambda x: "\n\n".join([doc.page_content for doc in chunks]),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }) | general_info_prompt | llm | parser

    factual_chain = RunnableMap({
        "context": lambda x: "\n\n".join([doc.page_content for doc in hybrid_retriever.invoke(x["rewritten"])]),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }) | factual_prompt | llm | parser

    decide_chain = RunnableBranch(
        (lambda x: x["intent"] in ["opinion", "summary", "structure"], summarizing_chain),
        (lambda x: x["intent"] in ["general information"], general_info_chain),
        factual_chain
    )

    return intent_map | decide_chain

def format_chat_history(history):
    return "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
