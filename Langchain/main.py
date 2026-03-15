import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

docs  = TextLoader("doc.txt").load()
texts = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)
store = Chroma.from_documents(texts, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

chain = (
    {"context": store.as_retriever() | (lambda d: "\n".join(x.page_content for x in d)),
     "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
    | ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    | StrOutputParser()
)

questions = [
    "What is LangChain?",
    "What is RAG?",
    "What is LCEL?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {chain.invoke(q)}")
    print("-" * 60)
