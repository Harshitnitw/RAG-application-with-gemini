from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st

st.title("RAG app demo with google gemini")

persist_directory='db'

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    loader=PyPDFLoader("/workspaces/codespaces-blank/my_paper.pdf")
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000)
    docs=text_splitter.split_documents(data)
    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vectorstore=Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory=persist_directory)

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":10})
llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3,max_tokens=500)

query=st.chat_input("Ask me anything:")

system_prompt=(
    """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, say that you don't know.
    Use three sentences maximum and keep the answer concise.
    \n\n
    {context}
    """
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)

if query:
    question_answer_chain=create_stuff_documents_chain(llm,prompt)
    rag_chain=create_retrieval_chain(retriever,question_answer_chain)
    response=rag_chain.invoke({"input":query})
    st.write(response["answer"])