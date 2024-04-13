#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import pandas as pd
import numpy as np
import ollama
import langchain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from concurrent.futures import ThreadPoolExecutor

def load_pdf_data(file_path):
    if file_path:
        loader = UnstructuredPDFLoader(file_path=file_path)
        return loader.load()
    else:
        raise ValueError("Invalid file path")


def split_text_into_chunks(data, chunk_size=512, chunk_overlap=51):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(data)


def create_vector_db(chunks, model="mxbai-embed-large", collection_name="GE-2024-8K"):
    return Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=model, show_progress=True),
        collection_name=collection_name,
    )


def concurrent_search(vector_db, query_vector, top_k=5, num_threads=16):
    """ Function to perform concurrent searches using ThreadPoolExecutor. """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_vector = {executor.submit(vector_db.search, vec, top_k): vec for vec in query_vector}
        results = {}
        for future in future_to_vector:
            vec = future_to_vector[future]
            try:
                results[vec] = future.result()
            except Exception as exc:
                results[vec] = str(type(exc))
        return results


def setup_retriever(vector_db, llm):
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are an AI language model assistant named Plex. You retrieve documents from vector databases and provide additional questions for the user once you read and analyzed the code.
        """,
    )
    return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=query_prompt)


def setup_chain(retriever, llm):
    template = """ANSWER THE QUESTION BASED ONLY ON THE FOLLOWING CONTEXT: {context}"""
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


import numpy as np

def get_query_vectors(chunks, model, num_samples=5):
    """Assuming you need to recompute vectors for querying, which is not efficient but for illustration."""
    embedding_model = OllamaEmbeddings(model=model)
    query_vectors = [embedding_model.embed(chunk.text) for chunk in chunks[:num_samples]]
    return query_vectors

def main():
    print(sys.prefix)
    os.system("ollama list")

    file_path = "./data/GE.pdf"
    data = load_pdf_data(file_path)

    chunks = split_text_into_chunks(data)
    vector_db = create_vector_db(chunks)

    local_model = "mistral:7b-instruct-v0.2-fp16"
    llm = ChatOllama(model=local_model)

    retriever = setup_retriever(vector_db, llm)
    chain = setup_chain(retriever, llm)

    # Compute query vectors properly
    query_vectors = get_query_vectors(chunks, model="mxbai-embed-large")

    # Perform concurrent searches
    results = concurrent_search(vector_db, query_vectors, top_k=3, num_threads=4)
    for result in results.values():
        print(result)

    user_input = input("Type something in, I guess:")
    result = chain.invoke(user_input)
    print(result)

if __name__ == "__main__":
    main()

