from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from llama_index import download_loader
from llama_index import ServiceContext
from llama_index import SimpleDirectoryReader

import os
import json
import torch
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:5000",
    "http://localhost:7000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/v1/embed")
async def embed(request: Request):
    data = await request.json()
    model_path = data["model_path"]
    device = data["device"]
    text = data["text"]
    
    if model_path == "llama-7b":
        model = download_loader("Llama-7B")
    elif model_path == "llama-13b":
        model = download_loader("Llama-13B")
    elif model_path == "llama-30b":
        model = download_loader("Llama-30B")
    elif model_path == "llama-65b":
        model = download_loader("Llama-65B")
    elif model_path == "llama-70b":
        model = download_loader("Llama-70B")
    elif model_path == "llama-130b":
        model = download_loader("Llama-130B")
    else:
        raise ValueError("Invalid model path")
    
    service_context = ServiceContext.from_defaults(model=model)
    embeddings = service_context.embed(text)
    return {"embeddings": embeddings.tolist()}

@app.post("/api/v1/query")
async def query(request: Request):
    data = await request.json()
    model_path = data["model_path"]
    device = data["device"]
    text = data["text"]
    
    if model_path == "llama-7b":
        model = download_loader("Llama-7B")
    elif model_path == "llama-13b":
        model = download_loader("Llama-13B")
    elif model_path == "llama-30b":
        model = download_loader("Llama-30B")
    elif model_path == "llama-65b":
        model = download_loader("Llama-65B")
    elif model_path == "llama-70b":
        model = download_loader("Llama-70B")
    elif model_path == "llama-130b":
        model = download_loader("Llama-130B")
    else:
        raise ValueError("Invalid model path")
    
    service_context = ServiceContext.from_defaults(model=model)
    query = service_context.query(text)
    return {"query": query}

@app.post("/api/v1/document")
async def document(request: Request):
    data = await request.json()
    model_path = data["model_path"]
    device = data["device"]
    text = data["text"]
    
    if model_path == "llama-7b":
        model = download_loader("Llama-7B")
    elif model_path == "llama-13b":
        model = download_loader("Llama-13B")
    elif model_path == "llama-30b":
        model = download_loader("Llama-30B")
    elif model_path == "llama-65b":
        model = download_loader("Llama-65B")
    elif model_path == "llama-70b":
        model = download_loader("Llama-70B")
    elif model_path == "llama-130b":
        model = download_loader("Llama-130B")
    else:
        raise ValueError("Invalid model path")
    
    service_context = ServiceContext.from_defaults(model=model)
    documents = service_context.document(text)
    return {"documents": documents}    

@app.post("/api/v1/text_splitter")
async def text_splitter(request: Request):
    data = await request.json()
    model_path = data["model_path"]
    device = data["device"]
    text = data["text"]
    
    if model_path == "llama-7b":
        model = download_loader("Llama-7B")
    elif model_path == "llama-13b":
        model = download_loader("Llama-13B")
    elif model_path == "llama-30b":
        model = download_loader("Llama-30B")
    elif model_path == "llama-65b":
        model = download_loader("Llama-65B")
    elif model_path == "llama-70b":
        model = download_loader("Llama-70B")
    elif model_path == "llama-130b":
        model = download_loader("Llama-130B")
    else:
        raise ValueError("Invalid model path")
    
    service_context = ServiceContext.from_defaults(model=model)
    text_splitter = service_context.text_splitter(text)
    return {"text_splitter": text_splitter}    

@app.post("/api/v1/langchain_chain")
async def langchain_chain(request: Request):
    data = await request.json()
    langchain_chain = data["langchain_chain"]
    service_context = ServiceContext.from_defaults(model=model)
    langchain_chain = service_context.langchain_chain(langchain_chain)
    return {"langchain_chain": langchain_chain}    

@app.post("/api/v1/llama")
async def llama(request: Request):
    data = await request.json()
    llama = data["llama"]
    service_context = ServiceContext.from_defaults(model=model)
    llama = service_context.llama(llama)
    return {"llama": llama}    

@app.post("/api/v1/llama_index")
async def llama_index(request: Request):
    data = await request.json()
    llama_index = data["llama_index"]
    service_context = ServiceContext.from_defaults(model=model)
    llama_index = service_context.llama_index(llama_index)
    return {"llama_index": llama_index}    

@app.post("/api/v1/llama_index_query")
async def llama_index_query(request: Request):
    data = await request.json()
    llama_index_query = data["llama_index_query"]
    service_context = ServiceContext.from_defaults(model=model)
    llama_index_query = service_context.llama_index_query(llama_index_query)
    return {"llama_index_query": llama_index_query}    

@app.post("/api/v1/llama_index_documents")
async def llama_index_documents(request: Request):
    data = await request.json()
    llama_index_documents = data["llama_index_documents"]
    service_context = ServiceContext.from_defaults(model=model)
    llama_index_documents = service_context.llama_index_documents(llama_index_documents)
    return {"llama_index_documents": llama_index_documents}    

@app.post("/api/v1/llama_index_text_splitter")
async def llama_index_text_splitter(request: Request):
    data = await request.json()
    llama_index_text_splitter = data["llama_index_text_splitter"]
    service_context = ServiceContext.from_defaults(model=model)
    llama_index_text_splitter = service_context.llama_index_text_splitter(llama_index_text_splitter)
    return {"llama_index_text_splitter": llama_index_text_splitter}    

@app.post("/api/v1/llama_index_langchain_chain")
async def llama_index_langchain_chain(request: Request):
    data = await request.json()
    llama_index_langchain_chain = data["llama_index_langchain_chain"]
    service_context = ServiceContext.from_defaults(model=model)
    llama_index_langchain_chain = service_context.llama_index_langchain_chain(llama_index_langchain_chain)        
    return {"llama_index_langchain_chain": llama_index_langchain_chain}