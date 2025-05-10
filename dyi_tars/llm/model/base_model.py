from llama_index.expermental import SingleDeviceEmbedding
from llama_index.embeddings import OpenAIEmbedding
from llama_index.indices.query.embedding_utils import get_embedding
from llama_index.indices.query.embedding_utils import get_embedding_function
from llama_index.indices.query.embedding_utils import get_embedding_model
from llama_index.indices.query.embedding_utils import get_llama_embedding
from llama_index.indices.vector_store.base_query import BaseQuery
from llama_index.langchain_helpers.chain_wrapper import LangChainChainWrapper
from llama_index.langchain_helpers.text_splitter import TextSplitter
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.langchain_helpers.text_splitter import RecursiveCharacterTextSplitter
from llama_index.langchain_helpers.text_splitter import SpacyTextSplitter
from llama_index.langchain_helpers.text_splitter import MarkdownTextSplitter
from llama_index.langchain_helpers.text_splitter import CharacterTextSplitter


class BaseModel:
    def __init__(self, model_path, device):
        self.model = OpenAIEmbedding(model_path=model_path)
        self.device = device
        self.model.to(self.device)

    def embed(self, text):
        return self.model.embed(text)

    def embed_query(self, query):
        return self.model.embed_query(query)

    def embed_documents(self, documents):
        return self.model.embed_documents(documents)

    def embed_text_splitter(self, text_splitter):
        return self.model.embed_text_splitter(text_splitter)

    def embed_langchain_chain(self, langchain_chain):
        return self.model.embed_langchain_chain(langchain_chain)

    def embed_llama(self, llama):
        return self.model.embed_llama(llama)

    def embed_llama_index(self, llama_index):
        return self.model.embed_llama_index(llama_index)

    def embed_llama_index_query(self, llama_index_query):
        return self.model.embed_llama_index_query(llama_index_query)

    def embed_llama_index_documents(self, llama_index_documents):
        return self.model.embed_llama_index_documents(llama_index_documents)

    def embed_llama_index_text_splitter(self, llama_index_text_splitter):
        return self.model.embed_llama_index_text_splitter(llama_index_text_splitter)

    def embed_llama_index_langchain_chain(self, llama_index_langchain_chain):
        return self.model.embed_llama_index_langchain_chain(llama_index_langchain_chain)    
    

    