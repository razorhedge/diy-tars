from models.llama import Llama7B
from models.llama import Llama13B
from models.llama import Llama30B
from models.llama import Llama65B
from models.llama import Llama70B
from models.llama import Llama130B
from models.llama import Llama7B
from models.llama import Llama13B
from models.llama import Llama30B
from models.llama import Llama65B
from models.llama import Llama70B
from models.llama import Llama130B


class PreTrainedModel:
    def __init__(self, model_path, device):
        if model_path == "llama-7b":
            self.model = Llama7B(model_path=model_path)
        elif model_path == "llama-13b":
            self.model = Llama13B(model_path=model_path)
        elif model_path == "llama-30b":
            self.model = Llama30B(model_path=model_path)
        elif model_path == "llama-65b":
            self.model = Llama65B(model_path=model_path)
        elif model_path == "llama-70b":
            self.model = Llama70B(model_path=model_path)
        elif model_path == "llama-130b":
            self.model = Llama130B(model_path=model_path)
        else:
            raise ValueError("Invalid model path")
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