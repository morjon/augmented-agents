from faiss import IndexFlatL2

from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS


from models.local_llamas import vicuna

llm = vicuna()
embeddings = llm.get_embeddings()


class AgentMemory(TimeWeightedVectorStoreRetriever):
    vectorstore = FAISS(
        embedding_function=embeddings.embed_query,
        index=IndexFlatL2(5120),
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )

    # def add_memory(self, text: str, doc_id: str = None):
    #     document = Document(text=text, doc_id=doc_id)
    #     self.memory.store(document)

    # def get_context(self, num_results=10):
    #     return self.memory.retrieve_top_k("", num_results)
