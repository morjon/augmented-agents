from faiss import IndexFlatL2

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import LlamaCppEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from agent.agent import Agent

# from memory.retriever import MemoryRetriever
from models.llama import llama
from utils.callbacks import ConsolePrettyPrintManager

llm = llama(model_name=1, callback_manager=ConsolePrettyPrintManager([]))
embeddings = LlamaCppEmbeddings(
    model_path="/home/ubuntu/repos/augmented-agents/llama.cpp/models/13B/ggml-model-q4_0.bin"
)


agent = Agent(
    name="Bob",
    description="A helpful assistant",
    llm=llm,
    long_term_memory=TimeWeightedVectorStoreRetriever(
        # long_term_memory=MemoryRetriever(
        vectorstore=FAISS(
            embedding_function=embeddings.embed_query,
            index=IndexFlatL2(5120),
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
    ),
    verbose=False,
    max_token_limit=1000,
    reflection_threshold=0.6,
)

observations = [
    "Bob feels tired from driving so far",
    "Bob remembers his dog, Bruno, from when he was a kid",
    "Bob sees the new home",
    "The new neighbors have a cat",
    "The road is noisy at night",
    "Bob is hungry",
    "Bob tries to get some rest.",
]
for observation in observations:
    agent.add_memory(observation)

summary = agent.get_summary(num_memories=3)
print(summary)

summary = agent.get_summary(num_memories=10)
print(summary)
