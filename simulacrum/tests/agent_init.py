from faiss import IndexFlatL2

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import LlamaCppEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from agent.agent import Agent
from models.local_llamas import vicuna


llm = vicuna()
embeddings = LlamaCppEmbeddings(
    model_path="../llama.cpp/models/wizard-vicuna-13B.ggmlv3.q4_0.bin"
)

agent = Agent(
    name="Bob",
    description="A helpful young man who's good at Math proofs.",
    llm=llm,
    long_term_memory=TimeWeightedVectorStoreRetriever(
        vectorstore=FAISS(
            embedding_function=embeddings.embed_query,
            index=IndexFlatL2(5120),
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
    ),
    verbose=True,
    max_token_limit=1200,
    reflection_threshold=0.6,
)

observations = [
    "Bob remembers his dog, Bruno, from when he was a kid",
    "Bob sees the new home his wife bought for them",
]
for observation in observations:
    agent.add_memory(observation)

context = agent.get_context()
