from llama_cpp import Llama
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp

path = "../llama.cpp/models/13B/ggml-model-q4_0.bin"

# standard model loading for embedding lib using high-level API
llm = Llama(model_path=path)
output = llm(
    "Q: What is the best song in the world? Think about the problem carefully,"
    "and step by step. A: ",
    max_tokens=50,
    stop=["Q:", "\n"],
    echo=True,
)
print(output)

# langchain embeddings
lang_llama = LlamaCppEmbeddings(model_path=path)
prompt = "Q: Who was the greatest NBA superstar to live? A: "
query_results = lang_llama.embed_query(prompt)
result = lang_llama.embed_documents([prompt])

# langchain streaming
llm = LlamaCpp(
    model_path=path,
    temperature = 0.5
)
for chunk in llm.stream("Ask 'Hi, how are you this afternoon?' like Caesar: '",
        stop=["'","\n"]):
    result = chunk["choices"][0]
    print(result["text"], end='', flush=True)