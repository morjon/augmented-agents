from langchain.llms import LlamaCpp
# from llama_cpp import Llama
# from langchain.embeddings import LlamaCppEmbeddings

path="/home/ubuntu/repos/augmented-agents/llama.cpp/models/30B/ggml-model-q4_0.bin"

# standard model loading for embedding lib using high-level API
# llm = Llama(model_path=path)
# output = llm(
#     "Q: What is the best song in the world? Think about the problem carefully,"
#     "and step by step. A: ",
#     max_tokens=50,
#     stop=["Q:", "\n"],
#     echo=True,
# )
# print(output)

# langchain embeddings
# lang_llama = LlamaCppEmbeddings(model_path=path)
# prompt = "Q: Who was the greatest NBA superstar to live? A: "
# query_results = lang_llama.embed_query(prompt)
# result = lang_llama.embed_documents([prompt])

# langchain streaming
llm = LlamaCpp(
    model_path=path,
    temperature=0.8,
    n_ctx=2000,
    # top_p=0.70,
    repeat_penalty=1.20
)
for chunk in llm.stream("On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory. Memory: Your mother died. Rating: <fill in> ",
        stop=["'","\n"]):
    result = chunk["choices"][0]
    print(result["text"], end='', flush=True)