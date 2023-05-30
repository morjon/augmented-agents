from langchain.llms import LlamaCpp
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

path = "../llama.cpp/models/13B/ggml-model-q4_0.bin"

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and God. God is quite dull, but is fascinated by the human's existence."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = LlamaCpp(
    model_path=path,
    n_ctx = 2048,
    # n_batch = 8,
    temperature = 0.5
)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

print(conversation.predict(input="Sup?"))
print(conversation.predict(input="Tell me about your favorite memory."))