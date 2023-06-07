from langchain.llms import LlamaCpp
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

path = "/home/ubuntu/repos/augmented-agents/llama.cpp/models/13B/ggml-model-q4_0.bin"


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "The following is a angry conversation between the President "
            + "of the United States and Hanna Montana. "
            + "The President doesn't believe that she's also Miley Cyrus."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

llm = LlamaCpp(model_path=path, n_ctx=2048, temperature=0.5)

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

print(conversation.predict(input="Sup?"))
print(conversation.predict(input="Tell me about your mom."))
