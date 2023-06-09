from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

model_path = "../llama.cpp/models/wizard-vicuna-13B.ggmlv3.q4_0.bin"
gpu_path = "/home/ubuntu/repos/llama.cpp/models/wz-gpu.ggmlv3.q4_1.bin"  # CUDA enabled
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 40  # based on model and GPU VRAM pool
n_batch = 512  # between 1 and n_ctx, consider VRAM in GPU

template = """
           On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing 
           teeth, making bed) and 10 is extremely poignant (e.g., a break up, 
           college acceptance), rate the likely poignancy of the following piece
           of memory, and give a short explanation on why the rating was given

           Memory: {fragment}

           Rating: <fill in> 
           """
prompt = PromptTemplate(template=template, input_variables=["fragment"])

llm = LlamaCpp(
    model_path=gpu_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

fragment = "Your dog ran away when you were a child."
llm_chain.run(fragment)
