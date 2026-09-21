from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation"
)

prompt1 = PromptTemplate(template="write a joke on {topic}", input_variables=["topic"])

model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="explain the following joke {text}", input_variables=["text"]
)

chain = RunnableSequence(prompt1 | model | parser | prompt2 | model | parser)
result = chain.invoke({"topic": "Ai"})

print(result)
