from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation"
)

prompt1 = PromptTemplate(template="write a joke on {topic}", input_variables=["topic"])
prompt2 = PromptTemplate(template="explain a joke {text}", input_variables=["text"])

model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()


joke_gen_chain = RunnableSequence(prompt1 | model | parser)

parallal_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(prompt2 | model | parser),
    }
)

chain = RunnableSequence(joke_gen_chain | parallal_chain)
result = chain.invoke({"topic": "AI In India"})

print(result)
print("joke", result["joke"])
print("explanation", result["explanation"])
