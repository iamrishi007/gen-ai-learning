from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation"
)

template1 = PromptTemplate(
    template="'Write a detailed report on {topic}'", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "black hole"})
print(result)
