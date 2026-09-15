from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()


class Person(TypedDict):
    name: str
    age: int
    proffetion: str


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text-generation",
)


model = ChatHuggingFace(llm=llm)
output = model.with_structured_output(Person)

result = output.invoke(
    "My name is John, I am 25 years old and I work as a software engineer."
)

print(result)
