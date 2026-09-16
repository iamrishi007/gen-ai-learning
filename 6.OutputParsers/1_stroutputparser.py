from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
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

Model = ChatHuggingFace(llm=llm)

prompt1 = template1.invoke({"topic": "black hole"})
result = Model.invoke(prompt1)
prompt2 = template2.invoke({"text": result.content})
result1 = Model.invoke(prompt2)
print(result1.content)
