from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation"
)


prompt1 = PromptTemplate(
    template="give me the details about this {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="give me the 5 key point about {text}", input_variables=["text"]
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "ai in india"})
print(result)

print(chain.get_graph().draw_ascii())
