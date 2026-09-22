from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation"
)


prompt = PromptTemplate(
    template="Write a summary for the following text in short - \n {text}",
    input_variable=["text"],
)

model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()


loader = TextLoader("demo.txt", encoding="utf-8")

docs = loader.load()


chain = prompt | model | parser

result = chain.invoke({"text": docs[0].page_content})
print(result)
