from langchain_community.document_loaders import PyPDFLoader
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

loader = PyPDFLoader(
    "9.DocumentLoaders\Approach-Paper-on-Regulation-of-Gen-AI-in-India.pdf"
)

model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"text": docs[4].page_content})

print(result)