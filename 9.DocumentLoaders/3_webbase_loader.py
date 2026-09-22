from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation"
)

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"],
)

model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()

url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"

loader = WebBaseLoader(url)

doct = loader.load()

chain = prompt | model | parser
result = chain.invoke(
    {
        "question": "What is the prodcut that we are talking about ?",
        "text": doct[0].page_content,
    }
)

print(result)
