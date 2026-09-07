from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Create an OpenAI LLM instance
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# Send a prompt to the LLM
result = llm.invoke("Who is the Prime Minister of India?")

print(result)
