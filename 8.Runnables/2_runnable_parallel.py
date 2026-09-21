from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation")

prompt1 = PromptTemplate(
    template="create a tweet post on {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="create a linkedin post on {text}", input_variables=["text"]
)

model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()

chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1 | model | parser),
        "linkedin": RunnableSequence(prompt2 | model | parser),
    }
)


result = chain.invoke({"topic": "AI","text":"openAI"})

print(result)


