from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation"
)

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}", input_variables=["text"]
)


model = ChatHuggingFace(llm=LLM)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser), RunnablePassthrough()
)

result = RunnableSequence(report_gen_chain, branch_chain)

final_result = result.invoke({"topic": "russia vs ukraine war"})
print(final_result)
