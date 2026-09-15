from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()


class ProductReview(TypedDict):
    product_name: Annotated[str, "Write the product name"]
    rating: Annotated[int, "Give the product rating"]
    summary: Annotated[str, "Write a short summary of the review"]
    pros: Annotated[list[str], "List the advantages/pros of the product"]
    cons: Annotated[list[str], "List the disadvantages/cons of the product"]


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text-generation",
)


model = ChatHuggingFace(llm=llm)
output = model.with_structured_output(ProductReview)

result = output.invoke(
    """"Overall, I love the Sony WH-1000XM5 headphones. The noise cancellation
is excellent, the sound quality is great, and they are very comfortable
for long use. However, the headphones are expensive and the microphone
quality could be better. I would rate them 4 out of 5 stars."""
)

print(result["product_name"])
