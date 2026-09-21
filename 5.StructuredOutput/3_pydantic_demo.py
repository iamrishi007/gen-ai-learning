from typing import Literal, Optional
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text-generation",
)


class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg", "neutral"] = Field(
        description="Return the sentiment as positive, negative, or neutral"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list",
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list",
    )
    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer",
    )


model = ChatHuggingFace(llm=llm)

structured_model = model.with_structured_output(
    Review.model_json_schema(),
    method="function_calling",
)

review_data = structured_model.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Rushikesh Gawae
"""
)

result = Review(**review_data)
print(result)
