from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash", task="text-generation"
)

schema = {
    "title": "MovieReview",
    "description": "Extract structured information from a movie review",
    "type": "object",
    "properties": {
        "movie_name": {"type": "string", "description": "Name of the movie"},
        "rating": {
            "type": "integer",
            "description": "Movie rating given by the reviewer",
        },
        "summary": {"type": "string", "description": "Brief summary of the review"},
        "pros": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Positive aspects mentioned in the review",
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Negative aspects mentioned in the review",
        },
    },
    "required": ["movie_name", "rating", "summary", "pros", "cons"],
}

model = ChatHuggingFace(llm=llm)
output = model.with_structured_output(schema)
result = output.invoke(
    """   Analyze this movie review and extract the movie information:
"I watched Interstellar last night. The movie was visually stunning and the story was emotionally powerful. Matthew McConaughey's performance was excellent. The only downside was that some parts of the story were difficult to understand. Overall, I would give it 9 out of 10."""
)

print(result)
