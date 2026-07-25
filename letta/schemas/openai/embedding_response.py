from typing import Literal

from pydantic import BaseModel


class EmbeddingResponse(BaseModel):
    """OpenAI embedding response model: https://platform.openai.com/docs/api-reference/embeddings/object"""

    index: int  # the index of the embedding in the list of embeddings
    embedding: list[float]
    object: Literal["embedding"] = "embedding"
