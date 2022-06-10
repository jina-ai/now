from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from deployment.bff.app.v1.models.helper import (
    BaseIndexRequestModel,
    BaseRequestModel,
    _NamedScore,
    _StructValueType,
)


# Request Model
class NowTextIndexRequestModel(BaseIndexRequestModel):
    texts: List[str] = Field(default=..., description='List of Texts to index.')


class NowTextSearchRequestModel(BaseRequestModel):
    text: str = Field(default=None, description='Text query')
    limit: int = Field(default=10, description='Number of matching results to return')


# Response Model
class NowTextResponseModel(BaseModel):
    id: str = Field(
        default=..., nullable=False, description='Id of the matching result.'
    )
    text: Optional[str] = Field(description='Matching text result.')
    scores: Optional[Dict[str, '_NamedScore']] = Field(
        description='Similarity score with respect to the query.'
    )
    tags: Optional[Dict[str, '_StructValueType']] = Field(
        description='Additional tags associated with the file.'
    )

    class Config:
        case_sensitive = False
        arbitrary_types_allowed = True


NowTextResponseModel.update_forward_refs()
