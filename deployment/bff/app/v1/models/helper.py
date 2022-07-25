from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

_ProtoValueType = Optional[Union[bool, float, str, list, dict]]
_StructValueType = Union[
    _ProtoValueType, List[_ProtoValueType], Dict[str, _ProtoValueType]
]


class _NamedScore(BaseModel):
    value: Optional[float] = None


# Base Request
class BaseRequestModel(BaseModel):
    host: str = Field(
        default='localhost', description='Host address returned by the flow deployment.'
    )
    port: int = Field(default=31080, description='Port at which to connect.')
    jwt: str = Field(default=..., description='User info obtained from the hubble')

    class Config:
        allow_mutation = False
        case_sensitive = False
        arbitrary_types_allowed = True


# Index extending Base Request
class BaseIndexRequestModel(BaseRequestModel):
    tags: List[Dict[str, Any]] = Field(
        default={}, description='List of tags of the documents to be indexed.'
    )


# Search extending Base Request
class BaseSearchRequestModel(BaseRequestModel):
    limit: int = Field(default=10, description='Number of matching results to return')


# Base Request for Search
class BaseSearchResponseModel(BaseModel):
    id: str = Field(
        default=..., nullable=False, description='Id of the matching result.'
    )
    scores: Optional[Dict[str, '_NamedScore']] = Field(
        description='Similarity score with respect to the query.'
    )
    tags: Optional[Dict[str, '_StructValueType']] = Field(
        description='Additional tags associated with the file.'
    )

    class Config:
        case_sensitive = False
        arbitrary_types_allowed = True
