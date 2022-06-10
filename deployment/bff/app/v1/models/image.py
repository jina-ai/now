from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from deployment.bff.app.v1.models.helper import (
    BaseIndexRequestModel,
    BaseRequestModel,
    _NamedScore,
    _StructValueType,
)


# Request Model
class NowImageIndexRequestModel(BaseIndexRequestModel):
    images: List[str] = Field(
        default=...,
        description='List of Images to index. Images should be base64encoded in `utf-8` format',
    )


class NowImageSearchRequestModel(BaseRequestModel):
    image: str = Field(
        default=None,
        description='Image query. Image should be base64encoded in `utf-8` format',
    )
    limit: int = Field(default=10, description='Number of matching results to return')


# Response Model
class NowImageResponseModel(BaseModel):
    id: str = Field(
        default=..., nullable=False, description='Id of the matching result.'
    )
    blob: Optional[str] = Field(
        description='Base64 encoded image in `utf-8` str format.'
    )
    uri: Optional[str] = Field(description='Uri of the image file.')
    scores: Optional[Dict[str, '_NamedScore']] = Field(
        description='Similarity score with respect to the query.'
    )
    tags: Optional[Dict[str, '_StructValueType']] = Field(
        description='Additional tags associated with the file.'
    )

    class Config:
        case_sensitive = False
        arbitrary_types_allowed = True


NowImageResponseModel.update_forward_refs()
