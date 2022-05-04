from typing import Dict, Optional

from pydantic import BaseModel, Field

from now.bff.v1.models.helper import _NamedScore, _StructValueType


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
