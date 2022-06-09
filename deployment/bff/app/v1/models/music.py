from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from deployment.bff.app.v1.models.helper import (
    BaseRequestModel,
    _NamedScore,
    _StructValueType,
)


class NowMusicIndexRequestModel(BaseRequestModel):
    songs: List[str] = Field(
        default=..., description='List of base64 encoded ' 'binary audio data to index.'
    )


class NowMusicSearchRequestModel(BaseRequestModel):
    text: str = Field(default=None, description='Text query')
    song: str = Field(
        default=None,
        description='Audio data query. Audio data should be base64encoded in `utf-8` format',
    )
    limit: int = Field(default=10, description='Number of matching results to return')


class NowMusicResponseModel(BaseModel):
    id: str = Field(
        default=..., nullable=False, description='Id of the matching result.'
    )
    blob: Optional[str] = Field(
        description='Matching song (base64encoded string `utf-8`) result.'
    )
    scores: Optional[Dict[str, '_NamedScore']] = Field(
        description='Similarity score with respect to the query.'
    )
    tags: Optional[Dict[str, '_StructValueType']] = Field(
        description='Additional tags associated with the file.'
    )
    uri: Optional[str] = Field(description='Uri of the audio file.')
    text: Optional[str] = Field(description='Matching text result.')

    class Config:
        case_sensitive = False
        arbitrary_types_allowed = True


NowMusicResponseModel.update_forward_refs()
