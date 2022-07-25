from typing import List, Optional

from pydantic import BaseModel, Field

from deployment.bff.app.v1.models.helper import (
    BaseIndexRequestModel,
    BaseSearchRequestModel,
)


# Request Model
class NowVideoIndexRequestModel(BaseIndexRequestModel):
    videos: List[str] = Field(
        default=...,
        description='List of Videos to index. Videos should be base64encoded in `utf-8` format',
    )


class NowVideoSearchRequestModel(BaseSearchRequestModel):
    video: str = Field(
        default=None,
        description='Video query. Video should be base64encoded in `utf-8` format',
    )


# Response Model
class NowVideoResponseModel(BaseModel):
    blob: Optional[str] = Field(
        description='Base64 encoded video in `utf-8` str format.'
    )
    uri: Optional[str] = Field(description='Uri of the video file.')


NowVideoResponseModel.update_forward_refs()
