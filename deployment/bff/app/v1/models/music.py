from typing import List, Optional

from pydantic import Field

from deployment.bff.app.v1.models.helper import (
    BaseIndexRequestModel,
    BaseSearchRequestModel,
    BaseSearchResponseModel,
)


class NowMusicIndexRequestModel(BaseIndexRequestModel):
    songs: List[str] = Field(
        default=..., description='List of base64 encoded binary audio data to index.'
    )


class NowMusicSearchRequestModel(BaseSearchRequestModel):
    song: str = Field(
        default=None,
        description='Audio data query. Audio data should be base64encoded in `utf-8` format',
    )


class NowMusicResponseModel(BaseSearchResponseModel):
    blob: Optional[str] = Field(
        description='Matching song (base64encoded string `utf-8`) result.'
    )
    uri: Optional[str] = Field(description='Uri of the audio file.')


NowMusicResponseModel.update_forward_refs()
