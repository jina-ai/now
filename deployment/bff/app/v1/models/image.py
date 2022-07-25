from typing import List, Optional

from pydantic import Field

from deployment.bff.app.v1.models.helper import (
    BaseIndexRequestModel,
    BaseSearchRequestModel,
    BaseSearchResponseModel,
)


# Request Model
class NowImageIndexRequestModel(BaseIndexRequestModel):
    images: List[str] = Field(
        default=...,
        description='List of Images to index. Images should be base64encoded in `utf-8` format',
    )


class NowImageSearchRequestModel(BaseSearchRequestModel):
    image: str = Field(
        default=None,
        description='Image query. Image should be base64encoded in `utf-8` format',
    )


# Response Model
class NowImageResponseModel(BaseSearchResponseModel):
    blob: Optional[str] = Field(
        description='Base64 encoded image in `utf-8` str format.'
    )
    uri: Optional[str] = Field(description='Uri of the image file.')


NowImageResponseModel.update_forward_refs()
