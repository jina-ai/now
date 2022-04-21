from pydantic import BaseModel, Field


class Data(BaseModel):

    version: str = Field(default='v0.0.1', description='Api version to use.')
    host: str = Field(
        default='localhost', description='Host address returned by the flow deployment.'
    )
    menu: str = Field(default='text-to-image', description='Type of search to perform.')
    dataset: str = Field(
        default='deepfashion', description='Dataset used for indexing.'
    )

    class Config:
        allow_mutation = False
