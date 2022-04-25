from pydantic import BaseModel, Field


class Data(BaseModel):

    host: str = Field(
        default='localhost', description='Host address returned by the flow deployment.'
    )
    dataset: str = Field(
        default='deepfashion', description='Dataset used for indexing.'
    )

    class Config:
        case_sensitive = False
        validate_assignment = True
        arbitrary_types_allowed = True
