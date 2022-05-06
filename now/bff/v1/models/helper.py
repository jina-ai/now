from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

_ProtoValueType = Optional[Union[bool, float, str, list, dict]]
_StructValueType = Union[
    _ProtoValueType, List[_ProtoValueType], Dict[str, _ProtoValueType]
]


class _NamedScore(BaseModel):
    value: Optional[float] = None


class BaseRequestModel(BaseModel):
    host: str = Field(
        default='localhost', description='Host address returned by the flow deployment.'
    )
    port: int = Field(default=31080, description='Port at which to connect.')

    class Config:
        allow_mutation = False
        case_sensitive = False
        arbitrary_types_allowed = True
