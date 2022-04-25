from typing import Any, List

from fastapi import APIRouter

from now.bff.v1.models.data import Data as DataAPIModel

router = APIRouter()


# Index
@router.post("/index", response_model=DataAPIModel)
def index_data(data: DataAPIModel):
    pass


# Search
@router.get("/search", response_model=List[DataAPIModel])
def search_data(data: DataAPIModel) -> Any:
    """
    Retrieve images for a given search image.
    """
    pass
