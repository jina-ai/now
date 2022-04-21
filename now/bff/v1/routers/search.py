from typing import Any, List

from fastapi import APIRouter

from now.bff.v1.models.data import Data as DataAPIModel

router = APIRouter()


# Index
@router.post("/index", response_model=DataAPIModel)
def create_experiment(
    exp: Data,
):
    pass


# Search
@router.get("/search", response_model=List[DataAPIModel])
def search(
    *,
    skip: int = 0,
    limit: int = 100,
    user_id: DataAPIModel,
) -> Any:
    """
    Retrieve experiments for a user.
    """
    pass


@router.get("/search", response_model=DataAPIModel)
def read_experiment(
    *,
    experiment_id: int,
    user_id: int,
) -> Any:
    """
    Read single experiment by experiment_id.
    """
    pass
