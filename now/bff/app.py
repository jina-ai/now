from fastapi import FastAPI

from now.bff.v1.api import api_router

app = FastAPI()


@app.get("/ping", summary="Check that the api service is operational")
def check_liveness():
    """
    Sanity check - this will let the user know that the service is operational.
    It is also used as part of the HEALTH CHECK.
    """
    return {"ping": "pong!"}


@app.get('/')
def read_root():
    return {'Hello': 'World!'}


app.include_router(api_router)
