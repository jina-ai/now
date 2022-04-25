from fastapi import APIRouter

from now.bff.v1.routers import image2image as bff_img2img_router
from now.bff.v1.routers import image2text as bff_img2txt_router
from now.bff.v1.routers import text2image as bff_txt2img_router

v1_router = APIRouter()
v1_router.include_router(
    bff_txt2img_router.router, prefix='/text_to_image', tags=['Text-to-Image']
)
v1_router.include_router(
    bff_img2txt_router.router, prefix='/image_to_text', tags=['Image-to-Text']
)
v1_router.include_router(
    bff_img2img_router.router, prefix='/image_to_image', tags=['Image-to-Image']
)
