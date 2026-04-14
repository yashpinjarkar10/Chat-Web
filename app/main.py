from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    return app


app = create_app()
