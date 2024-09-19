"""
Main module to define the FastAPI application.
"""

from textwrap import dedent

from fastapi import FastAPI
from src import __app_name__, __version__
from src.redis import lifespan
from src.router import router

app = FastAPI(
    title=__app_name__,
    version=__version__,
    description=dedent(
        """
        ⭐ **GLiClass**: Generalist and Lightweight Model for Sequence Classification
        This is an efficient zero-shot classifier inspired by GLiNER work. It demonstrates the same performance as a cross-encoder while being more compute-efficient because classification is done at a single forward path.

        It can be used for topic classification, sentiment analysis and as a reranker in RAG pipelines.
        """
    ).strip(),
    summary="Classification API",
    lifespan=lifespan,
)

app.include_router(router)
