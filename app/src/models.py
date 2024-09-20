"""
Pydantic models for the API.
"""

from textwrap import dedent

from pydantic import BaseModel


class Task(BaseModel):
    """Celery task representation"""

    task_id: str
    status: str


class PredictResponse(BaseModel):
    """Predict Response"""

    task_id: str
    status: str
    result: dict


class PredictRequest(BaseModel):
    """Request body"""

    inputs: list[str] = [
        dedent(
            """
        One day I will see the world!
        """
        ).strip(),
    ]
    labels: list[str] = [
        "travel",
        "dreams",
        "sport",
        "science",
        "politics",
    ]  # This is just a default, can be anything the user wants

    classification_type: str = "single-label"
    batch_size: int = 12
    threshold: float = 0.5
