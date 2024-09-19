"""
Celery tasks for the GLiClass service.
"""

from typing import Any

from celery import Task
from loguru import logger

from .celery_worker import gliclass_app
from .gliclass import ClassificationModel


class PredictTask(Task):
    """Predict Task"""

    abstract = True

    def __init__(self) -> None:
        """Init predict task."""
        super().__init__()
        logger.debug("Init predict task.")
        self.model: ClassificationModel = None

    def __call__(self, *args, **kwargs) -> Any:
        """Call predict task."""
        if not self.model:
            # Protecting this in the __call__ method to avoid
            # loading the model in the fastapi server process.
            self.model = ClassificationModel()
        return self.run(*args, **kwargs)


@gliclass_app.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
    name="gliclass.predict",
)
def predict(
    self,
    inputs: list[str],
    labels: list[str],
    classification_type: str = "single-label",
    batch_size: int = 12,
) -> list[list[dict[str, Any]]]:
    """Predict task.

    Args:
        self (Task): Task instance.
        inputs (list[str]): List of inputs.
        labels (list[str]): List of labels.
        classification_type (str, optional): Classification type. Defaults to "single-label".
        batch_size (int, optional): Batch size. Defaults to 12.

    Returns:
        list[list[dict[str, Any]]]: List of results.
    """
    try:
        logger.info(f"Predicting [{len(inputs):,}] inputs.")
        results = self.model.batch_predict(
            targets=inputs,
            labels=labels,
            classification_type=classification_type,
            batch_size=batch_size,
        )
        logger.info("Prediction complete.")
        return results
    except Exception as e:
        logger.exception("Prediction failed.", e)
        raise e
