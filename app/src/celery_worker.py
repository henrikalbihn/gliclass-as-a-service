import os

from celery import Celery

gliclass_app = Celery(
    "gliclass_app",
    broker=os.environ.get("REDIS_URL"),
    backend=os.environ.get("REDIS_URL"),
    include=["app.src.tasks"],
)
