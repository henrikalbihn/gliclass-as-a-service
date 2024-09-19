# `gliclass-as-a-service`

Scalable [GLiClass](https://github.com/Knowledgator/GLiClass) as a Service API.

[![LinkedIn][linkedin-shield]][linkedin-url]

## About

This project leverages FastAPI, Celery, and Redis to create a scalable and efficient system for generating and serving classification predictions on text data. The architecture is designed to handle high volumes of requests with optimal performance and reliability.

- FastAPI is used as the high-performance web framework, providing fast and asynchronous endpoints to receive text data and return predictions.
- Celery manages the asynchronous task execution, distributing the workload across multiple workers, ensuring that the system can handle large-scale processing demands.
- Redis serves a dual role: as a message broker between FastAPI and Celery, and as a caching layer to store task results for quick retrieval.
- Flower is integrated for real-time monitoring and management of Celery tasks, giving you visibility into the system's performance and task execution status.
- Locust is integrated for load testing the API.
- Streamlit is integrated for a user interface to the API.

### Built With

- [GLiClass](https://github.com/Knowledgator/GLiClass)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Celery](https://docs.celeryq.dev/en/stable/index.html#)
- [Redis](https://redis.io/)
- [Flower](https://flower.readthedocs.io/en/latest/)
- [Locust](https://locust.io/)
- [Streamlit](https://streamlit.io/)

## Getting Started

The application is build with docker-compose to create the various microservices. These include the FastAPI application itself, the Redis database and the Celery application. To build the Dockerfile for both FastAPI and Celery, use the following command:

```bash
# Copy the .env.example file to .env
cp .env.example .env

# Build the Docker image
docker build . -f Dockerfile.gliclass -t gliclass-service:latest

# Start the containers in detached mode
docker compose up -d
```

The following environment variables  needed in a ```.env``` file.
Leave as is for local testing.

```bash
REDIS_URL=redis://redis:6379/0
```

Open the browsers to see FastAPI's SwaggerUI, Locust, and Flower UI:

```bash
python scripts/open_browsers.py
```

![img](img/screenshot-swaggerui.jpeg)

Finally you can test the model:

```bash
./scripts/test-predict.sh
```

![img](img/screenshot-test-predict.jpeg)

```json
[
  [
    {
      "label": "Spanish",
      "score": 0.7295899391174316
    }
  ]
]
```

## User Interface

The UI is built with [Streamlit](https://streamlit.io/) and can be run with the following command:

```bash
./scripts/start-ui.sh
```

![img](img/screenshot-ui.jpeg)

*Note: The UI is an independent client (and thus has extra dependencies). It is not required to run the API.*

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-white.svg?
[linkedin-url]: https://linkedin.com/in/henrikalbihn
