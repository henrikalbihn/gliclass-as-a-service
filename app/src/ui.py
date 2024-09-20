"""
Streamlit UI for the GLiClass package.

uv pip install -r dependencies/requirements-ui.txt
"""

import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import polars as pl
import streamlit as st
from httpx import AsyncClient
from loguru import logger
from PIL import Image

from app.src import __app_name__ as APP_NAME
from app.src import __author__
from app.src import __version__ as APP_VERSION
from app.src.models import PredictRequest

STATE = st.session_state
GITHUB_URL = "https://github.com/henrikalbihn/gliclass-as-a-service"
BACKEND_HOST = "http://localhost:8080"


class UserInterface:
    """User interface."""

    def __init__(
        self,
        tabs: list[str] = [
            "home",
            "about",
            "model",
            "settings",
        ],
    ) -> None:
        """Initialize the app.

        Args:
            tabs (list[str], optional): The tabs to render. Defaults to ["home", "about", "model", "settings"].
        """
        self.version = APP_VERSION
        self.STATE = STATE
        self.init_state()
        self.page_config()
        self.tabs = tabs
        self._tabs = {t: c for t, c in zip(self.tabs, st.tabs(self.tabs))}
        self.widgets = {k: getattr(self, k) for k in self.__dir__()}

    def init_session(self) -> None:
        """Initialize session.

        Args:
            session_logging (bool, optional): Whether to initialize session logging. Defaults to False.

        Returns:
            None
        """
        self.STATE.session_id = str(uuid4())
        self.STATE.session_start = datetime.utcnow()
        self.STATE.session_end = None
        self.STATE.session_duration = None
        self.STATE.random_seed = 42
        self.STATE.tz = "America/Los_Angeles"
        logger.info(
            f"Initializing session [{self.STATE.session_id}] @ [{str(self.STATE.session_start)}]..."
        )

    def init_state(self) -> None:
        """Initialize session state."""
        self.init_session()

        self.STATE.data = {}
        self.STATE.data["model_status"] = self.check_model_status()

        self.STATE.plots = {}

    def page_config(
        self,
        favicon: str = "⭐",
        sidebar_state: str = "collapsed",
        help_url: str = GITHUB_URL,
        bugs_url: str = f"{GITHUB_URL}/issues",
        about_url: str = f"{GITHUB_URL}/blob/master/README.md",
    ) -> None:
        """Page configuration.

        Args:
            favicon (str, optional): Favicon. Defaults to "⭐".
            sidebar_state (str, optional): Sidebar state. Defaults to "collapsed".
            help_url (str, optional): Help URL.
            bugs_url (str, optional): Bugs URL.
            about_url (str, optional): About URL.
        """
        st.set_page_config(
            APP_NAME,
            page_icon=favicon,
            layout="wide",
            initial_sidebar_state=sidebar_state,  # "expanded",
            menu_items={
                "Get help": help_url,
                "Report a bug": bugs_url,
                "About": about_url,
            },
        )

    def sidebar(self) -> None:
        """Sidebar."""
        about = f"""`GliClass`: Generalist and Lightweight Model for Sequence Classification. This is an efficient zero-shot classifier inspired by [GLiNER](https://github.com/urchade/GLiNER/tree/main) work. It demonstrates the same performance as a cross-encoder while being more compute-efficient because classification is done at a single forward path. It can be used for topic classification, sentiment analysis and as a reranker in RAG pipelines."""
        contact = "[Henrik Albihn, MS](http://github.com/henrikalbihn)"
        st.sidebar.header("**GliClass ⭐**", divider="rainbow")
        st.sidebar.subheader("About")
        st.sidebar.write(about)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Contact")
        st.sidebar.write(contact)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Legal")
        st.sidebar.markdown("---")

    def home(self) -> None:
        """Title & divider."""
        readme = Path(__file__).parent.parent.parent / "README.md"
        text = readme.read_text()

        # Split the text into parts based on image tags
        parts = text.split("![img](")

        # Display the header (text before the first image)
        st.markdown(parts[0])

        # Process each image and subsequent text
        for part in parts[1:]:
            img_path, body = part.split(")", maxsplit=1)
            img_path = Path(__file__).parent.parent.parent / img_path

            img = Image.open(img_path)
            st.image(img, width=1000)
            st.markdown(body)

    def settings(self) -> None:
        """Settings widget"""
        with st.container():
            c0, c1 = st.columns([0.5, 0.5])

            with c0:
                st.metric(
                    "Model Status",
                    value=self.STATE.data["model_status"]["message"],
                    delta=(
                        "✅"
                        if self.STATE.data["model_status"]["message"] == "Ok"
                        else "❌"
                    ),
                )
            with c1:
                docs_url = f"{BACKEND_HOST}/docs"
                st.markdown(
                    f"Documentation: [{docs_url}]({docs_url})",
                    unsafe_allow_html=True,
                )

    def about(self) -> None:
        """About widget"""
        about_md = f"""
```yaml
App Details:
    Name:\t{APP_NAME}
    Author:\t{__author__}
    Version:\t{self.version}
```
"""
        st.markdown(about_md)

    def predict(
        self,
        targets: list[str] = ["hello world"],
        labels: list[str] = ["O"],
        classification_type: str = "single-label",
        batch_size: int = 12,
    ) -> None:
        """Predict."""

        async def __predict(
            targets: list[str] = ["hello world"],
            labels: list[str] = ["O"],
        ) -> None:
            """Predict."""

            async with AsyncClient() as client:
                data = PredictRequest(
                    inputs=targets,
                    labels=labels,
                    classification_type=classification_type,
                    batch_size=batch_size,
                ).model_dump()
                r = await client.post(
                    f"{BACKEND_HOST}/predict",
                    json=data,
                    timeout=30,
                )
                data = r.json()
                task_id = data["task_id"]
                status = data["status"]

                while status == "Processing":
                    logger.info(f"Processing [{task_id}]...")
                    st.toast("Processing...", icon="🔄")
                    await asyncio.sleep(1)
                    r = await client.get(
                        f"{BACKEND_HOST}/result/{task_id}",
                        timeout=30,
                    )
                    data = r.json()
                    status = data["status"]
                    logger.info(f"Status [{task_id}]: {status}")

                if status == "Success":
                    st.toast("Done!", icon="🚀")
                    logger.info(f"Predictions [{task_id}]: {data}")
                    return data

        return asyncio.run(__predict(targets, labels))

    def check_model_status(self) -> dict:
        """Check model status."""

        async def __check_health() -> dict:
            """Check health."""
            async with AsyncClient() as client:
                r = await client.get(
                    f"{BACKEND_HOST}/health",
                    timeout=30,
                )
                data = r.json()
                return data

        return asyncio.run(__check_health())

    def model(self) -> None:
        """Model widget"""

        with st.container():
            st.write("## GliClass ⭐")

            c0, c1, c2 = st.columns([0.3, 0.3, 0.2])
            with c0:
                default_labels = "travel,dreams,sport,science,politics"
                labels = st.text_input(
                    "labels (comma-separated)",
                    value=default_labels,
                )
                labels = labels.split(",")
            with c1:
                classification_type = st.selectbox(
                    "classification type",
                    options=["single-label", "multi-label"],
                    index=0,
                )

            with c2:
                submit = st.button("Submit")

            default_target = "One day I will see the world!"

            target = st.text_area(
                "target",
                value=default_target,
                height=150,
            )

            if submit:
                with st.spinner("Loading..."):
                    data = self.predict(
                        targets=[target],
                        labels=labels,
                        classification_type=classification_type,
                    )

                st.write("### Results")

                # st.json(data)

                label = data["result"]["predictions"][0][0]["label"]
                score = data["result"]["predictions"][0][0]["score"]

                st.metric(
                    f"task_id: [{data['task_id']}]",
                    value=f"prediction: [{label}]",
                    delta=f"score: {score*100:.2f}%",
                )

    def navbar(self) -> None:
        """Nav bar"""
        for k, v in self._tabs.items():
            # render tab component
            with v:
                # render tab title
                st.subheader(k.title(), divider="rainbow")
                self.widgets[k]()

    def main_page(self) -> None:
        """Main page."""
        self.navbar()
        self.sidebar()


def main() -> None:
    """Main function."""
    ui = UserInterface()
    ui.main_page()


if __name__ == "__main__":
    main()
