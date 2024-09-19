"""
https://github.com/Knowledgator/GLiClass/tree/main
https://huggingface.co/knowledgator/gliclass-small-v1.0
https://github.com/Knowledgator/GLiClass/blob/main/finetuning.ipynb
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Literal

import GPUtil
import torch
from flupy import flu

# from gliclass import GLiClass
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from loguru import logger

# from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import random

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from gliclass import GLiClassModel, ZeroShotClassificationPipeline

# from gliclass.data_processing import DataCollatorWithPadding, GLiClassDataset
# from gliclass.training import Trainer, TrainingArguments
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer

try:
    # Try to enable hf_transfer if available
    # - pip install huggingface_hub[hf_transfer])
    # - hf_transfer is a power-user that enables faster downloads from the Hub
    # https://github.com/GLiClass-project/GLiClass/issues/2907
    # https://github.com/huggingface/hf_transfer
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass

MODELS = {
    "GLiClass-S": "knowledgator/gliclass-small-v1.0",
    "GLiClass-M": "knowledgator/gliclass-medium-v1.0",
    "GLiClass-L": "knowledgator/gliclass-large-v1.0",
}
"""
available models:
  - https://huggingface.co/urchade
  - https://huggingface.co/collections/numind/nunerzero-zero-shot-ner-662b59803b9b438ff56e49e2
"""

DEFAULT_MODEL = "GLiClass-S"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(predicts, true_labels):
    micro = f1_score(true_labels, predicts, average="micro")
    macro = f1_score(true_labels, predicts, average="macro")
    weighted = f1_score(true_labels, predicts, average="weighted")
    return {"micro": micro, "macro": macro, "weighted": weighted}


def get_train_dataset(dataset, N, label_column="label"):
    ids = []
    label2count = {}
    train_dataset = dataset.shuffle(seed=41)
    for id, example in enumerate(train_dataset):
        if example[label_column] not in label2count:
            label2count[example[label_column]] = 1
        elif label2count[example[label_column]] >= N:
            continue
        else:
            label2count[example[label_column]] += 1
        ids.append(id)
    return train_dataset.select(ids)


def prepare_dataset(
    dataset,
    classes=None,
    text_column="text",
    label_column="label",
    split=None,
):
    if "test" in dataset:
        test_dataset = dataset["test"]
    elif isinstance(dataset, Dataset):
        test_dataset = dataset
    else:
        test_dataset = dataset["train"]

    if classes is None:
        classes = test_dataset.features[label_column].names
        if split is not None:
            classes = [" ".join(class_.split(split)) for class_ in classes]

    texts = test_dataset[text_column]

    true_labels = test_dataset[label_column]

    print(classes)
    if type(test_dataset[label_column][0]) == int:
        true_labels = [classes[label] for label in true_labels]

    return texts, classes, true_labels


def prepare_dataset_for_training(
    train_dataset,
    classes,
    text_column="text",
    label_column="label",
):
    id2class = {id: class_ for id, class_ in enumerate(classes)}
    dataset = []
    for example in train_dataset:
        label = example[label_column]
        if type(label) == int:
            label = id2class[label]
        item = {
            "text": example[text_column],
            "all_labels": classes,
            "true_labels": [label],
        }
        dataset.append(item)
    random.shuffle(dataset)
    return dataset


class ClassificationModel:
    """Classification model."""

    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        local_model_path: str = None,
        overwrite: bool = False,
        # train_config: dict = TRAIN_CONFIG,
    ) -> None:
        """Initialize the ClassificationModel.

        Args:
            name: The model name.
            local_model_path: The local model path.
            overwrite: Whether to overwrite the model path.
            train_config: The training config.
        """
        if name not in MODELS:
            raise ValueError(f"Invalid model name: {name}")
        # Define the model ID
        self.model_id: str = MODELS[name]

        # Create a models directory
        workdir = Path.cwd() / "models"
        workdir.mkdir(parents=True, exist_ok=True)
        if local_model_path is None:
            local_model_path = Path(name).resolve()
        else:
            local_model_path = (workdir / local_model_path).resolve()
        if local_model_path.exists():
            import warnings

            warnings.warn(f"Model path already exists: {str(local_model_path)}")
            if overwrite is False:
                raise ValueError(f"Model path already exists: {str(local_model_path)}")

        # Define the local model path
        self.local_model_path: Path = local_model_path

        # Use GPU acceleration if available
        self.device: str = DEVICE
        logger.info(f"Device: [{self.device}]")

        # Define the hyperparameters in a config variable
        # self.train_config: SimpleNamespace = SimpleNamespace(**train_config)

        # Init empty for lazy loading of model weights
        self.model: GLiClassModel = None
        self.tokenizer: AutoTokenizer = None
        self.pipeline: ZeroShotClassificationPipeline = None

    def __load_model_remote(self) -> None:
        """Actually load the model.

        Args:
          model_id: The model ID.
        """
        self.model = GLiClassModel.from_pretrained(self.model_id)  # .to(device)

    def __load_model_local(self) -> None:
        """Load the model from a local path.

        Args:
          model_path: The model path.
        """
        try:
            local_model_path = str(self.local_model_path.resolve())
            self.model = GLiClassModel.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
        except Exception as e:
            logger.exception("Failed to load model from local path.", e)

    def load(self, mode: Literal["local", "remote", "auto"] = "auto") -> None:
        """Load the model.

        Args:
          model_id: The model ID.
        """
        if self.model is None:
            if mode == "local":
                self.__load_model_local()
            elif mode == "remote":
                self.__load_model_remote()
            elif mode == "auto":
                local_model_path = str(self.local_model_path.resolve())
                if Path(local_model_path).exists():
                    self.__load_model_local()
                else:
                    self.__load_model_remote()
            else:
                raise ValueError(f"Invalid mode: {mode}")
            GPUtil.showUtilization()
            logger.info(
                f"Loaded model: [{self.model_id}] | N Params: [{self.model_param_count}] | [{self.model_size_in_mb}]"
            )
        else:
            logger.warning("Model already loaded.")

        logger.info(f"Moving model weights to: [{self.device}]")
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @property
    def model_size_in_bytes(self) -> int:
        """Returns the approximate size of the model parameters in bytes."""
        total_size = 0
        for param in self.model.parameters():
            # param.numel() returns the total number of elements in the parameter,
            # param.element_size() returns the size in bytes of an individual element.
            total_size += param.numel() * param.element_size()
        return total_size

    @property
    def model_param_count(self) -> str:
        """Returns the number of model parameters in billions."""
        return f"{sum(p.numel() for p in self.model.parameters()) / 1e9:,.2f} B"

    @property
    def model_size_in_mb(self) -> str:
        """Returns the string repr of the model parameter size in MB."""
        return f"{self.model_size_in_bytes / 1024**2:,.2f} MB"

    # TODO: implement training
    # def train(
    #     self,
    #     train_data: list[dict[str, str]],
    #     eval_data: dict[str, list[Any]] = None,
    # ) -> None:
    #     """Train the GLiClass model.

    #     Args:
    #       model: The GLiClass model.
    #       config: The hyperparameters.
    #       train_data: The training data.
    #       eval_data: The evaluation data.
    #     """
    #     if self.model is None:
    #         self.load()

    #     GPUtil.showUtilization()

    #     # Set sampling parameters from config
    #     self.model.set_sampling_params(
    #         max_types=self.train_config.max_types,
    #         shuffle_types=self.train_config.shuffle_types,
    #         random_drop=self.train_config.random_drop,
    #         max_neg_type_ratio=self.train_config.max_neg_type_ratio,
    #         max_len=self.train_config.max_len,
    #     )

    #     self.model.train()

    #     # Initialize data loaders
    #     self.train_loader = self.model.create_dataloader(
    #         train_data,
    #         batch_size=self.train_config.train_batch_size,
    #         shuffle=True,
    #     )

    #     # Optimizer
    #     self.optimizer = self.model.get_optimizer(
    #         self.train_config.lr_encoder,
    #         self.train_config.lr_others,
    #         self.train_config.freeze_token_rep,
    #     )

    #     pbar = tqdm(range(self.train_config.num_steps))

    #     if self.train_config.warmup_ratio < 1:
    #         num_warmup_steps = int(
    #             self.train_config.num_steps * self.train_config.warmup_ratio
    #         )
    #     else:
    #         num_warmup_steps = int(self.train_config.warmup_ratio)

    #     self.scheduler = get_cosine_schedule_with_warmup(
    #         self.optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=self.train_config.num_steps,
    #     )

    #     iter_train_loader = iter(self.train_loader)

    #     for step in pbar:
    #         try:
    #             x = next(iter_train_loader)
    #         except StopIteration:
    #             iter_train_loader = iter(self.train_loader)
    #             x = next(iter_train_loader)

    #         for k, v in x.items():
    #             if isinstance(v, torch.Tensor):
    #                 x[k] = v.to(self.device)

    #         loss = self.model(x)  # Forward pass

    #         # Check if loss is nan
    #         if torch.isnan(loss):
    #             continue

    #         loss.backward()  # Compute gradients
    #         self.optimizer.step()  # Update parameters
    #         self.scheduler.step()  # Update learning rate schedule
    #         self.optimizer.zero_grad()  # Reset gradients

    #         description = f"step: {step} | epoch: {step // len(self.train_loader)} | loss: {loss.item():.2f}"
    #         pbar.set_description(description)

    #         if (step + 1) % self.train_config.eval_every == 0:

    #             self.model.eval()

    #             if eval_data is not None:
    #                 results, f1 = self.model.evaluate(
    #                     eval_data["samples"],
    #                     flat_ner=True,
    #                     threshold=0.5,
    #                     batch_size=12,
    #                     entity_types=eval_data["entity_types"],
    #                 )

    #                 logger.debug(f"Step={step} | F1 [{f1:.2f}]\n{results}")

    #             checkpoint_dir = Path(self.train_config.save_directory)
    #             if not checkpoint_dir.exists():
    #                 checkpoint_dir.mkdir(exist_ok=True, parents=True)

    #             self.model.save_pretrained(
    #                 f"{self.train_config.save_directory}/finetuned_{step}"
    #             )

    #             self.model.train()

    #     GPUtil.showUtilization()

    #     logger.success("Training complete!")

    def batch_predict(
        self,
        targets: List[str],
        labels: List[str],
        classification_type: Literal["single-label", "multi-label"] = "single-label",
        batch_size: int = 12,
    ) -> List[List[str]]:
        """Batch predict.

        Args:
          targets: The targets.
          labels: The labels.
          threshold: The threshold.
          batch_size: The batch size.

        Returns:
          The predictions.
        """

        if self.model is None:
            self.load()

        self.model.eval()
        predictions = []
        if self.pipeline is None:
            self.pipeline = ZeroShotClassificationPipeline(
                self.model,
                self.tokenizer,
                classification_type=classification_type,
                device=self.device,
            )

        predictions = self.pipeline(
            targets,
            labels,
            batch_size=batch_size,
        )
        # predictions = [result[0]["label"] for result in predictions]
        return predictions

    def save(self, file_name: str) -> None:
        """Save the model to a file.

        Args:
          file_name: The file name.
        """
        self.model.save_pretrained(file_name)

    def test(self) -> None:
        """Test the model."""
        targets = ["hello John, your reservation is at 6pm"]
        labels = ["greeting", "reservation"]
        predictions = self.batch_predict(targets, labels)
        logger.info(predictions)
        if predictions[0] == labels[0]:
            logger.success("Test passed!")
        else:
            logger.error("Test failed!")