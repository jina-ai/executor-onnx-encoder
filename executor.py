from typing import Optional

import onnxruntime
from jina import requests, DocumentArray, Executor
import numpy as np


class ONNXEncoder(Executor):
    """
    An executor that can be used to load embedding models in ONNX format
    and encode documents using the loaded model via the ONNX runtime.
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """
        Initialization

        :param model_path: The path to an embedding model in ONNX format.
        :param device: The device to use.
        :param batch_size: The batch size to use.
        """
        super().__init__(**kwargs)
        self._session = onnxruntime.InferenceSession(model_path)
        self._device = device
        self._batch_size = batch_size

    @requests
    def encode(self, docs: DocumentArray, **_) -> Optional[DocumentArray]:
        """Encode docs."""
        docs.blobs = docs.blobs.astype(np.float32)
        docs.embed(self._session, device=self._device, batch_size=self._batch_size, to_numpy=True)
        return docs
