"""
Connector implementations.
Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import logging

T = TypeVar("T")
logger = logging.getLogger(__name__)

class Connector(ABC, nn.Module):
    """Abstract base class defining the connector interface.

    Subclasses must implement:
    - forward()
    - get_input_dim():
    - get_output_dim():
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._built = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input tensor to output tensor."""
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return expected input dimension."""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output dimension."""
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Template method - handles common preprocessing/postprocessing."""
        output = self.forward(x)
        return output


### Connector implementations
class LinearConnector(Connector):
    """Simple linear projection connector, inspired by SLAM-ASR paper"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(input_dim, output_dim)
        self.projection = nn.Linear(input_dim, output_dim, dtype=dtype)
        self.dropout = (
            nn.Dropout(dropout, dtype=dtype) if dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(x))

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


class TransformerConnector(Connector):
    """Transformer encoder connector module"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(input_dim, output_dim)
        self.proj = nn.Linear(input_dim, 1280, dtype=dtype)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280,
            nhead=20,
            dim_feedforward=8192,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=dtype,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.proj2 = nn.Linear(1280, output_dim, dtype=dtype)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.transformer(x)
        x = self.proj2(x)
        return x

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


class QFormerConnector(Connector):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_query_tokens: int = 32,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 4,
        cross_attention_every: int = 2,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(input_dim, output_dim)
        logger.info(f"⚠⚠⚠ QFormer is used only for experimantal usage ⚠⚠⚠")
        logger.info(f"QFormer has degraded performance")
        self.num_query_tokens = num_query_tokens
        self.cross_attention_every = cross_attention_every
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_dim, dtype=dtype)
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim, dtype=dtype)

        self.layers = nn.ModuleList(
            [
                self._create_layer(
                    hidden_dim,
                    num_heads,
                    dropout,
                    dtype,
                    has_cross_attn=(i % cross_attention_every == 0),
                )
                for i in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim, dtype=dtype)
        self.norm = nn.RMSNorm(hidden_dim, dtype=dtype)

        nn.init.normal_(self.query_tokens, std=0.02)

    def _create_layer(self, hidden_dim, num_heads, dropout, dtype, has_cross_attn):
        layer = nn.ModuleDict(
            {
                "self_attn": nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                    dtype=dtype,
                ),
                "self_norm": nn.RMSNorm(hidden_dim, dtype=dtype),
                "ffn_norm": nn.RMSNorm(hidden_dim, dtype=dtype),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim, dtype=dtype),
                    nn.Dropout(dropout),
                ),
            }
        )
        if has_cross_attn:
            layer["cross_attn"] = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
                dtype=dtype,
            )
            layer["cross_norm"] = nn.RMSNorm(hidden_dim, dtype=dtype)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.input_proj(x)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            q_norm = layer["self_norm"](queries)
            self_out, _ = layer["self_attn"](q_norm, q_norm, q_norm)
            queries = queries + self_out
            if "cross_attn" in layer:
                q_norm = layer["cross_norm"](queries)
                attn_out, _ = layer["cross_attn"](q_norm, x, x)
                queries = queries + attn_out
            q_norm = layer["ffn_norm"](queries)
            queries = queries + layer["ffn"](q_norm)
        queries = self.norm(queries)
        return self.output_proj(queries)

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


class ConnectorFactory:
    """Factory to create connector instances based on configuration."""

    # Add your connectors here
    _connectors = {
        "linear": LinearConnector,
        "transformer": TransformerConnector,
        "qformer": QFormerConnector,
    }

    @classmethod
    def create(cls, connector_type: str, **kwargs) -> Connector:
        if connector_type not in cls._connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        return cls._connectors[connector_type](**kwargs)
