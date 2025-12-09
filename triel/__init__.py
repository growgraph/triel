from importlib.metadata import version

from triel.client import (
    TrielClient,
    TrielClientError,
    TrielConnectionError,
    TrielServerError,
)

__version__ = version(__name__)

__all__ = [
    "TrielClient",
    "TrielClientError",
    "TrielConnectionError",
    "TrielServerError",
]
