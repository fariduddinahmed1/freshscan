"""Upload guard tests. Real UploadFile, no mocks."""
import asyncio
import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from core.errors import read_upload_bytes


def _upload(data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename="test.jpg")


def test_empty_bytes_422():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(read_upload_bytes(_upload(b"")))
    assert exc.value.status_code == 422


def test_oversize_413():
    big = b"x" * (5 * 1024 * 1024 + 1)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(read_upload_bytes(_upload(big)))
    assert exc.value.status_code == 413


def test_small_valid_passthrough():
    assert asyncio.run(read_upload_bytes(_upload(b"abc"))) == b"abc"
