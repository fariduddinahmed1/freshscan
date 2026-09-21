"""Shared upload guards. Single copy used by all three upload routes."""
from fastapi import HTTPException, UploadFile

MAX_UPLOAD_MB = 5


def invalid(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=message)


async def read_upload_bytes(file: UploadFile, max_mb: int = MAX_UPLOAD_MB) -> bytes:
    data = await file.read()
    try:
        await file.close()
    except Exception:
        pass
    if not data:
        raise invalid("Empty upload")
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    return data
