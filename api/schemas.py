"""Request models. Same fields the API always accepted."""
from typing import Optional

from pydantic import BaseModel


class ShelfItem(BaseModel):
    shelf_number: str
    box_number: str
    item_name: str
    email: Optional[str] = None


class EmailConfig(BaseModel):
    sender_email: str
    sender_password: str
    recipient_email: str
