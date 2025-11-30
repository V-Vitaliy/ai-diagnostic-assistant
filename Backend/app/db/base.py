from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeEngine
from typing import Any


class Base(AsyncAttrs, DeclarativeBase):


    __abstract__ = True


    metadata: Any

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)