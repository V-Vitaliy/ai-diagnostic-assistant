from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeEngine
from typing import Any


class Base(AsyncAttrs, DeclarativeBase):

    __abstract__ = True

