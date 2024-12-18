import os

from sqlmodel import Session

from chap_core.database.database import engine


def get_session():
    with Session(engine) as session:
        yield session


def get_database_url():
    return os.getenv("CHAP_DATABASE_URL")
