
from sqlalchemy import create_engine
import pytest
from sqlmodel import SQLModel, Session
from .data_fixtures import dataset, prediction

@pytest.fixture
def seeded_database_url(tmp_path):
    db_path = tmp_path / "seeded_db.sqlite"
    return f"sqlite:///{db_path}"

@pytest.fixture
def base_engine(seeded_database_url):
    engine = create_engine(seeded_database_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    from sqlmodel import Session

    from chap_core.database.model_template_seed import seed_configured_models_from_config_dir

    with Session(engine) as session:
        seed_configured_models_from_config_dir(session)
    return engine


@pytest.fixture
def p_seeded_engine(base_engine, prediction):
    with Session(base_engine) as session:
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
    return base_engine

@pytest.fixture
def seeded_session(p_seeded_engine):
    with Session(p_seeded_engine) as session:
        yield session
