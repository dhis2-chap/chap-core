from pydantic import BaseModel
from sqlmodel import select

from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session

from .dependencies import get_session
from chap_core.rest_api_src.celery_tasks import CeleryPool
from chap_core.database.tables import BackTest, DataSet, BackTestMetric, BackTestForecast
import chap_core.rest_api_src.db_worker_functions as wf

router = APIRouter(prefix="/crud", tags=["crud"])
worker = CeleryPool()


class BackTestCreate(BaseModel):
    dataset_id: int
    estimator_id: str


class BackTestRead(BackTestCreate):
    id: int

    class Config:
        orm_mode = True  # Enable compatibility with ORM models


class BackTestFull(BackTestRead):
    metrics: list[BackTestMetric]
    forecasts: list[BackTestForecast]


@router.get("/backtest/{backtest_id}", response_model=BackTestFull)
async def get_backtest(backtest_id: int, session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    return backtest


@router.post("/backtest", response_model=BackTestCreate)
async def create_backtest(backtest: BackTestCreate, session: Session = Depends(get_session)):
    worker.queue(wf.run_backtest,
                 backtest.estimator_id, backtest.dataset_id, 12, 2, 1)
    return backtest


@router.get("/backtest", response_model=list[BackTestRead])
async def get_backtests(session: Session = Depends(get_session)):
    backtests = session.exec(select(BackTest)).all()
    return backtests


@router.get('/dataset/{dataset_id}', response_model=DataSet)
async def get_dataset(dataset_id: int, session: Session = Depends(get_session)):
    dataset = session.get(DataSet, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


class DataSetRead(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True  # Enable compatibility with ORM models


@router.get('/datasets', response_model=list[DataSetRead])
async def get_datasets(session: Session = Depends(get_session)):
    datasets = session.exec(select(DataSet)).all()
    return datasets
