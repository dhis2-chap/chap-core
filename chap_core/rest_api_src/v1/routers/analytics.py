from typing import List

import numpy as np
from pydantic import BaseModel, confloat

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlmodel import Session

from chap_core.api_types import EvaluationEntry, DataList, DataElement, PredictionEntry
from .dependencies import get_session
from chap_core.database.tables import BackTest, DataSet
import logging

router = APIRouter(prefix="/analytics", tags=["analytics"])

logger = logging.getLogger(__name__)


class EvaluationEntryRequest(BaseModel):
    backtest_id: int
    quantiles: List[confloat(ge=0, le=1)]


@router.get("/evaluation_entry", response_model=List[EvaluationEntry])
async def get_evaluation_entries(
        backtest_id: int,
        quantiles: List[float] = Query(...),
        session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    return [
        EvaluationEntry(period=forecast.period_id, orgUnit=forecast.region_id, quantile=q,
                        splitPeriod=forecast.last_seen_period_id,
                        value=np.quantile(forecast.values, q)) for forecast in backtest.forecasts for q in
        quantiles
    ]


@router.get("/prediction_entry/{prediction_id}", response_model=List[PredictionEntry])
def get_prediction_entries(prediction_id: int, session: Session = Depends(get_session)):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/actual_cases/{backtest_id}", response_model=DataList)
async def get_actual_cases(backtest_id: int, session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    logger.info(f"Backtest: {backtest}")
    data = session.get(DataSet, backtest.dataset_id)
    logger.info(f"Data: {data}")
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    data_list = [DataElement(pe=observation.period_id, ou=observation.region_id, value=float(observation.value) if not (
                np.isnan(observation.value) or observation.value is None) else None) for
                 observation in data.observations if observation.element_id == "disease_cases"]
    logger.info(f"DataList: {len(data_list)}")
    return DataList(
        featureId="disease_cases",
        dhis2Id="disease_cases",
        data=data_list
    )
