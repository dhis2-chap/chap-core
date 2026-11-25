"""Deprecated alert routes, from before alerts were renamed to signals (CLIM-1560).

Each route translates the old names and calls the signal or threshold-policy route
that replaced it, so the two cannot drift apart. Remove this module, and the alert
shapes in `signal_tables`, once no client calls these paths.
"""

from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from sqlmodel import Field, Session

from chap_core.auth.permissions import Permission
from chap_core.database.base_tables import DBModel, UtcDatetime
from chap_core.database.signal_tables import (
    AlertLevel,
    AlertPolicyRead,
    Signal,
    SignalBase,
    SignalReleaseStatus,
    ThresholdPolicy,
    ThresholdPolicyBase,
)
from chap_core.rest_api.authorization import require
from chap_core.rest_api.data_models import (
    DataBaseResponse,
    SignalCreate,
    SignalIdsResponse,
    SignalReleaseStatusRequest,
    SignalsCreate,
    ThresholdPolicyCodesUpdate,
    ThresholdPolicyCreate,
)
from chap_core.rest_api.v1.routers import crud
from chap_core.rest_api.v1.routers.dependencies import get_session

router = APIRouter(prefix="/crud", tags=["Alerts (deprecated)"], deprecated=True)


class AlertApproval(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"


_RELEASE_STATUS = {
    AlertApproval.PENDING: SignalReleaseStatus.PENDING,
    AlertApproval.APPROVED: SignalReleaseStatus.RELEASED,
    AlertApproval.DECLINED: SignalReleaseStatus.WITHHELD,
}
_APPROVAL = {status: approval for approval, status in _RELEASE_STATUS.items()}


class AlertCreate(SignalBase):
    threshold_policy_id: int = Field(alias="alertPolicyId")


class AlertsCreate(DBModel):
    alerts: list[AlertCreate] = Field(min_length=1)


class AlertRead(AlertCreate):
    id: int
    created: UtcDatetime | None
    release_status: AlertApproval = Field(alias="approved")
    reviewed_by: str | None = Field(alias="approvedBy")
    reviewed_at: UtcDatetime | None = Field(alias="approvedAt")


class AlertApprovalRequest(DBModel):
    alert_ids: list[int] = Field(min_length=1)
    approved: AlertApproval
    approved_by: str


class AlertPolicyCreate(ThresholdPolicyBase):
    no_signal_output_code: str = Field(default="0", min_length=1, alias="noAlertOutputCode")
    levels: list[AlertLevel] = Field(min_length=1)  # type: ignore[assignment]


class AlertPolicyCodesUpdate(ThresholdPolicyCodesUpdate):
    no_signal_output_code: str | None = Field(default=None, min_length=1, alias="noAlertOutputCode")


def _alert(signal: Signal) -> AlertRead:
    return AlertRead.model_validate({**signal.model_dump(), "release_status": _APPROVAL[signal.release_status]})


def _alert_policy(policy: ThresholdPolicy) -> AlertPolicyRead:
    return AlertPolicyRead.model_validate(policy.model_dump())


@router.post(
    "/alerts",
    dependencies=[require(Permission.WRITE_SIGNALS)],
    response_model=SignalIdsResponse,
    summary="Record alerts (deprecated, use POST /v1/crud/signals)",
)
def create_alerts(request: AlertsCreate, session: Session = Depends(get_session)):
    signals = [SignalCreate.model_validate(alert.model_dump()) for alert in request.alerts]
    return crud.create_signals(SignalsCreate(signals=signals), session)


@router.get(
    "/alerts",
    dependencies=[require(Permission.READ_SIGNALS)],
    response_model=list[AlertRead],
    summary="Browse alerts (deprecated, use GET /v1/crud/signals)",
)
def list_alerts(
    alert_policy_id: Annotated[int | None, Query(alias="alertPolicyId")] = None,
    prediction_id: Annotated[int | None, Query(alias="predictionId")] = None,
    org_unit: Annotated[str | None, Query(alias="orgUnit")] = None,
    approved: AlertApproval | None = None,
    session: Session = Depends(get_session),
):
    release_status = None if approved is None else _RELEASE_STATUS[approved]
    return [
        _alert(signal)
        for signal in crud.list_signals(alert_policy_id, prediction_id, org_unit, release_status, session)
    ]


@router.get(
    "/alerts/{alertId}",
    dependencies=[require(Permission.READ_SIGNALS)],
    response_model=AlertRead,
    summary="View one alert (deprecated, use GET /v1/crud/signals/{signalId})",
)
def get_alert(alert_id: Annotated[int, Path(alias="alertId")], session: Session = Depends(get_session)):
    return _alert(crud.get_signal(alert_id, session))


@router.delete(
    "/alerts/{alertId}",
    dependencies=[require(Permission.DELETE_SIGNALS)],
    summary="Remove an alert (deprecated, use DELETE /v1/crud/signals/{signalId})",
)
def delete_alert(alert_id: Annotated[int, Path(alias="alertId")], session: Session = Depends(get_session)):
    return crud.delete_signal(alert_id, session)


@router.post(
    "/alerts/$approve",
    dependencies=[require(Permission.APPROVE_SIGNALS)],
    response_model=list[AlertRead],
    summary="Approve or decline alerts (deprecated, use POST /v1/crud/signals/$set-release-status)",
)
def approve_alerts(request: AlertApprovalRequest, session: Session = Depends(get_session)):
    signals = crud.set_signal_release_status(
        SignalReleaseStatusRequest(
            signal_ids=request.alert_ids,
            release_status=_RELEASE_STATUS[request.approved],
            reviewed_by=request.approved_by,
        ),
        session,
    )
    return [_alert(signal) for signal in signals]


@router.post(
    "/alert-policies",
    dependencies=[require(Permission.WRITE_SIGNALS)],
    response_model=DataBaseResponse,
    summary="Create an alert policy (deprecated, use POST /v1/crud/threshold-policies)",
)
def create_alert_policy(request: AlertPolicyCreate, session: Session = Depends(get_session)):
    return crud.create_threshold_policy(ThresholdPolicyCreate.model_validate(request.model_dump()), session)


@router.get(
    "/alert-policies",
    dependencies=[require(Permission.READ_SIGNALS)],
    response_model=list[AlertPolicyRead],
    summary="Browse alert policies (deprecated, use GET /v1/crud/threshold-policies)",
)
def list_alert_policies(session: Session = Depends(get_session)):
    return [_alert_policy(policy) for policy in crud.list_threshold_policies(session)]


@router.get(
    "/alert-policies/{alertPolicyId}",
    dependencies=[require(Permission.READ_SIGNALS)],
    response_model=AlertPolicyRead,
    summary="View one alert policy (deprecated, use GET /v1/crud/threshold-policies/{thresholdPolicyId})",
)
def get_alert_policy(
    alert_policy_id: Annotated[int, Path(alias="alertPolicyId")], session: Session = Depends(get_session)
):
    return _alert_policy(crud.get_threshold_policy(alert_policy_id, session))


@router.patch(
    "/alert-policies/{alertPolicyId}",
    dependencies=[require(Permission.WRITE_SIGNALS)],
    response_model=AlertPolicyRead,
    summary="Update an alert policy's codes (deprecated, use PATCH /v1/crud/threshold-policies/{thresholdPolicyId})",
)
def update_alert_policy_codes(
    alert_policy_id: Annotated[int, Path(alias="alertPolicyId")],
    request: AlertPolicyCodesUpdate,
    session: Session = Depends(get_session),
):
    update = ThresholdPolicyCodesUpdate.model_validate(request.model_dump())
    return _alert_policy(crud.update_threshold_policy_codes(alert_policy_id, update, session))


@router.delete(
    "/alert-policies/{alertPolicyId}",
    dependencies=[require(Permission.DELETE_SIGNALS)],
    summary="Remove an alert policy (deprecated, use DELETE /v1/crud/threshold-policies/{thresholdPolicyId})",
)
def delete_alert_policy(
    alert_policy_id: Annotated[int, Path(alias="alertPolicyId")], session: Session = Depends(get_session)
):
    return crud.delete_threshold_policy(alert_policy_id, session)
