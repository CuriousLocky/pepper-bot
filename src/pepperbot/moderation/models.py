import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field, StrictBool


class Classification(BaseModel):
    suspected: StrictBool
    reason: str = Field(min_length=1)


class ReviewDecision(BaseModel):
    is_spam: StrictBool
    reason: str = Field(min_length=1)


Decision = TypeVar("Decision", bound=BaseModel)


def parse_decision(text: str, model: Type[Decision]) -> Decision:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return model.model_validate(json.loads(text))


CaseOutcome = Literal["banned", "cleared", "cancelled"]
CaseStatus = Literal["screening", "challenging", "pending", "final_pending", "ban_pending", "announcing", "banned", "cleared", "cancelled"]


class SpamCase(BaseModel):
    id: str
    chat_id: int
    user_id: int
    user: Dict[str, Any]
    joined_at: datetime
    join_info: Dict[str, Any] = Field(default_factory=dict)
    status: CaseStatus = "screening"
    thread_id: Optional[str] = None
    profile: Dict[str, Any] = Field(default_factory=dict)
    classification_reason: str = ""
    challenged_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    warned_at: Optional[datetime] = None
    final_deadline: Optional[datetime] = None
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    decision: Optional[ReviewDecision] = None
    outcome: Optional[CaseOutcome] = None
    outcome_reason: str = ""
    last_error: str = ""

    @property
    def active(self) -> bool:
        return self.status in {"screening", "challenging", "pending", "final_pending", "ban_pending", "announcing"}


class SpamCaseStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.cases: Dict[str, SpamCase] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.cases = {key: SpamCase.model_validate(value) for key, value in data["cases"].items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps({
            "version": 1,
            "cases": {key: case.model_dump(mode="json") for key, case in self.cases.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def active_for(self, chat_id: int, user_id: int) -> Optional[SpamCase]:
        return next((case for case in self.cases.values()
                     if case.chat_id == chat_id and case.user_id == user_id and case.active), None)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
