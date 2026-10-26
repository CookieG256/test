from datetime import datetime

from config import ACCESS_COOLDOWN_SECONDS, UNKNOWN_COOLDOWN_SECONDS
from database import EventStore


class AccessController:
    def __init__(self, event_store: EventStore) -> None:
        self.event_store = event_store
        self._last_unknown_at: datetime | None = None

    def register_known(
        self, name: str, role: str, timestamp: datetime
    ) -> str | None:
        person_state = self.event_store.get_person_state(name)
        if person_state:
            last_event = datetime.fromisoformat(person_state["last_event_at"])
            if (timestamp - last_event).total_seconds() < ACCESS_COOLDOWN_SECONDS:
                return None
            current_status = person_state["status"]
        else:
            current_status = "Fuera"

        action = "SALIDA" if current_status == "Dentro" else "ENTRADA"
        self.event_store.record_known_event(name, role, action, timestamp)
        return action

    def allow_unknown_attempt(self, timestamp: datetime) -> bool:
        if self._last_unknown_at is not None:
            elapsed = (timestamp - self._last_unknown_at).total_seconds()
            if elapsed < UNKNOWN_COOLDOWN_SECONDS:
                return False
        self._last_unknown_at = timestamp
        return True

    def register_unknown(self, timestamp: datetime, image_path) -> None:
        self.event_store.record_unknown_event(timestamp, image_path)