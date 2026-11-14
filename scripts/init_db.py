"""Create/upgrade the SQLite schema, register the installed dataset and import the reference bundle."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main() -> None:
    from app.db.session import migrate, session_scope
    from app.services import dataset_service, reference_service

    migrate()
    with session_scope() as db:
        snap = dataset_service.sync_from_disk(db)
    print("database ready;", "dataset registered" if snap else "dataset not installed yet")
    n = reference_service.import_if_needed()
    print(f"reference experiments imported: {n}")


if __name__ == "__main__":
    main()
