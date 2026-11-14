"""SQLAlchemy ORM models.

Small, queryable records live here; large numeric arrays (predictions, errors,
thresholds) and model checkpoints are files under ``artifacts/`` whose paths are
stored in these tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    source_url: Mapped[str] = mapped_column(String(512))
    retrieved_at: Mapped[str] = mapped_column(String(64))
    dataset_checksum: Mapped[str] = mapped_column(String(64))
    labels_sha256: Mapped[str] = mapped_column(String(64))
    n_files: Mapped[int] = mapped_column(Integer)
    total_bytes: Mapped[int] = mapped_column(Integer)
    validation_ok: Mapped[bool] = mapped_column(Boolean)
    report: Mapped[dict[str, Any]] = mapped_column(JSON)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Channel(Base):
    __tablename__ = "channels"

    chan_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    spacecraft: Mapped[str] = mapped_column(String(8), index=True)
    train_len: Mapped[int] = mapped_column(Integer)
    test_len: Mapped[int] = mapped_column(Integer)
    n_features: Mapped[int] = mapped_column(Integer)
    n_intervals: Mapped[int] = mapped_column(Integer)
    labelled_points: Mapped[int] = mapped_column(Integer)
    label_fraction: Mapped[float] = mapped_column(Float)
    value_min: Mapped[float] = mapped_column(Float)
    value_max: Mapped[float] = mapped_column(Float)
    train_std: Mapped[float] = mapped_column(Float)
    commands_active_train: Mapped[int] = mapped_column(Integer)
    commands_active_test: Mapped[int] = mapped_column(Integer)
    label_rows: Mapped[int] = mapped_column(Integer, default=1)
    issues: Mapped[list[Any]] = mapped_column(JSON, default=list)

    intervals: Mapped[list["LabelInterval"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan", order_by="LabelInterval.start"
    )


class LabelInterval(Base):
    __tablename__ = "label_intervals"

    id: Mapped[int] = mapped_column(primary_key=True)
    chan_id: Mapped[str] = mapped_column(ForeignKey("channels.chan_id", ondelete="CASCADE"), index=True)
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    cls: Mapped[str] = mapped_column(String(16))
    merged_from: Mapped[list[Any]] = mapped_column(JSON, default=list)

    channel: Mapped[Channel] = relationship(back_populates="intervals")


class Experiment(Base):
    """One run of one method over one or more channels (also acts as the job record)."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    method: Mapped[str] = mapped_column(String(32), index=True)
    profile: Mapped[str | None] = mapped_column(String(32), nullable=True)
    spacecraft: Mapped[str] = mapped_column(String(16))
    channels: Mapped[list[Any]] = mapped_column(JSON)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    threshold_config: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_channel: Mapped[str | None] = mapped_column(String(16), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    device: Mapped[str | None] = mapped_column(String(32), nullable=True)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    train_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    infer_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    origin: Mapped[str] = mapped_column(String(16), default="local")
    environment: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    runs: Mapped[list["ChannelRun"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", order_by="ChannelRun.chan_id"
    )


class ChannelRun(Base):
    __tablename__ = "channel_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    chan_id: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    epochs_run: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    infer_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    arrays_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    eval_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    n_detections: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    epochs: Mapped[list["EpochLog"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="EpochLog.epoch"
    )
    detections: Mapped[list["Detection"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Detection.start"
    )


class EpochLog(Base):
    __tablename__ = "epoch_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_run_id: Mapped[int] = mapped_column(ForeignKey("channel_runs.id", ondelete="CASCADE"), index=True)
    epoch: Mapped[int] = mapped_column(Integer)
    train_loss: Mapped[float] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    seconds: Mapped[float] = mapped_column(Float)

    run: Mapped[ChannelRun] = relationship(back_populates="epochs")


class Detection(Base):
    """A model-detected anomaly interval (test-stream indices, inclusive)."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_run_id: Mapped[int] = mapped_column(ForeignKey("channel_runs.id", ondelete="CASCADE"), index=True)
    experiment_id: Mapped[int] = mapped_column(Integer, index=True)
    chan_id: Mapped[str] = mapped_column(String(16), index=True)
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    peak_error: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(8))
    outcome: Mapped[str] = mapped_column(String(16))  # matches_label | no_label_overlap
    matched_labels: Mapped[list[Any]] = mapped_column(JSON, default=list)
    window: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    run: Mapped[ChannelRun] = relationship(back_populates="detections")


Index("ix_detections_exp_chan", Detection.experiment_id, Detection.chan_id)
