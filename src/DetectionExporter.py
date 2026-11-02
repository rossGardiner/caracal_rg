"""Qt-free, bounded-memory CSV export of persisted detection scan results."""

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from src.Detection import DetectionRun
from src.DetectionMetadata import DetectionMetadata
from src.DetectionStore import DetectionStore


COLUMNS = (
    "run_id", "classifier_id", "model_id", "scan_created_at_utc", "scan_threshold",
    "pipeline_hash", "embedding_name", "recording_id", "recording_name",
    "chunk_index", "score", "start_s", "end_s", "duration_s",
    "device_id", "card_id", "recording_capture_time_utc", "latitude",
    "longitude", "human_label", "source_status",
)


@dataclass(frozen=True)
class DetectionExportProgress:
    written: int
    total: int


@dataclass(frozen=True)
class DetectionExportResult:
    path: Path
    written: int


class DetectionExportCancelled(RuntimeError):
    """Cancellation removes the temporary file without touching the destination."""


def _csv_text(value):
    """Prevent recording metadata from being interpreted as a spreadsheet formula."""
    if value is None:
        return ""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


class DetectionExporter:
    def __init__(self, store: DetectionStore):
        self.store = store

    def export_csv(
        self, run: DetectionRun, destination, *, min_score=None,
        recording_id=None, sort_by="score", descending=True,
        metadata: DetectionMetadata | None = None,
        batch_size=1000, cancel_event=None, progress_callback=None,
    ) -> DetectionExportResult:
        """Export all matching rows, not just the visible Qt table page.

        A temporary file in the output folder is atomically moved into place
        only on success. The metadata resolver is an optional snapshot of the
        current Explorer context; unavailable recording data stays blank.
        """
        if not isinstance(run, DetectionRun):
            raise TypeError("run must be a DetectionRun")
        saved = self.store.get_run(run.run_id)
        if saved is None or saved != run:
            raise ValueError("Detection run no longer matches the saved results")
        if metadata is None:
            metadata = DetectionMetadata()
        if not isinstance(metadata, DetectionMetadata):
            raise TypeError("metadata must be a DetectionMetadata instance")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")

        destination = Path(destination).expanduser()
        if not destination.name or not destination.parent.is_dir():
            raise ValueError("Export destination must have an existing parent directory")

        def check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise DetectionExportCancelled("CSV export cancelled")

        check_cancel()
        total = self.store.count_detections(
            run.run_id, min_score=min_score, recording_id=recording_id,
        )
        if progress_callback is not None:
            progress_callback(DetectionExportProgress(0, total))
        fd, temporary = tempfile.mkstemp(
            dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".part",
        )
        written = 0
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as output:
                writer = csv.DictWriter(output, fieldnames=COLUMNS)
                writer.writeheader()
                for detection in self.store.iter_detections(
                    run.run_id, min_score=min_score, recording_id=recording_id,
                    sort_by=sort_by, descending=descending, batch_size=batch_size,
                ):
                    # Cancellation is checked per row (even with large batches).
                    check_cancel()
                    fields = metadata.resolve(run, detection)
                    row = {
                        "run_id": run.run_id,
                        "classifier_id": run.classifier_id,
                        "model_id": run.model_id,
                        "scan_created_at_utc": run.created_at,
                        "scan_threshold": run.threshold,
                        "pipeline_hash": run.embedding_space.pipeline_hash,
                        "embedding_name": run.embedding_space.embedding_name,
                        "recording_id": detection.ref.recording_id,
                        "recording_name": fields["recording"],
                        "chunk_index": detection.ref.chunk_index,
                        "score": detection.score,
                        "start_s": fields["start_s"],
                        "end_s": fields["end_s"],
                        "duration_s": fields["duration_s"],
                        "device_id": fields["device_id"],
                        "card_id": fields["card_id"],
                        "recording_capture_time_utc": fields["capture_start"],
                        "latitude": fields["latitude"],
                        "longitude": fields["longitude"],
                        "human_label": fields["human_label"],
                        "source_status": fields["source_status"],
                    }
                    writer.writerow({key: _csv_text(value) for key, value in row.items()})
                    written += 1
                    if progress_callback is not None and written % batch_size == 0:
                        progress_callback(DetectionExportProgress(written, total))
                check_cancel()
            check_cancel()
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        if progress_callback is not None:
            progress_callback(DetectionExportProgress(written, total))
        return DetectionExportResult(destination, written)
