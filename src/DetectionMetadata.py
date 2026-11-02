"""Resolve optional Explorer recording metadata without audio I/O or Qt."""

from datetime import datetime, timezone

from src.CanonicalChunkGrid import CanonicalChunkGrid
from src.EmbeddingCorpus import EmbeddingSpaceKey


class DetectionMetadata:
    """Only use packets belonging to the detection run's embedding space."""

    def __init__(self):
        self.space = None
        self.grid = None
        self._packets = {}
        self._labels = {}

    def set_context(self, space, packets=(), grid=None):
        self.space = space
        self.grid = grid
        self._packets = {
            str(packet.recording_id): packet
            for packet in packets
            if packet.recording_id is not None
        }

    def set_labels(self, labels):
        """Human labels for the currently viewed classifier (not predictions)."""
        self._labels = dict(labels)

    def resolve(self, run, detection):
        """Return stable, display-friendly metadata (missing values are None)."""
        result = {
            "recording": detection.ref.recording_id,
            "recording_id": detection.ref.recording_id,
            "chunk": detection.ref.chunk_index,
            "score": detection.score,
            "human_label": self._labels.get(detection.ref, "Unlabelled"),
            "start_s": None,
            "end_s": None,
            "duration_s": None,
            "device_id": None,
            "card_id": None,
            "latitude": None,
            "longitude": None,
            "capture_start": None,
            "source_status": "Matching pipeline not active in Explorer",
        }
        if self.space != run.embedding_space:
            return result
        if self.grid is None:
            result["source_status"] = "Explorer has no active chunk grid"
            return result
        packet = self._packets.get(detection.ref.recording_id)
        if packet is None:
            result["source_status"] = "Recording not present in Explorer's active dataset"
            return result
        result["source_status"] = "Chunk unavailable in the current recording"

        result["recording"] = packet.display_name or detection.ref.recording_id
        if isinstance(self.grid, CanonicalChunkGrid):
            try:
                start, end = self.grid.chunk_bounds(
                    detection.ref.chunk_index, packet.duration
                )
                result.update(start_s=start, end_s=end, duration_s=end - start)
                result["source_status"] = "Chunk reference resolves in Explorer"
            except (ValueError, IndexError, TypeError):
                pass

        misc = packet.misc_metadata if isinstance(packet.misc_metadata, dict) else {}
        # CARACAL metadata keys are optional. Never stringify arbitrary parsed
        # headers or large nested values into a Qt table cell.
        for target, keys in (
            ("device_id", ("device_id", "device", "Device ID")),
            ("card_id", ("card_id", "card", "Card ID")),
        ):
            for key in keys:
                value = misc.get(key)
                if isinstance(value, (str, int, float, bool)):
                    result[target] = str(value)
                    break
        result["latitude"] = packet.lat
        result["longitude"] = packet.lon
        if packet.capture_start_time is not None:
            try:
                result["capture_start"] = datetime.fromtimestamp(
                    float(packet.capture_start_time), tz=timezone.utc
                ).isoformat()
            except (OverflowError, ValueError, TypeError, OSError):
                pass
        return result
