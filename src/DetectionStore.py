"""SQLite-backed detection results; embedding vectors remain in EmbeddingCache."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from src.ApplicationIdentity import STORAGE_DIRECTORY, LEGACY_STORAGE_DIRECTORY
import sqlite3

from src.Detection import Detection, DetectionRun, _identity, _score
from src.EmbeddingCorpus import EmbeddingRef, EmbeddingSpaceKey
from src.CanonicalChunkGrid import CanonicalChunkGrid
from src.DetectionTemporalGrouper import (
    ALGORITHM_VERSION, DetectionTemporalGrouper, TemporalGroup,
    TemporalGrouping, TemporalGroupingCancelled, TemporalGroupingConfig,
)


class _DetectionRunWriter:
    """Write one scan in a transaction; never expose incomplete results."""

    def __init__(self, connection, run):
        self._connection = connection
        self.run = run
        self.scored_count = 0
        self.detection_count = 0

    def add_batch(self, detections, *, scanned_count):
        """Persist one bounded scan batch, including its zero-detection cases.

        Each item must be a ``Detection`` for this run with a score at or
        above the run's threshold. Duplicates are rejected, not overwritten.
        """
        scanned_count = int(scanned_count)
        if scanned_count < 0:
            raise ValueError("scanned_count cannot be negative")

        def rows():
            for detection in detections:
                if not isinstance(detection, Detection):
                    raise TypeError("detections must contain Detection objects")
                if detection.run_id != self.run.run_id:
                    raise ValueError("detection belongs to a different run")
                if detection.score < self.run.threshold:
                    raise ValueError("detection score is below the scan threshold")
                yield (
                    detection.run_id,
                    detection.ref.recording_id,
                    detection.ref.chunk_index,
                    detection.score,
                )

        cursor = self._connection.executemany(
            "INSERT INTO detections (run_id, recording_id, chunk_index, score) "
            "VALUES (?, ?, ?, ?)",
            rows(),
        )
        added = cursor.rowcount
        if added > scanned_count:
            raise ValueError("batch contains more detections than scored embeddings")
        self.scored_count += scanned_count
        self.detection_count += added


class DetectionStore:
    """Query completed detection runs in bounded pages using indexed SQL.

    Each operation uses its own connection; one writer can scan on a background
    thread while readers browse previously completed runs in the GUI thread.
    SQLite WAL mode permits those operations to overlap.
    """

    def __init__(self, path=None):
        self.path = Path(path or self.default_path()).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS detection_runs (
                    run_id TEXT PRIMARY KEY,
                    classifier_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    pipeline_hash TEXT NOT NULL,
                    embedding_name TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    scored_count INTEGER NOT NULL,
                    detection_count INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_detection_runs_classifier
                    ON detection_runs(classifier_id, created_at DESC, run_id DESC);

                CREATE TABLE IF NOT EXISTS detections (
                    run_id TEXT NOT NULL REFERENCES detection_runs(run_id)
                        ON DELETE CASCADE,
                    recording_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    score REAL NOT NULL,
                    PRIMARY KEY (run_id, recording_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_detections_score
                    ON detections(run_id, score DESC, recording_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_detections_chunk
                    ON detections(run_id, chunk_index, recording_id);

                -- Derived, immutable grouping configurations. New gap or grid
                -- settings create another grouping; detections are unchanged.
                CREATE TABLE IF NOT EXISTS temporal_groupings (
                    grouping_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES detection_runs(run_id)
                        ON DELETE CASCADE,
                    algorithm_version INTEGER NOT NULL,
                    max_gap_s REAL NOT NULL,
                    chunk_duration_s REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    group_count INTEGER NOT NULL,
                    detection_count INTEGER NOT NULL,
                    UNIQUE(run_id, algorithm_version, max_gap_s, chunk_duration_s)
                );
                CREATE INDEX IF NOT EXISTS idx_temporal_groupings_run
                    ON temporal_groupings(run_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS temporal_groups (
                    grouping_id TEXT NOT NULL REFERENCES temporal_groupings(grouping_id)
                        ON DELETE CASCADE,
                    group_id INTEGER NOT NULL,
                    recording_id TEXT NOT NULL,
                    start_s REAL NOT NULL,
                    end_s REAL NOT NULL,
                    detection_count INTEGER NOT NULL,
                    PRIMARY KEY (grouping_id, group_id)
                );

                CREATE TABLE IF NOT EXISTS temporal_group_members (
                    grouping_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    recording_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    group_id INTEGER NOT NULL,
                    PRIMARY KEY (grouping_id, recording_id, chunk_index),
                    FOREIGN KEY (grouping_id, group_id)
                        REFERENCES temporal_groups(grouping_id, group_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (run_id, recording_id, chunk_index)
                        REFERENCES detections(run_id, recording_id, chunk_index)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_temporal_group_members_group
                    ON temporal_group_members(
                        grouping_id, group_id, recording_id, chunk_index
                    );
                """
            )

    @staticmethod
    def default_path():
        root = os.environ.get("XDG_DATA_HOME")
        home = Path(root).expanduser() if root else Path.home() / ".local" / "share"
        target = home / STORAGE_DIRECTORY / "detections.sqlite3"
        legacy = home / LEGACY_STORAGE_DIRECTORY / "detections.sqlite3"
        return legacy if not target.exists() and legacy.exists() else target

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection

    @contextmanager
    def write_run(self, run: DetectionRun):
        """Atomically create a completed scan, or roll everything back.

        Usage::

            with store.write_run(run) as writer:
                writer.add_batch(detections, scanned_count=len(batch))

        Results become visible only after a successful exit. Any exception
        (including a cancelled job) rolls back both run and detections.
        """
        if not isinstance(run, DetectionRun):
            raise TypeError("run must be a DetectionRun")
        if run.scored_count or run.detection_count:
            raise ValueError("new runs must have zero counts")
        with self._connect() as connection:
            with connection:
                connection.execute(
                    "INSERT INTO detection_runs VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)",
                    (
                        run.run_id,
                        run.classifier_id,
                        run.model_id,
                        run.embedding_space.pipeline_hash,
                        run.embedding_space.embedding_name,
                        run.threshold,
                        run.created_at,
                    ),
                )
                writer = _DetectionRunWriter(connection, run)
                yield writer
                connection.execute(
                    "UPDATE detection_runs SET scored_count = ?, "
                    "detection_count = ? WHERE run_id = ?",
                    (writer.scored_count, writer.detection_count, run.run_id),
                )

    def get_run(self, run_id):
        run_id = _identity(run_id, "run_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM detection_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row) if row is not None else None

    def list_runs(self, *, classifier_id=None, limit=100, offset=0):
        limit, offset = self._page(limit, offset)
        query = "SELECT * FROM detection_runs"
        parameters = []
        if classifier_id is not None:
            query += " WHERE classifier_id = ?"
            parameters.append(_identity(classifier_id, "classifier_id"))
        query += " ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?"
        parameters.extend((limit, offset))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._run_from_row(row) for row in rows)

    def count_detections(self, run_id, *, min_score=None, recording_id=None,
                         cancel_event=None):
        where, parameters = self._filters(run_id, min_score, recording_id)
        with self._connect() as connection:
            if cancel_event is not None:
                connection.set_progress_handler(
                    lambda: int(cancel_event.is_set()), 1000,
                )
            row = connection.execute(
                "SELECT COUNT(*) FROM detections" + where, parameters
            ).fetchone()
        return int(row[0])

    def _ordered_query(self, run_id, *, min_score, recording_id,
                       sort_by, descending, sort_value, grouping_id):
        """Build a validated full-run ordering, including SQL-backed group order."""
        if sort_by == "group":
            if grouping_id is None:
                raise ValueError("group sorting requires an active temporal grouping")
            grouping_id = _identity(grouping_id, "grouping_id")
            grouping = self.get_temporal_grouping(grouping_id)
            if grouping is None or grouping.run_id != run_id:
                raise ValueError("temporal grouping does not belong to this scan")
            # Qualify all fields: the membership table also has run/recording IDs.
            where, parameters = self._filters(run_id, min_score, recording_id)
            where = where.replace(" WHERE run_id =", " WHERE d.run_id =")
            where = where.replace(" AND score >=", " AND d.score >=")
            where = where.replace(" AND recording_id =", " AND d.recording_id =")
            direction = "DESC" if descending else "ASC"
            query = (
                "SELECT d.run_id, d.recording_id, d.chunk_index, d.score "
                "FROM detections AS d JOIN temporal_group_members AS gm "
                "ON gm.grouping_id = ? AND gm.run_id = d.run_id "
                "AND gm.recording_id = d.recording_id "
                "AND gm.chunk_index = d.chunk_index" + where
                + f" ORDER BY gm.group_id {direction}, "
                "d.recording_id ASC, d.chunk_index ASC"
            )
            return query, (grouping_id, *parameters), False

        where, parameters = self._filters(run_id, min_score, recording_id)
        order, needs_function = self._order_by(
            sort_by, descending, with_metadata=sort_value is not None,
        )
        query = (
            "SELECT run_id, recording_id, chunk_index, score FROM detections"
            + where + " ORDER BY " + order
        )
        return query, parameters, needs_function

    def list_detections(
        self, run_id, *, min_score=None, recording_id=None,
        sort_by="score", descending=True, limit=200, offset=0,
        sort_value=None, grouping_id=None, cancel_event=None,
    ):
        """Return a stable, globally sorted database page (never page-local sort)."""
        limit, offset = self._page(limit, offset)
        query, parameters, needs_function = self._ordered_query(
            run_id, min_score=min_score, recording_id=recording_id,
            sort_by=sort_by, descending=descending, sort_value=sort_value,
            grouping_id=grouping_id,
        )
        with self._connect() as connection:
            if cancel_event is not None:
                # Abort expensive sorts when a newer table request supersedes
                # this one; otherwise the one-worker queue can lag behind.
                connection.set_progress_handler(
                    lambda: int(cancel_event.is_set()), 1000,
                )
            if needs_function:
                connection.create_function("detection_sort_value", 2, sort_value)
            rows = connection.execute(
                query + " LIMIT ? OFFSET ?", (*parameters, limit, offset),
            ).fetchall()
        return tuple(self._detection_from_row(row) for row in rows)

    def iter_detections(
        self, run_id, *, min_score=None, recording_id=None,
        sort_by="score", descending=True, batch_size=1000,
        sort_value=None, grouping_id=None,
    ):
        """Stream the full ordered dataset with bounded fetches for export."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        query, parameters, needs_function = self._ordered_query(
            run_id, min_score=min_score, recording_id=recording_id,
            sort_by=sort_by, descending=descending, sort_value=sort_value,
            grouping_id=grouping_id,
        )
        with self._connect() as connection:
            if needs_function:
                connection.create_function("detection_sort_value", 2, sort_value)
            cursor = connection.execute(query, parameters)
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield self._detection_from_row(row)

    def iter_selection_refs(
        self, run_id, *, min_score=None, recording_id=None,
        grouping_id=None, group_id=None, batch_size=1000,
    ):
        """Stream identities for a filtered scan or one temporal group.

        Used by group/matching selection, not the table's current page. Group
        membership is always scoped to the original saved detection run.
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (group_id is None) != (grouping_id is None):
            raise ValueError("group_id and grouping_id must be provided together")
        run_id = _identity(run_id, "run_id")
        where, parameters = self._filters(run_id, min_score, recording_id)
        if grouping_id is not None:
            grouping_id = _identity(grouping_id, "grouping_id")
            group_id = self._positive_group_id(group_id)
            grouping = self.get_temporal_grouping(grouping_id)
            if grouping is None or grouping.run_id != run_id:
                raise ValueError("Temporal grouping does not belong to this scan")
            where = where.replace(" WHERE run_id =", " WHERE d.run_id =")
            where = where.replace(" AND score >=", " AND d.score >=")
            where = where.replace(" AND recording_id =", " AND d.recording_id =")
            query = (
                "SELECT d.recording_id, d.chunk_index FROM detections AS d "
                "JOIN temporal_group_members AS gm "
                "ON gm.run_id = d.run_id "
                "AND gm.recording_id = d.recording_id "
                "AND gm.chunk_index = d.chunk_index "
                + where + " AND gm.grouping_id = ? AND gm.group_id = ? "
                "ORDER BY d.recording_id, d.chunk_index"
            )
            parameters = (*parameters, grouping_id, group_id)
        else:
            query = (
                "SELECT recording_id, chunk_index FROM detections "
                + where + " ORDER BY recording_id, chunk_index"
            )
        with self._connect() as connection:
            cursor = connection.execute(query, parameters)
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                for row in batch:
                    yield EmbeddingRef(row["recording_id"], row["chunk_index"])

    def temporal_groups_for_page(self, grouping_id, refs):
        """Fetch memberships for one UI page in one query, avoiding N+1 reads."""
        refs = tuple(refs)
        if not refs:
            return {}
        if len(refs) > 400:
            raise ValueError("group lookups must be bounded to a table page")
        if any(not isinstance(ref, EmbeddingRef) for ref in refs):
            raise TypeError("refs must contain EmbeddingRef objects")
        placeholders = ", ".join("(?, ?)" for _ in refs)
        parameters = [_identity(grouping_id, "grouping_id")]
        for ref in refs:
            parameters.extend((ref.recording_id, ref.chunk_index))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT gm.recording_id, gm.chunk_index, g.* "
                "FROM temporal_group_members AS gm "
                "JOIN temporal_groups AS g ON g.grouping_id = gm.grouping_id "
                "AND g.group_id = gm.group_id "
                "WHERE gm.grouping_id = ? "
                f"AND (gm.recording_id, gm.chunk_index) IN ({placeholders})",
                parameters,
            ).fetchall()
        return {
            EmbeddingRef(row["recording_id"], row["chunk_index"]):
            self._group_from_row(row) for row in rows
        }

    def build_temporal_grouping(
        self, run_id, *, max_gap_s, grid: CanonicalChunkGrid,
        cancel_event=None, progress_callback=None,
    ) -> TemporalGrouping:
        """Atomically group ALL stored detections, regardless of view filters.

        ``grid`` must be the canonical grid of the run's embedding pipeline.
        The detection-run schema does not contain chunk duration, so callers
        must provide it explicitly instead of assuming five-second chunks.
        The group key includes the supplied duration; incorrect grid settings
        cannot be detected from an older run's stored metadata alone.

        Progress callbacks receive (assigned_count, total_detection_count).
        Cancellation raises TemporalGroupingCancelled and rolls back all writes.
        A successful grouping is cached for subsequent calls with the same
        run, gap and grid duration. Results are independent of SQLite pages,
        user filters, current labels, or the Explorer recording index.
        """
        run_id = _identity(run_id, "run_id")
        if not isinstance(grid, CanonicalChunkGrid):
            raise TypeError("grid must be a CanonicalChunkGrid")
        config = TemporalGroupingConfig(max_gap_s, grid.chunk_duration_s)
        grouping_id = config.grouping_id(run_id)

        def cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise TemporalGroupingCancelled("Temporal grouping cancelled")

        cancelled()
        with self._connect() as connection:
            # Serialize concurrent builders; never publish partial groups.
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM temporal_groupings WHERE grouping_id = ?",
                    (grouping_id,),
                ).fetchone()
                if existing is not None:
                    return self._grouping_from_row(existing)
                run = connection.execute(
                    "SELECT detection_count FROM detection_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if run is None:
                    raise ValueError(f"Unknown detection run: {run_id}")
                total = int(run["detection_count"])
                connection.execute(
                    "INSERT INTO temporal_groupings VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
                    (grouping_id, run_id, ALGORITHM_VERSION,
                     config.max_gap_s, config.chunk_duration_s,
                     datetime.now(timezone.utc).isoformat()),
                )
                grouper = DetectionTemporalGrouper(config)
                cursor = connection.execute(
                    "SELECT recording_id, chunk_index FROM detections "
                    "WHERE run_id = ? ORDER BY recording_id, chunk_index",
                    (run_id,),
                )
                pending = []
                assigned = 0
                active_id = None
                active_count = 0
                active_end = None
                groups = 0

                def flush_members():
                    if pending:
                        connection.executemany(
                            "INSERT INTO temporal_group_members "
                            "(grouping_id, run_id, recording_id, chunk_index, group_id) "
                            "VALUES (?, ?, ?, ?, ?)", pending,
                        )
                        pending.clear()

                def finish_group():
                    if active_id is not None:
                        connection.execute(
                            "UPDATE temporal_groups SET end_s = ?, "
                            "detection_count = ? "
                            "WHERE grouping_id = ? AND group_id = ?",
                            (active_end, active_count, grouping_id, active_id),
                        )

                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                    for row in rows:
                        cancelled()
                        assignment = grouper.assign(
                            EmbeddingRef(row["recording_id"], row["chunk_index"])
                        )
                        if assignment.new_group:
                            finish_group()
                            groups += 1
                            active_id = assignment.group_id
                            active_count = 0
                            connection.execute(
                                "INSERT INTO temporal_groups VALUES (?, ?, ?, ?, ?, 0)",
                                (grouping_id, active_id,
                                 assignment.ref.recording_id,
                                 assignment.start_s, assignment.end_s),
                            )
                        active_end = assignment.end_s
                        active_count += 1
                        pending.append((grouping_id, run_id,
                                        assignment.ref.recording_id,
                                        assignment.ref.chunk_index,
                                        assignment.group_id))
                        assigned += 1
                    flush_members()
                    if progress_callback is not None:
                        progress_callback(assigned, total)
                finish_group()
                if assigned != total:
                    raise RuntimeError("Detection run count changed during grouping")
                cancelled()
                connection.execute(
                    "UPDATE temporal_groupings SET group_count = ?, "
                    "detection_count = ? WHERE grouping_id = ?",
                    (groups, assigned, grouping_id),
                )
                if progress_callback is not None and assigned == 0:
                    progress_callback(0, total)
                cancelled()
            # The transaction has committed before we return a grouping.
            row = connection.execute(
                "SELECT * FROM temporal_groupings WHERE grouping_id = ?",
                (grouping_id,),
            ).fetchone()
            return self._grouping_from_row(row)

    def get_temporal_grouping(self, grouping_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM temporal_groupings WHERE grouping_id = ?",
                (_identity(grouping_id, "grouping_id"),),
            ).fetchone()
        return self._grouping_from_row(row) if row else None

    def list_temporal_groupings(self, run_id, *, limit=100, offset=0):
        limit, offset = self._page(limit, offset)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM temporal_groupings WHERE run_id = ? "
                "ORDER BY created_at DESC, grouping_id DESC LIMIT ? OFFSET ?",
                (_identity(run_id, "run_id"), limit, offset),
            ).fetchall()
        return tuple(self._grouping_from_row(row) for row in rows)

    def list_temporal_groups(self, grouping_id, *, limit=200, offset=0):
        limit, offset = self._page(limit, offset)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM temporal_groups WHERE grouping_id = ? "
                "ORDER BY group_id LIMIT ? OFFSET ?",
                (_identity(grouping_id, "grouping_id"), limit, offset),
            ).fetchall()
        return tuple(self._group_from_row(row) for row in rows)

    def get_temporal_group_for_detection(self, grouping_id, ref):
        if not isinstance(ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT g.* FROM temporal_group_members AS m "
                "JOIN temporal_groups AS g "
                "ON g.grouping_id = m.grouping_id AND g.group_id = m.group_id "
                "WHERE m.grouping_id = ? AND m.recording_id = ? "
                "AND m.chunk_index = ?",
                (_identity(grouping_id, "grouping_id"),
                 ref.recording_id, ref.chunk_index),
            ).fetchone()
        return self._group_from_row(row) if row else None

    def count_temporal_group_members(self, grouping_id, group_id):
        group_id = self._positive_group_id(group_id)
        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM temporal_group_members "
                "WHERE grouping_id = ? AND group_id = ?",
                (_identity(grouping_id, "grouping_id"), group_id),
            ).fetchone()[0])

    def list_temporal_group_members(
        self, grouping_id, group_id, *, limit=200, offset=0,
    ):
        """Page over full group membership, even if table filters hide rows."""
        group_id = self._positive_group_id(group_id)
        limit, offset = self._page(limit, offset)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT d.run_id, d.recording_id, d.chunk_index, d.score "
                "FROM temporal_group_members AS m "
                "JOIN detections AS d ON d.run_id = m.run_id "
                "AND d.recording_id = m.recording_id "
                "AND d.chunk_index = m.chunk_index "
                "WHERE m.grouping_id = ? AND m.group_id = ? "
                "ORDER BY d.recording_id, d.chunk_index LIMIT ? OFFSET ?",
                (_identity(grouping_id, "grouping_id"), group_id, limit, offset),
            ).fetchall()
        return tuple(self._detection_from_row(row) for row in rows)

    @staticmethod
    def _positive_group_id(group_id):
        group_id = int(group_id)
        if group_id <= 0:
            raise ValueError("group_id must be positive")
        return group_id

    @staticmethod
    def _grouping_from_row(row):
        if row["algorithm_version"] != ALGORITHM_VERSION:
            raise ValueError("Unsupported temporal grouping algorithm version")
        return TemporalGrouping(
            grouping_id=row["grouping_id"], run_id=row["run_id"],
            config=TemporalGroupingConfig(row["max_gap_s"], row["chunk_duration_s"]),
            created_at=row["created_at"],
            group_count=row["group_count"], detection_count=row["detection_count"],
        )

    @staticmethod
    def _group_from_row(row):
        return TemporalGroup(
            grouping_id=row["grouping_id"], group_id=row["group_id"],
            recording_id=row["recording_id"], start_s=row["start_s"],
            end_s=row["end_s"], detection_count=row["detection_count"],
        )

    @staticmethod
    def _detection_from_row(row):
        return Detection(
            run_id=row["run_id"],
            ref=EmbeddingRef(row["recording_id"], row["chunk_index"]),
            score=row["score"],
        )

    # Only these identifiers can appear in ORDER BY. Never interpolate user
    # input directly into SQL, including a displayed column heading.
    _NATIVE_SORTS = {
        "score": ("score", False),
        "recording_id": ("recording_id", True),
        "chunk": ("chunk_index", False),
    }
    _METADATA_SORTS = {
        "recording": True,
        "human_label": True,
        "start_s": False,
        "end_s": False,
        "duration_s": False,
        "device_id": True,
        "capture_start": True,
    }

    @classmethod
    def _order_by(cls, sort_by, descending, *, with_metadata=False):
        """Return validated ORDER BY SQL and whether it needs a SQLite UDF.

        Missing metadata always sorts last in both directions. Recording ID
        and chunk break every tie, keeping paging and keyboard navigation
        deterministic even when many rows have the same score/metadata.
        """
        if sort_by in cls._NATIVE_SORTS:
            expression, is_text = cls._NATIVE_SORTS[sort_by]
            needs_function = False
        elif sort_by == "recording" and not with_metadata:
            # Legacy callers without Explorer metadata sort by recording ID.
            expression, is_text, needs_function = "recording_id", True, False
        elif sort_by in cls._METADATA_SORTS:
            if not with_metadata:
                raise ValueError(f"sorting by {sort_by!r} requires metadata")
            expression = "detection_sort_value(recording_id, chunk_index)"
            is_text = cls._METADATA_SORTS[sort_by]
            needs_function = True
        else:
            raise ValueError(f"unsupported detection sort column: {sort_by!r}")
        direction = "DESC" if descending else "ASC"
        collation = " COLLATE NOCASE" if is_text else ""
        return (
            f"{expression}{collation} {direction} NULLS LAST, "
            "recording_id ASC, chunk_index ASC",
            needs_function,
        )

    @staticmethod
    def _page(limit, offset):
        limit, offset = int(limit), int(offset)
        if limit <= 0 or offset < 0:
            raise ValueError("limit must be positive and offset nonnegative")
        return limit, offset

    @staticmethod
    def _filters(run_id, min_score, recording_id):
        where = " WHERE run_id = ?"
        parameters = [_identity(run_id, "run_id")]
        if min_score is not None:
            where += " AND score >= ?"
            parameters.append(_score(min_score, "min_score"))
        if recording_id is not None:
            where += " AND recording_id = ?"
            parameters.append(str(recording_id))
        return where, tuple(parameters)

    @staticmethod
    def _run_from_row(row):
        return DetectionRun(
            run_id=row["run_id"],
            classifier_id=row["classifier_id"],
            model_id=row["model_id"],
            embedding_space=EmbeddingSpaceKey(
                pipeline_hash=row["pipeline_hash"],
                embedding_name=row["embedding_name"],
            ),
            threshold=row["threshold"],
            created_at=row["created_at"],
            scored_count=row["scored_count"],
            detection_count=row["detection_count"],
        )
