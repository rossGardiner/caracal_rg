"""Controller for saved detection scans, paged browsing, and Explorer links."""

from functools import partial
from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from src.BinaryEmbeddingClassifier import BinaryEmbeddingClassifier
from src.DetectionMetadata import DetectionMetadata
from src.DetectionExporter import (
    DetectionExporter, DetectionExportProgress,
)
from src.DetectionProvenance import describe_run_model
from src.DetectionScanner import (
    DetectionScanProgress, DetectionScanner, fingerprint_classifier_state,
)
from src.DetectionStore import DetectionStore
from src.DetectionsModel import DetectionsModel, PAGE_SIZE
from src.DetectionsView import DetectionsView
from src.LatestJobRunner import LatestJobRunner


class DetectionsController(QObject):
    """Own browsing and background scan orchestration, not audio playback."""

    # The composition root validates the embedding space before navigating.
    open_detection_requested = Signal(object, object)
    label_detection_requested = Signal(str, object, object, object)

    def __init__(
        self, *, model: DetectionsModel, view: DetectionsView,
        store: DetectionStore, active_learning_model, classifier_store,
        parent=None,
    ):
        super().__init__(parent)
        self.model = model
        self.view = view
        self.store = store
        self.active_learning_model = active_learning_model
        self.classifier_store = classifier_store
        self.metadata = DetectionMetadata()
        self._available_classifiers = ()
        self._labels_version = None
        self._scan_request_id = 0
        self._scan_cancel_event = None
        self._export_request_id = 0
        self._export_cancel_event = None
        self._exporting = False
        self._provenance_request_id = 0
        self._provenance_key = None
        self._model_state = "unknown"
        self._current_model_id = None
        self._runs = ()
        self.provenance_jobs = LatestJobRunner(name="detection-provenance", parent=self)
        self.provenance_jobs.ready.connect(self._provenance_ready)
        self.provenance_jobs.failed.connect(self._provenance_failed)
        self.scan_jobs = LatestJobRunner(name="detection-scans", parent=self)
        self.scan_jobs.ready.connect(self._scan_ready)
        self.scan_jobs.failed.connect(self._scan_failed)
        self.scan_jobs.progress.connect(self._scan_progress)

        view.classifier_selected.connect(self.select_classifier)
        view.run_selected.connect(self.select_run)
        view.filters_requested.connect(self.apply_filters)
        view.sort_requested.connect(self.set_sort)
        view.page_requested.connect(self.change_page)
        view.detection_row_selected.connect(self.select_row)
        view.previous_requested.connect(self.previous)
        view.next_requested.connect(self.next)
        view.open_requested.connect(self.open_selected)
        view.refresh_requested.connect(self.refresh_runs)
        view.generate_requested.connect(self.generate)
        view.cancel_requested.connect(self.cancel_scan)
        self.export_jobs = LatestJobRunner(name="detection-csv-export", parent=self)
        self.export_jobs.ready.connect(self._export_ready)
        self.export_jobs.failed.connect(self._export_failed)
        self.export_jobs.progress.connect(self._export_progress)
        view.export_requested.connect(self.export_csv)
        view.cancel_export_requested.connect(self.cancel_export)
        view.context_menu_requested.connect(self.show_label_context_menu)
        view.human_label_requested.connect(self.request_human_label)
        self.refresh_from_active_learning()

    @Slot()
    def refresh_from_active_learning(self):
        """Update classifier names and scan readiness without resetting pages."""
        definitions = tuple(self.active_learning_model.classifier_definitions)
        signature = tuple((item.classifier_id, item.name) for item in definitions)
        old_signature = tuple((item.classifier_id, item.name) for item in self._available_classifiers)
        self._available_classifiers = definitions
        if signature != old_signature:
            ids = {item.classifier_id for item in definitions}
            if self.model.classifier_id not in ids:
                self.model.classifier_id = (
                    self.active_learning_model.selected_classifier_id
                    if self.active_learning_model.selected_classifier_id in ids
                    else definitions[0].classifier_id if definitions else None
                )
                self.view.set_classifiers(definitions, self.model.classifier_id)
                self.refresh_runs()
            else:
                self.view.set_classifiers(definitions, self.model.classifier_id)
        self._refresh_labels()
        self._refresh_model_provenance()
        self._update_scan_enabled()
        self._update_export_enabled()

    def _refresh_labels(self):
        definition = next(
            (item for item in self._available_classifiers
             if item.classifier_id == self.model.classifier_id), None
        )
        version = (self.model.classifier_id, id(definition))
        if version == self._labels_version:
            return
        self._labels_version = version
        self.metadata.set_labels(
            {item.ref: item.label.name.title() for item in definition.labels}
            if definition is not None else {}
        )
        if self.model.run is not None and self.model.rows:
            selected = self.model.selected_index
            self._load_page(
                self.model.page,
                select_local=(selected % PAGE_SIZE if selected is not None else 0),
            )

    def _refresh_model_provenance(self):
        """Rehash saved weights off the GUI thread only when their file changes."""
        definition = next(
            (item for item in self._available_classifiers
             if item.classifier_id == self.model.classifier_id), None
        )
        classifier_id = self.model.classifier_id
        try:
            if classifier_id is None:
                signature = None
            elif hasattr(self.classifier_store, "model_state_signature"):
                signature = self.classifier_store.model_state_signature(classifier_id)
            else:
                # Allow simple classifier-store stubs in GUI tests. In the
                # application the real store always supplies a file token.
                signature = (0, 0) if self.classifier_store.has_model_state(classifier_id) else None
        except OSError:
            signature = None
        # Include the parameters because they contribute to the model hash.
        parameters = getattr(definition, "classifier_parameters", None)
        key = (classifier_id, signature, repr(parameters))
        if key == self._provenance_key:
            return
        self._provenance_key = key
        self._provenance_request_id += 1
        request_id = self._provenance_request_id
        self._current_model_id = None
        if classifier_id is None or signature is None:
            self._model_state = "missing"
        elif parameters is None:
            self._model_state = "error"
        else:
            self._model_state = "checking"

            def job():
                saved = self.classifier_store.load_model_state(classifier_id)
                if saved is None:
                    raise RuntimeError("Saved classifier weights disappeared")
                return fingerprint_classifier_state(parameters, saved)

            self.provenance_jobs.submit(request_id, job)
        self._show_provenance()

    def _show_provenance(self):
        if self._runs:
            selected_id = self.model.run.run_id if self.model.run else None
            self.view.set_runs(
                self._runs, selected_id, current_model_id=self._current_model_id,
                model_state=self._model_state,
            )
        if self.model.run is not None and self.model.rows and not self.model.scanning:
            self.view.show_status(self._compatibility_status())

    @Slot(int, object)
    def _provenance_ready(self, request_id, model_id):
        if request_id != self._provenance_request_id:
            return
        # The saved file can be replaced while a background hash runs.
        classifier_id, signature, _ = self._provenance_key
        current_signature = (
            self.classifier_store.model_state_signature(classifier_id)
            if hasattr(self.classifier_store, "model_state_signature") else
            (0, 0) if self.classifier_store.has_model_state(classifier_id) else None
        )
        if current_signature != signature:
            self._refresh_model_provenance()
            return
        self._current_model_id = model_id
        self._model_state = "ready"
        self._show_provenance()

    @Slot(int, str)
    def _provenance_failed(self, request_id, _message):
        if request_id != self._provenance_request_id:
            return
        self._current_model_id = None
        self._model_state = "error"
        self._show_provenance()

    def _update_scan_enabled(self):
        active = self.active_learning_model
        can_scan = (
            not self._exporting
            and not self.model.scanning
            and self.model.classifier_id is not None
            and self.model.classifier_id == active.selected_classifier_id
            and active.is_ready
            and not active.round_running
            and active.learner.classifier.is_fitted
            # Labelling invalidates the persisted weights, even while the
            # prepared learner still has the old fitted head in memory.
            and self.classifier_store.has_model_state(self.model.classifier_id)
        )
        self.view.generate_button.setEnabled(bool(can_scan))
        self.view.scan_threshold.setEnabled(not self.model.scanning)
        self.view.cancel_button.setEnabled(
            self.model.scanning
            and self._scan_cancel_event is not None
            and not self._scan_cancel_event.is_set()
        )

    def _update_export_enabled(self):
        self.view.set_export_state(
            available=self.model.run is not None and not self.model.scanning,
            running=self._exporting,
            cancelling=(
                self._export_cancel_event is not None
                and self._export_cancel_event.is_set()
            ),
        )

    @Slot(str)
    def select_classifier(self, classifier_id):
        if classifier_id == self.model.classifier_id:
            return
        self.model.classifier_id = classifier_id or None
        self._refresh_labels()
        self._refresh_model_provenance()
        self.refresh_runs()
        self._update_scan_enabled()
        self._update_export_enabled()

    @Slot()
    def refresh_runs(self, *, preferred_id=None):
        """Load compact run headers; detection rows remain paged on disk."""
        if self.model.classifier_id is None:
            self.model.clear_results()
            self._runs = ()
            self.view.set_runs((), None)
            self._show_empty("Select a classifier to browse saved detections.")
            return
        self._refresh_model_provenance()
        try:
            runs = self.store.list_runs(classifier_id=self.model.classifier_id, limit=100)
        except Exception as exc:
            self._runs = ()
            self._show_empty(f"Could not load detection scans: {exc}")
            return
        previous = preferred_id or (self.model.run.run_id if self.model.run else None)
        selected = next((run for run in runs if run.run_id == previous), None)
        if selected is None and runs:
            selected = runs[0]
        self._runs = runs
        self.view.set_runs(
            runs, selected.run_id if selected else None,
            current_model_id=self._current_model_id, model_state=self._model_state,
        )
        self._set_run(selected)

    @Slot(str)
    def select_run(self, run_id):
        if self.model.run is not None and self.model.run.run_id == run_id:
            return
        try:
            run = self.store.get_run(run_id) if run_id else None
            if run is not None and run.classifier_id != self.model.classifier_id:
                raise ValueError("Detection scan belongs to a different classifier")
        except Exception as exc:
            self._show_empty(f"Cannot load detection scan: {exc}")
            return
        self._set_run(run)

    def _set_run(self, run):
        self.model.run = run
        self._update_export_enabled()
        self.model.min_score = run.threshold if run else None
        self.model.recording_id = None
        self.view.recording_filter.clear()
        self.view.set_filters_enabled(run is not None, run.threshold if run else 0.0)
        if run is not None:
            self.view.min_score.setValue(run.threshold)
            self._load_page(0)
        else:
            self._show_empty("No saved scans. Prepare and train a classifier in Active Learning, then generate detections.")

    @Slot(float, str)
    def apply_filters(self, min_score, recording_id):
        if self.model.run is None:
            return
        self.model.min_score = max(self.model.run.threshold, float(min_score))
        self.model.recording_id = recording_id.strip() or None
        self._load_page(0)

    @Slot(str, bool)
    def set_sort(self, sort_by, descending):
        if sort_by not in self.view.table_model.FIELDS:
            return
        self.model.sort_by, self.model.descending = sort_by, bool(descending)
        self.view.set_sort_indicator(sort_by, descending)
        self._load_page(0)

    @Slot(int)
    def change_page(self, change):
        if self.model.run is None:
            return
        page = self.model.page + int(change)
        if 0 <= page < self._page_count():
            self._load_page(page)

    def _page_count(self):
        return (self.model.total + PAGE_SIZE - 1) // PAGE_SIZE

    def _load_page(self, page, *, select_local=0, open_in_explorer=False):
        run = self.model.run
        if run is None:
            return
        try:
            total = self.store.count_detections(
                run.run_id, min_score=self.model.min_score,
                recording_id=self.model.recording_id,
            )
            page = min(int(page), max(0, (total - 1) // PAGE_SIZE))
            rows = self.store.list_detections(
                run.run_id, min_score=self.model.min_score,
                recording_id=self.model.recording_id,
                sort_by=self.model.sort_by, descending=self.model.descending,
                limit=PAGE_SIZE, offset=page * PAGE_SIZE,
                sort_value=self._sort_value(run),
            )
        except Exception as exc:
            self._show_empty(f"Could not load detections: {exc}")
            return
        self.model.set_page(page, rows, total)
        metadata = tuple(self.metadata.resolve(run, detection) for detection in rows)
        self.view.show_page(rows, metadata, selected_row=select_local if rows else None)
        if rows:
            # Refreshing a page, changing filters or reloading metadata must
            # not trigger playback. Only explicit navigation should do that.
            self._select_row(select_local, open_in_explorer=open_in_explorer)
        if not rows:
            self.view.show_details(None, None, None)
            self.view.set_navigation(0, total, page, self._page_count())
            self.view.show_status("No detections match these filters." if total == 0 else "No results in this page.")
        else:
            self.view.show_status(self._compatibility_status())

    def _sort_value(self, run, *, metadata=None):
        """Live values for columns whose sort keys are not stored in SQLite."""
        sort_by = self.model.sort_by
        if sort_by in self.store._NATIVE_SORTS:
            return None
        source = self.metadata if metadata is None else metadata
        return lambda recording_id, chunk_index: source.sort_value(
            run, sort_by, recording_id, chunk_index,
        )

    def _show_empty(self, message):
        self.model.set_page(0, (), 0)
        self._update_export_enabled()
        self.view.show_page((), ())
        self.view.show_details(None, None, None)
        self.view.set_navigation(0, 0, 0, 0)
        self.view.show_status(message)

    @Slot(int)
    def select_row(self, local_row):
        self._select_row(local_row, open_in_explorer=True)

    def _select_row(self, local_row, *, open_in_explorer):
        detection = self.model.select_local_row(local_row)
        self.view.set_navigation(
            self.model.position, self.model.total, self.model.page, self._page_count()
        )
        if detection is None:
            self.view.show_details(None, None, None)
            return
        metadata = self.metadata.resolve(self.model.run, detection)
        self.view.show_details(self.model.run, detection, metadata)
        source_status = metadata.get("source_status")
        status = self._compatibility_status()
        if (self.metadata.space == self.model.run.embedding_space
                and source_status != "Chunk reference resolves in Explorer"):
            status += " Selected source: " + str(source_status) + "."
        self.view.show_status(status)
        if open_in_explorer:
            # The bridge in explorer.py enforces compatibility a second time.
            self.open_selected()

    @Slot(int, object)
    def show_label_context_menu(self, local_row, global_position):
        run = self.model.run
        if run is None or not 0 <= local_row < len(self.model.rows):
            return
        detection = self.model.rows[local_row]
        # Right-click changes the inspector selection but never starts playback.
        self._select_row(local_row, open_in_explorer=False)
        compatible = tuple(
            (definition.classifier_id, definition.name,
             definition.label_for(detection.ref))
            for definition in self.active_learning_model.classifier_definitions
            if getattr(definition, "embedding_space", None) == run.embedding_space
        )
        # Put the scan's classifier first, followed by other compatible models.
        compatible = tuple(sorted(
            compatible, key=lambda item: (item[0] != run.classifier_id, item[1].lower())
        ))
        self.view.show_human_label_menu(global_position, detection, run, compatible)

    @Slot(object, object, str, object)
    def request_human_label(self, ref, embedding_space, classifier_id, label):
        # A menu may outlive the active page/run if a modal event loop runs.
        # The AL controller enforces compatibility and the current round state.
        if not any(
            definition.classifier_id == classifier_id
            and getattr(definition, "embedding_space", None) == embedding_space
            for definition in self.active_learning_model.classifier_definitions
        ):
            self.view.show_status("Classifier is missing or uses a different embedding space.")
            return
        self.label_detection_requested.emit(classifier_id, ref, embedding_space, label)

    @Slot()
    def previous(self):
        self._move(-1)

    @Slot()
    def next(self):
        self._move(1)

    def _move(self, amount):
        if self.model.total == 0:
            return
        current = self.model.selected_index
        target = (0 if current is None else current + amount)
        if not 0 <= target < self.model.total:
            return
        page, local = divmod(target, PAGE_SIZE)
        if page != self.model.page:
            self._load_page(page, select_local=local, open_in_explorer=True)
        else:
            self.view.table.selectRow(local)
            # A programmatic row selection may not emit currentRowChanged if
            # Qt already considers the requested row current.
            if self.model.selected_index != target:
                self._select_row(local, open_in_explorer=True)

    @Slot()
    def open_selected(self):
        run, detection = self.model.run, self.model.selected
        if run is None or detection is None:
            return
        if self.metadata.space != run.embedding_space:
            self.view.show_status(self._compatibility_status())
            return
        self.open_detection_requested.emit(run.embedding_space, detection.ref)

    def _compatibility_status(self):
        run = self.model.run
        if run is None:
            return "No detection run selected."
        provenance = " " + describe_run_model(
            run, current_model_id=self._current_model_id,
            model_state=self._model_state,
        )
        if self.metadata.space != run.embedding_space:
            return (
                "Saved results can be browsed. To inspect audio and full metadata, "
                "activate the matching pipeline in Explorer "
                f"({run.embedding_space.embedding_name}, "
                f"{run.embedding_space.pipeline_hash[:12]}…)." + provenance
            )
        return (
            f"Ready: {self.model.total:,} matching detections. "
            "Click a row to open it in Explorer." + provenance
        )

    def set_explorer_context(self, space, packets, grid):
        """Called after Explorer activates a pipeline, not on each chunk change."""
        self.metadata.set_context(space, packets, grid)
        if self.model.run is None:
            return
        selected = self.model.selected_index
        if selected is None:
            self._load_page(self.model.page)
        else:
            page, local = divmod(selected, PAGE_SIZE)
            self._load_page(page, select_local=local)

    @Slot(float)
    def generate(self, threshold):
        """Scan saved trained weights only; never score a mutable training head."""
        active = self.active_learning_model
        if self.model.scanning or self._exporting:
            return
        if (
            not active.is_ready or active.round_running or
            self.model.classifier_id != active.selected_classifier_id
        ):
            self.view.show_status("Select and prepare this classifier in Active Learning first.")
            return
        classifier_id = self.model.classifier_id
        if not self.classifier_store.has_model_state(classifier_id):
            self.view.show_status(
                "No saved fitted weights. Run an Active Learning round after "
                "your latest label changes before generating detections."
            )
            self._update_scan_enabled()
            return
        parameters = active.selected_classifier.classifier_parameters
        corpus = active.corpus
        store = self.store
        classifier_store = self.classifier_store
        if corpus.space != active.embedding_space:
            self.view.show_status("Prepared corpus does not match the classifier embedding space.")
            return

        self._scan_request_id += 1
        request_id = self._scan_request_id
        cancel_event = Event()
        self._scan_cancel_event = cancel_event
        self.model.scanning = True
        self._update_scan_enabled()
        self._update_export_enabled()
        self.view.show_status("Scanning embeddings… existing completed scans remain available.")

        def job():
            saved_state = classifier_store.load_model_state(classifier_id)
            if saved_state is None:
                raise RuntimeError("No saved fitted model. Run an Active Learning round after your latest label changes.")
            classifier = BinaryEmbeddingClassifier(parameters)
            classifier.restore_state(saved_state)
            scanner = DetectionScanner(
                corpus=corpus, classifier=classifier,
                store=store, classifier_id=classifier_id,
            )
            return scanner.scan(
                threshold=threshold,
                progress_callback=partial(self.scan_jobs.report_progress, request_id),
                cancel_event=cancel_event,
            )

        self.scan_jobs.submit(request_id, job)

    @Slot()
    def cancel_scan(self):
        """Cooperatively stop the running scan without discarding older runs."""
        if self.model.scanning and self._scan_cancel_event is not None:
            self._scan_cancel_event.set()
            self.view.show_status(
                "Cancelling scan… waiting for the current embedding batch to finish."
            )
            self._update_scan_enabled()

    @Slot(int, object)
    def _scan_progress(self, request_id, progress):
        if request_id != self._scan_request_id or not isinstance(progress, DetectionScanProgress):
            return
        if self._scan_cancel_event is not None and self._scan_cancel_event.is_set():
            return
        self.view.show_status(
            f"Scanning… {progress.scored_count:,} embeddings scored, "
            f"{progress.detection_count:,} detections found."
        )

    @Slot(int, object)
    def _scan_ready(self, request_id, run):
        if request_id != self._scan_request_id:
            return
        self.model.scanning = False
        self._scan_cancel_event = None
        self._update_scan_enabled()
        self._update_export_enabled()
        if run.classifier_id == self.model.classifier_id:
            self.refresh_runs(preferred_id=run.run_id)
            self.view.show_status(self._compatibility_status())
        else:
            self.view.show_status(
                "Scan completed for another classifier; select it to view its saved results."
            )

    @Slot(int, str)
    def _scan_failed(self, request_id, message):
        if request_id != self._scan_request_id:
            return
        self.model.scanning = False
        cancelled = message.startswith("DetectionScanCancelled:")
        self._scan_cancel_event = None
        self._update_scan_enabled()
        self._update_export_enabled()
        self.view.show_status(
            "Scan cancelled. Previously completed detection runs were preserved."
            if cancelled else f"Could not generate detections: {message}"
        )

    @Slot()
    def export_csv(self):
        """Export from SQLite (not from the visible 200-row table page)."""
        if self.model.run is None or self._exporting or self.model.scanning:
            return
        run = self.model.run
        chosen = self.view.choose_csv_export(run, self.model.total)
        if chosen is None:
            return
        destination, filtered = chosen
        # Snapshot active context and labels before entering the worker thread;
        # switching classifier/pipeline while exporting must not affect CSV rows.
        metadata = DetectionMetadata()
        metadata.set_context(
            self.metadata.space, tuple(self.metadata._packets.values()), self.metadata.grid
        )
        metadata.set_labels(dict(self.metadata._labels))
        filters = dict(
            min_score=self.model.min_score if filtered else None,
            recording_id=self.model.recording_id if filtered else None,
            sort_by=self.model.sort_by,
            descending=self.model.descending,
            sort_value=self._sort_value(run, metadata=metadata),
        )
        self._export_request_id += 1
        request_id = self._export_request_id
        cancel_event = Event()
        self._export_cancel_event = cancel_event
        self._exporting = True
        self._update_scan_enabled()
        self._update_export_enabled()
        self.view.show_export_status("Exporting detections…")

        def job():
            return DetectionExporter(self.store).export_csv(
                run, destination, metadata=metadata,
                cancel_event=cancel_event,
                progress_callback=partial(self.export_jobs.report_progress, request_id),
                **filters,
            )

        self.export_jobs.submit(request_id, job)

    @Slot()
    def cancel_export(self):
        if self._exporting and self._export_cancel_event is not None:
            self._export_cancel_event.set()
            self._update_export_enabled()
            self.view.show_export_status("Cancelling CSV export…")

    @Slot(int, object)
    def _export_progress(self, request_id, progress):
        if (request_id != self._export_request_id
                or not isinstance(progress, DetectionExportProgress)
                or self._export_cancel_event is None
                or self._export_cancel_event.is_set()):
            return
        self.view.show_export_status(
            f"Exporting CSV: {progress.written:,} / {progress.total:,} detections"
        )

    @Slot(int, object)
    def _export_ready(self, request_id, result):
        if request_id != self._export_request_id:
            return
        self._exporting = False
        self._export_cancel_event = None
        self._update_scan_enabled()
        self._update_export_enabled()
        self.view.show_export_status(
            f"Exported {result.written:,} detections to {result.path}"
        )

    @Slot(int, str)
    def _export_failed(self, request_id, message):
        if request_id != self._export_request_id:
            return
        cancelled = message.startswith("DetectionExportCancelled:")
        self._exporting = False
        self._export_cancel_event = None
        self._update_scan_enabled()
        self._update_export_enabled()
        self.view.show_export_status(
            "CSV export cancelled; existing destination was preserved."
            if cancelled else f"CSV export failed: {message}"
        )

    def shutdown(self):
        if self._scan_cancel_event is not None:
            self._scan_cancel_event.set()
        if self._export_cancel_event is not None:
            self._export_cancel_event.set()
        self.export_jobs.shutdown()
        self.scan_jobs.shutdown()
        self.provenance_jobs.shutdown()
