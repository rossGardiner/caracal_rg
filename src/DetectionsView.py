"""Qt view for paged detection browsing; no scanning or persistence logic."""

from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSignalBlocker, Qt, Signal
from src.ActiveLearner import BinaryLabel

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class DetectionTableModel(QAbstractTableModel):
    """A fixed-size page of detections, not one widget per corpus chunk."""

    HEADERS = (
        "Score", "Recording", "Recording ID", "Chunk", "Human label",
        "Start (s)", "End (s)", "Duration (s)", "Device", "Captured (UTC)",
    )
    FIELDS = (
        "score", "recording", "recording_id", "chunk", "human_label",
        "start_s", "end_s", "duration_s", "device_id", "capture_start",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = ()
        self.metadata = ()

    def set_page(self, rows, metadata):
        self.beginResetModel()
        self.rows = tuple(rows)
        self.metadata = tuple(metadata)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return section + 1

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (
            Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.TextAlignmentRole,
        ):
            return None
        field = self.FIELDS[index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return (
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                if field in ("score", "chunk", "start_s", "end_s", "duration_s")
                else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
        value = self.metadata[index.row()].get(field)
        if value is None:
            return "—"
        if field == "score":
            return f"{value:.3f}"
        if field in ("start_s", "end_s", "duration_s"):
            return f"{value:.2f}"
        return str(value)


class DetectionTableView(QTableView):
    """Allow keyboard browsing to continue across database page boundaries."""

    previous_at_boundary = Signal()
    next_at_boundary = Signal()
    open_at_selection = Signal()

    def keyPressEvent(self, event):
        if event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return super().keyPressEvent(event)

        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.open_at_selection.emit()
            event.accept()
            return

        index = self.currentIndex()
        count = self.model().rowCount() if self.model() is not None else 0
        if index.isValid() and count:
            if event.key() == Qt.Key.Key_Down and index.row() == count - 1:
                self.next_at_boundary.emit()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Up and index.row() == 0:
                self.previous_at_boundary.emit()
                event.accept()
                return
        super().keyPressEvent(event)


class DetectionsView(QWidget):
    """Displays detection snapshots and emits user intent to its controller."""

    classifier_selected = Signal(str)
    run_selected = Signal(str)
    filters_requested = Signal(float, str)
    sort_requested = Signal(str, bool)
    page_requested = Signal(int)
    detection_row_selected = Signal(int)
    previous_requested = Signal()
    next_requested = Signal()
    open_requested = Signal()
    refresh_requested = Signal()
    generate_requested = Signal(float)
    cancel_requested = Signal()
    export_requested = Signal()
    cancel_export_requested = Signal()
    context_menu_requested = Signal(int, object)
    human_label_requested = Signal(object, object, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("detections_workspace")
        self.setMinimumWidth(0)
        self._sort_by = "score"
        self._descending = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        setup = QGridLayout()
        setup.addWidget(QLabel("Classifier"), 0, 0)
        self.classifier_combo = QComboBox()
        self.classifier_combo.setObjectName("detections_classifier")
        setup.addWidget(self.classifier_combo, 0, 1)
        setup.addWidget(QLabel("Saved scan"), 0, 2)
        self.run_combo = QComboBox()
        self.run_combo.setObjectName("detections_run")
        setup.addWidget(self.run_combo, 0, 3)

        setup.addWidget(QLabel("Scan ≥"), 1, 0)
        self.scan_threshold = QDoubleSpinBox()
        self.scan_threshold.setRange(0.0, 1.0)
        self.scan_threshold.setDecimals(2)
        self.scan_threshold.setSingleStep(0.05)
        self.scan_threshold.setValue(0.5)
        self.scan_threshold.setToolTip("Minimum model score to store in a new scan")
        setup.addWidget(self.scan_threshold, 1, 1)
        self.generate_button = QPushButton("Generate detections")
        setup.addWidget(self.generate_button, 1, 2)
        self.cancel_button = QPushButton("Cancel scan")
        self.cancel_button.setEnabled(False)
        setup.addWidget(self.cancel_button, 1, 3)
        self.refresh_button = QPushButton("Refresh saved runs")
        setup.addWidget(self.refresh_button, 1, 4)

        setup.addWidget(QLabel("Show score ≥"), 2, 0)
        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setDecimals(3)
        self.min_score.setSingleStep(0.01)
        setup.addWidget(self.min_score, 2, 1)
        setup.addWidget(QLabel("Recording ID (exact)"), 2, 2)
        self.recording_filter = QLineEdit()
        self.recording_filter.setPlaceholderText("Leave blank for all recordings")
        setup.addWidget(self.recording_filter, 2, 3)
        self.apply_button = QPushButton("Apply filters")
        setup.addWidget(self.apply_button, 2, 4)
        setup.setColumnStretch(1, 1)
        setup.setColumnStretch(3, 2)
        layout.addLayout(setup)

        self.status_label = QLabel("Select a classifier to browse saved detections.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.count_label = QLabel("0 detections")
        layout.addWidget(self.count_label)

        export_controls = QHBoxLayout()
        self.export_button = QPushButton("Export CSV…")
        self.export_button.setObjectName("detections_export_csv")
        self.export_button.setToolTip("Export all or filtered results from the selected saved scan")
        self.export_button.setEnabled(False)
        self.cancel_export_button = QPushButton("Cancel export")
        self.cancel_export_button.setEnabled(False)
        self.export_status_label = QLabel("")
        self.export_status_label.setWordWrap(True)
        export_controls.addWidget(self.export_button)
        export_controls.addWidget(self.cancel_export_button)
        export_controls.addWidget(self.export_status_label, 1)
        layout.addLayout(export_controls)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = DetectionTableView()
        self.table.setObjectName("detections_table")
        self.table_model = DetectionTableModel(self.table)
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu_at)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().setSortIndicator(0, Qt.SortOrder.DescendingOrder)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 180)
        self.table.setColumnWidth(2, 140)
        self.table.setColumnWidth(3, 65)
        self.table.setColumnWidth(4, 100)
        self.table.setColumnWidth(5, 90)
        self.table.setColumnWidth(6, 90)
        self.table.setColumnWidth(7, 90)
        self.table.setColumnWidth(8, 130)
        self.table.setToolTip(
            "Click a row to inspect it; use Up/Down to browse across pages, "
            "Enter to open in Explorer, or click any column heading to sort "
            "the entire filtered detection list (click again to reverse). "
            "Right-click a detection to set a model-specific human label."
        )
        split.addWidget(self.table)

        inspector = QGroupBox("Selected detection")
        inspector_layout = QVBoxLayout(inspector)
        self.details = QPlainTextEdit()
        self.details.setObjectName("detection_details")
        self.details.setReadOnly(True)
        self.details.setPlainText("Select a result to see its metadata.")
        inspector_layout.addWidget(self.details)
        self.open_button = QPushButton("Open in Explorer")
        inspector_layout.addWidget(self.open_button)
        split.addWidget(inspector)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        footer = QHBoxLayout()
        self.prev_page_button = QPushButton("◀ Page")
        self.next_page_button = QPushButton("Page ▶")
        self.prev_button = QPushButton("◀ Previous detection")
        self.next_button = QPushButton("Next detection ▶")
        self.position_label = QLabel("0 / 0")
        footer.addWidget(self.prev_page_button)
        footer.addWidget(self.prev_button)
        footer.addWidget(self.position_label, 1, Qt.AlignmentFlag.AlignCenter)
        footer.addWidget(self.next_button)
        footer.addWidget(self.next_page_button)
        layout.addLayout(footer)

        self.classifier_combo.currentIndexChanged.connect(
            lambda _: self.classifier_selected.emit(self.classifier_combo.currentData() or "")
        )
        self.run_combo.currentIndexChanged.connect(
            lambda _: self.run_selected.emit(self.run_combo.currentData() or "")
        )
        self.generate_button.clicked.connect(
            lambda: self.generate_requested.emit(self.scan_threshold.value())
        )
        self.refresh_button.clicked.connect(self.refresh_requested)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.cancel_export_button.clicked.connect(self.cancel_export_requested)
        self.apply_button.clicked.connect(self._apply_filters)
        self.recording_filter.returnPressed.connect(self._apply_filters)
        self.table.selectionModel().currentRowChanged.connect(
            lambda current, _: self.detection_row_selected.emit(current.row())
        )
        self.table.doubleClicked.connect(lambda _: self.open_requested.emit())
        self.table.open_at_selection.connect(self.open_requested)
        self.table.previous_at_boundary.connect(self.previous_requested)
        self.table.next_at_boundary.connect(self.next_requested)
        self.table.horizontalHeader().sectionClicked.connect(self._sort_column)
        self.prev_button.clicked.connect(self.previous_requested)
        self.next_button.clicked.connect(self.next_requested)
        self.open_button.clicked.connect(self.open_requested)
        self.prev_page_button.clicked.connect(lambda: self.page_requested.emit(-1))
        self.next_page_button.clicked.connect(lambda: self.page_requested.emit(1))
        self.set_navigation(0, 0, 0, 0)
        self.generate_button.setEnabled(False)
        self.set_filters_enabled(False)

    def _context_menu_at(self, position):
        """Ask the controller which classifiers are compatible with this row."""
        index = self.table.indexAt(position)
        if not index.isValid():
            return
        # Right-clicking must not open audio in Explorer.
        with QSignalBlocker(self.table.selectionModel()):
            self.table.selectRow(index.row())
        self.context_menu_requested.emit(
            index.row(), self.table.viewport().mapToGlobal(position)
        )

    def show_human_label_menu(self, global_position, detection, run, targets):
        """The scan's classifier gets one-click actions; others use submenus."""
        menu = QMenu(self.table)

        def add_actions(parent, classifier_id, existing_label):
            for text, label in (
                ("Positive", BinaryLabel.POSITIVE),
                ("Negative", BinaryLabel.NEGATIVE),
                ("Remove human label", None),
            ):
                action = parent.addAction(text)
                action.setCheckable(label is not None)
                action.setChecked(label is not None and existing_label is label)
                if label is None:
                    action.setEnabled(existing_label is not None)
                action.triggered.connect(
                    lambda _checked=False, cid=classifier_id, choice=label:
                        self.human_label_requested.emit(
                            detection.ref, run.embedding_space, cid, choice
                        )
                )

        default = next(
            (item for item in targets if item[0] == run.classifier_id), None
        )
        if default is not None:
            menu.addSection(f"Human label for {default[1]}")
            add_actions(menu, default[0], default[2])
        others = [item for item in targets if item[0] != run.classifier_id]
        if others:
            parent = menu.addMenu("Label for another classifier…") if default else menu
            for classifier_id, name, existing_label in others:
                add_actions(parent.addMenu(name), classifier_id, existing_label)
        if not targets:
            empty = menu.addAction("No compatible classifiers")
            empty.setEnabled(False)
        menu.exec(global_position)

    def _apply_filters(self):
        self.filters_requested.emit(self.min_score.value(), self.recording_filter.text().strip())

    def _sort_column(self, column):
        # The table only contains a database page. Sorting MUST be delegated
        # to the controller/store, never QTableView's page-local sort().
        if not 0 <= column < len(self.table_model.FIELDS):
            return
        new_sort = self.table_model.FIELDS[column]
        descending = (not self._descending) if new_sort == self._sort_by else (new_sort == "score")
        self.set_sort_indicator(new_sort, descending)
        self.sort_requested.emit(new_sort, descending)

    def set_sort_indicator(self, sort_by, descending):
        self._sort_by, self._descending = sort_by, bool(descending)
        try:
            column = self.table_model.FIELDS.index(sort_by)
        except ValueError:
            return
        self.table.horizontalHeader().setSortIndicator(
            column, Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder
        )

    def set_classifiers(self, definitions, selected_id):
        self.classifier_combo.blockSignals(True)
        self.classifier_combo.clear()
        self.classifier_combo.addItem("Select classifier", "")
        for definition in definitions:
            self.classifier_combo.addItem(definition.name, definition.classifier_id)
        index = self.classifier_combo.findData(selected_id)
        self.classifier_combo.setCurrentIndex(max(0, index))
        self.classifier_combo.blockSignals(False)

    def set_runs(self, runs, selected_id, *, current_model_id=None, model_state="unknown"):
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in runs:
            if model_state == "ready":
                provenance = "Current model" if run.model_id == current_model_id else "Historical model"
            elif model_state == "missing":
                provenance = "No fitted model saved"
            elif model_state == "checking":
                provenance = "Checking model…"
            elif model_state == "error":
                provenance = "Model status unavailable"
            else:
                provenance = "Model not checked"
            name = (f"{run.created_at[:19]} · {run.detection_count:,} detections "
                    f"· ≥{run.threshold:.2f} · {provenance}")
            self.run_combo.addItem(name, run.run_id)
        index = self.run_combo.findData(selected_id)
        self.run_combo.setCurrentIndex(max(0, index)) if runs else None
        self.run_combo.blockSignals(False)

    def set_filters_enabled(self, enabled, minimum=0.0):
        self.min_score.setEnabled(enabled)
        self.recording_filter.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
        self.min_score.setMinimum(minimum)
        if self.min_score.value() < minimum:
            self.min_score.setValue(minimum)

    def show_page(self, rows, metadata, *, selected_row=None):
        # Model resets and programmatic selection must not look like a user's
        # row click. The controller explicitly updates the inspector after
        # populating the page and only requests audio on user navigation.
        with QSignalBlocker(self.table.selectionModel()):
            self.table_model.set_page(rows, metadata)
            if selected_row is not None and 0 <= selected_row < len(rows):
                self.table.selectRow(selected_row)
                self.table.setCurrentIndex(self.table_model.index(selected_row, 0))
                self.table.scrollTo(self.table_model.index(selected_row, 0))

    def set_navigation(self, position, total, page, page_count):
        self.position_label.setText(f"{position:,} / {total:,}   ·   Page {page + 1} / {max(1, page_count)}")
        self.count_label.setText(f"{total:,} matching detections")
        self.prev_button.setEnabled(position > 1)
        self.next_button.setEnabled(0 < position < total)
        self.prev_page_button.setEnabled(page > 0)
        self.next_page_button.setEnabled(page + 1 < page_count)
        self.open_button.setEnabled(position > 0)

    def show_details(self, run, detection, metadata):
        if run is None or detection is None or metadata is None:
            self.details.setPlainText("Select a result to see its metadata.")
            return
        lines = [
            f"Recording: {metadata['recording']}",
            f"Recording ID: {detection.ref.recording_id}",
            f"Chunk: {detection.ref.chunk_index}",
            f"Score: {detection.score:.4f}",
            f"Human label: {metadata['human_label']}",
        ]
        for label, field in (
            ("Start (s)", "start_s"), ("End (s)", "end_s"),
            ("Duration (s)", "duration_s"), ("Device ID", "device_id"),
            ("Card ID", "card_id"), ("Latitude", "latitude"),
            ("Longitude", "longitude"), ("Capture time (UTC)", "capture_start"),
        ):
            value = metadata.get(field)
            lines.append(f"{label}: {value if value is not None else '—'}")
        lines.extend((
            "", "Prediction provenance:",
            f"Classifier: {run.classifier_id}",
            f"Model version: {run.model_id[:16]}…",
            f"Pipeline hash: {run.embedding_space.pipeline_hash}",
            f"Embedding: {run.embedding_space.embedding_name}",
            f"Scanned: {run.scored_count:,} embeddings",
            f"Scan cutoff: {run.threshold:.2f}",
            f"Scan date: {run.created_at}",
            f"Source: {metadata.get('source_status', 'Unknown')}",
        ))
        self.details.setPlainText("\n".join(lines))

    def show_status(self, message):
        self.status_label.setText(message)

    def choose_csv_export(self, run, matching_count):
        """Ask for export scope and destination on the Qt main thread."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Export detections to CSV")
        contents = QVBoxLayout(dialog)
        contents.addWidget(QLabel(
            f"Saved scan: {run.created_at[:19]}\n"
            f"{run.detection_count:,} stored detections. "
            "Only scores retained by the scan can be exported."
        ))
        all_radio = QRadioButton("All detections in this saved scan")
        filtered_radio = QRadioButton(
            f"Currently applied filters ({matching_count:,} matching, all pages)"
        )
        all_radio.setChecked(True)
        contents.addWidget(all_radio)
        contents.addWidget(filtered_radio)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        contents.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        suggested = f"detections-{run.run_id[:8]}.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save detections as CSV", suggested, "CSV files (*.csv)"
        )
        if not filename:
            return None
        destination = Path(filename)
        if not destination.suffix:
            destination = destination.with_suffix(".csv")
        return destination, filtered_radio.isChecked()

    def set_export_state(self, *, available, running, cancelling=False):
        self.export_button.setEnabled(bool(available and not running))
        self.cancel_export_button.setEnabled(bool(running and not cancelling))

    def show_export_status(self, message):
        self.export_status_label.setText(message)
