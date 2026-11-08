"""Qt view for paged detection browsing; no scanning or persistence logic."""

from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, QItemSelection, QItemSelectionModel,
    QSignalBlocker, Qt, Signal,
)
from PySide6.QtGui import QBrush, QColor
from src.ActiveLearner import BinaryLabel

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class DetectionTableModel(QAbstractTableModel):
    """A fixed-size page of detections, not one widget per corpus chunk."""

    HEADERS = (
        "Score", "Group", "Recording", "Recording ID", "Chunk", "Label",
        "Start (s)", "End (s)", "Duration (s)", "Device", "Captured (UTC)",
    )
    FIELDS = (
        "score", "group", "recording", "recording_id", "chunk", "human_label",
        "start_s", "end_s", "duration_s", "device_id", "capture_start",
    )
    _GROUP_COLOURS = (
        QColor("#3b82f6"), QColor("#22a878"), QColor("#e3a12f"),
        QColor("#a78bfa"), QColor("#ed7795"), QColor("#0baabd"),
        QColor("#b0a34b"), QColor("#df8b55"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = ()
        self.metadata = ()
        self.groups = ()

    def set_page(self, rows, metadata, groups=None):
        self.beginResetModel()
        self.rows = tuple(rows)
        self.metadata = tuple(metadata)
        self.groups = tuple(groups) if groups is not None else (None,) * len(self.rows)
        if len(self.rows) != len(self.metadata) or len(self.rows) != len(self.groups):
            raise ValueError("Detection table rows, metadata and groups must align")
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
        if not index.isValid():
            return None
        field = self.FIELDS[index.column()]
        group = self.groups[index.row()]
        if role == Qt.ItemDataRole.BackgroundRole and group is not None:
            # Blend rather than replace the palette, remaining legible in
            # dark/light themes. Group numbers provide non-colour identity.
            parent = self.parent()
            base = parent.palette().base().color() if parent is not None else QColor("#ffffff")
            accent = self._GROUP_COLOURS[(group.group_id - 1) % len(self._GROUP_COLOURS)]
            weight = 0.16
            return QBrush(QColor(
                round(base.red() * (1-weight) + accent.red() * weight),
                round(base.green() * (1-weight) + accent.green() * weight),
                round(base.blue() * (1-weight) + accent.blue() * weight),
            ))
        if role == Qt.ItemDataRole.ToolTipRole and group is not None:
            return (f"Temporal group {group.group_id} · "
                    f"{group.detection_count:,} detections (entire scan) · "
                    f"{group.start_s:.2f}–{group.end_s:.2f} s")
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return (
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                if field in ("score", "group", "chunk", "start_s", "end_s", "duration_s")
                else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if field == "group":
            return str(group.group_id) if group is not None else "—"
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bulk_selection_in_progress = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            # Qt's default right-click handling can collapse an existing
            # ExtendedSelection before the context menu action is chosen.
            # Leave highlighting untouched: contextMenuEvent still fires.
            event.accept()
            return
        self.bulk_selection_in_progress = bool(event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        ))
        try:
            super().mousePressEvent(event)
        finally:
            self.bulk_selection_in_progress = False

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
    human_labels_requested = Signal(object, str, object, str, object)
    grouping_requested = Signal(bool, float)
    selected_page_refs_changed = Signal(object)
    clear_selection_requested = Signal()
    select_all_matching_requested = Signal()
    select_group_requested = Signal(int, bool)  # group ID, respect filters
    label_selected_requested = Signal()


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("detections_workspace")
        self.setMinimumWidth(0)
        self._sort_by = "score"
        self._descending = True
        self._context_group = None
        self._context_label_refs = frozenset()
        self._context_outside_selection = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Keep the everyday workflow compact: select scan, filter, inspect rows.
        header = QGridLayout()
        header.addWidget(QLabel("Classifier"), 0, 0)
        self.classifier_combo = QComboBox()
        self.classifier_combo.setObjectName("detections_classifier")
        header.addWidget(self.classifier_combo, 0, 1)
        header.addWidget(QLabel("Scan"), 1, 0)
        self.run_combo = QComboBox()
        self.run_combo.setObjectName("detections_run")
        header.addWidget(self.run_combo, 1, 1)
        self.generate_button = QPushButton("Scan…")
        self.generate_button.setObjectName("detections_scan")
        header.addWidget(self.generate_button, 1, 2)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.hide()
        header.addWidget(self.cancel_button, 1, 3)
        header.setColumnStretch(1, 1)
        layout.addLayout(header)

        # Keep the scan cutoff in its own dialogue; the filter does not rescan.
        self.scan_dialog = QDialog(self)
        self.scan_dialog.setWindowTitle("New scan")
        scan_layout = QVBoxLayout(self.scan_dialog)
        scan_cutoff = QHBoxLayout()
        scan_cutoff.addWidget(QLabel("Minimum score"))
        self.scan_threshold = QDoubleSpinBox()
        self.scan_threshold.setRange(0.0, 1.0)
        self.scan_threshold.setDecimals(2)
        self.scan_threshold.setSingleStep(0.05)
        self.scan_threshold.setValue(0.5)
        self.scan_threshold.setToolTip("Only detections at or above this score are stored")
        scan_cutoff.addWidget(self.scan_threshold)
        scan_layout.addLayout(scan_cutoff)
        scan_actions = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        scan_actions.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        scan_actions.accepted.connect(self.scan_dialog.accept)
        scan_actions.rejected.connect(self.scan_dialog.reject)
        scan_layout.addWidget(scan_actions)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Score ≥"))
        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setDecimals(3)
        self.min_score.setSingleStep(0.01)
        filters.addWidget(self.min_score)
        filters.addWidget(QLabel("Recording"))
        self.recording_filter = QLineEdit()
        self.recording_filter.setPlaceholderText("Recording ID (exact)")
        filters.addWidget(self.recording_filter, 1)
        self.apply_button = QPushButton("Apply")
        filters.addWidget(self.apply_button)
        self.grouping_checkbox = QCheckBox("Group detections")
        self.grouping_checkbox.setObjectName("detections_grouping_enabled")
        self.grouping_checkbox.setChecked(True)
        self.grouping_checkbox.setToolTip(
            "Colour nearby detections in the same recording. "
            "Every detection remains a separate row."
        )
        filters.addWidget(self.grouping_checkbox)
        layout.addLayout(filters)

        # Details… occupies its own row (not the filter toolbar).
        details_row = QHBoxLayout()
        self.details_toggle = QToolButton()
        self.details_toggle.setObjectName("detections_details_toggle")
        self.details_toggle.setText("Details…")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        details_row.addWidget(self.details_toggle)
        self.group_status = QLabel("")
        self.group_status.setWordWrap(True)
        details_row.addWidget(self.group_status, 1)
        layout.addLayout(details_row)

        self.details_panel = QWidget()
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(0, 0, 0, 0)
        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Max group gap"))
        self.group_gap = QDoubleSpinBox()
        self.group_gap.setObjectName("detections_group_gap")
        self.group_gap.setRange(0.0, 86400.0)
        self.group_gap.setDecimals(3)
        self.group_gap.setSingleStep(1.0)
        self.group_gap.setValue(10.0)
        self.group_gap.setSuffix(" s")
        self.group_gap.setToolTip("Maximum gap between consecutive detection intervals")
        options_row.addWidget(self.group_gap)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Reload saved scans")
        options_row.addWidget(self.refresh_button)
        self.select_all_matching_button = QPushButton("Select all")
        self.select_all_matching_button.setObjectName("detections_select_all_matching")
        self.select_all_matching_button.setToolTip(
            "Select every detection matching the applied filters, across all pages"
        )
        options_row.addWidget(self.select_all_matching_button)
        self.prev_page_button = QPushButton("◀ Page")
        self.next_page_button = QPushButton("Page ▶")
        options_row.addWidget(self.prev_page_button)
        options_row.addWidget(self.next_page_button)
        self.columns_button = QToolButton()
        self.columns_button.setText("Columns…")
        self.columns_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.columns_menu = QMenu(self.columns_button)
        self.columns_button.setMenu(self.columns_menu)
        options_row.addWidget(self.columns_button)
        options_row.addStretch(1)
        details_layout.addLayout(options_row)
        self.details = QPlainTextEdit()
        self.details.setObjectName("detection_details")
        self.details.setReadOnly(True)
        self.details.setPlainText("Select a result to see its metadata.")
        self.details.setMinimumHeight(90)
        self.details.setMaximumHeight(190)
        details_layout.addWidget(self.details)
        self.details_panel.hide()
        layout.addWidget(self.details_panel)

        self.status_label = QLabel("Select a classifier to browse saved detections.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.count_label = QLabel("0 detections")
        layout.addWidget(self.count_label)

        self.table = DetectionTableView()
        self.table.setObjectName("detections_table")
        self.table_model = DetectionTableModel(self.table)
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu_at)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().setSortIndicator(0, Qt.SortOrder.DescendingOrder)
        for column, width in enumerate((70, 72, 180, 140, 65, 100, 90, 90, 90, 130, 160)):
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setToolTip(
            "Click a row to inspect it; use Up/Down to browse across pages, "
            "Enter to open in Explorer, or click any column heading to sort "
            "the entire filtered detection list (click again to reverse). "
            "Ctrl-click / Shift-click to select multiple rows, even across pages. "
            "Right-click a selected row to label every selected detection; "
            "right-click an unselected row to label only that row."
        )
        # The results table gets the whole width; metadata lives under Details….
        for column in (1, 3, 4, 8, 9, 10):
            self.table.hideColumn(column)
        for column, label in enumerate(self.table_model.HEADERS):
            action = self.columns_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(not self.table.isColumnHidden(column))
            action.toggled.connect(
                lambda checked, col=column: self.table.setColumnHidden(col, not checked)
            )
        layout.addWidget(self.table, 1)

        selection_controls = QHBoxLayout()
        self.selection_count_label = QLabel("0 selected")
        self.selection_count_label.setObjectName("detections_selection_count")
        self.label_button = QPushButton("Label…")
        self.label_button.setObjectName("detections_label_selected")
        self.label_button.setEnabled(False)
        self.label_button.setToolTip("Label selected detections across all pages")
        self.clear_selection_button = QPushButton("Clear")
        self.clear_selection_button.setObjectName("detections_clear_selection")
        self.clear_selection_button.setEnabled(False)
        self.open_button = QPushButton("Open")
        self.open_button.setToolTip("Open the current detection in Explorer")
        self.export_button = QPushButton("Export…")
        self.export_button.setObjectName("detections_export_csv")
        self.export_button.setToolTip("Export all, filtered, or selected detections to CSV")
        self.export_button.setEnabled(False)
        self.cancel_export_button = QPushButton("Cancel export")
        self.cancel_export_button.setEnabled(False)
        self.cancel_export_button.hide()
        selection_controls.addWidget(self.selection_count_label)
        selection_controls.addWidget(self.label_button)
        selection_controls.addWidget(self.clear_selection_button)
        selection_controls.addStretch(1)
        selection_controls.addWidget(self.open_button)
        selection_controls.addWidget(self.export_button)
        selection_controls.addWidget(self.cancel_export_button)
        layout.addLayout(selection_controls)
        self.export_status_label = QLabel("")
        self.export_status_label.setWordWrap(True)
        self.export_status_label.hide()
        layout.addWidget(self.export_status_label)

        footer = QHBoxLayout()
        self.prev_button = QPushButton("◀")
        self.prev_button.setToolTip("Previous detection")
        self.next_button = QPushButton("▶")
        self.next_button.setToolTip("Next detection")
        self.position_label = QLabel("0 / 0")
        footer.addStretch(1)
        footer.addWidget(self.prev_button)
        footer.addWidget(self.position_label)
        footer.addWidget(self.next_button)
        footer.addStretch(1)
        layout.addLayout(footer)

        self.classifier_combo.currentIndexChanged.connect(
            lambda _: self.classifier_selected.emit(self.classifier_combo.currentData() or "")
        )
        self.run_combo.currentIndexChanged.connect(
            lambda _: self.run_selected.emit(self.run_combo.currentData() or "")
        )
        self.generate_button.clicked.connect(self._show_scan_dialog)
        self.refresh_button.clicked.connect(self.refresh_requested)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.cancel_export_button.clicked.connect(self.cancel_export_requested)
        self.apply_button.clicked.connect(self._apply_filters)
        self.recording_filter.returnPressed.connect(self._apply_filters)
        self.table.selectionModel().currentRowChanged.connect(
            lambda current, _: self.detection_row_selected.emit(current.row())
        )
        self.table.selectionModel().selectionChanged.connect(self._emit_page_selection)
        self.select_all_matching_button.clicked.connect(self.select_all_matching_requested)
        self.label_button.clicked.connect(self.label_selected_requested)
        self.clear_selection_button.clicked.connect(self.clear_selection_requested)
        self.table.doubleClicked.connect(lambda _: self.open_requested.emit())
        self.table.open_at_selection.connect(self.open_requested)
        self.table.previous_at_boundary.connect(self.previous_requested)
        self.table.next_at_boundary.connect(self.next_requested)
        self.table.horizontalHeader().sectionClicked.connect(self._sort_column)
        self.group_gap.editingFinished.connect(self._apply_grouping)
        self.details_toggle.toggled.connect(self._toggle_details)
        self.grouping_checkbox.toggled.connect(self._grouping_toggled)
        self.prev_button.clicked.connect(self.previous_requested)
        self.next_button.clicked.connect(self.next_requested)
        self.open_button.clicked.connect(self.open_requested)
        self.prev_page_button.clicked.connect(lambda: self.page_requested.emit(-1))
        self.next_page_button.clicked.connect(lambda: self.page_requested.emit(1))
        self.set_navigation(0, 0, 0, 0)
        self.generate_button.setEnabled(False)
        self.set_filters_enabled(False)
        self.set_grouping_available(False)
        self.set_selection_state(0, available=False)

    def _show_scan_dialog(self):
        if self.scan_dialog.exec() == QDialog.DialogCode.Accepted:
            self.generate_requested.emit(self.scan_threshold.value())

    def _toggle_details(self, expanded):
        self.details_panel.setVisible(expanded)
        self.details_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _grouping_toggled(self, enabled):
        self.group_gap.setEnabled(enabled and self.grouping_checkbox.isEnabled())
        self._apply_grouping()

    def _apply_grouping(self):
        if self.grouping_checkbox.isEnabled():
            self.grouping_requested.emit(self.grouping_checkbox.isChecked(), self.group_gap.value())

    def set_grouping_available(self, available):
        self.grouping_checkbox.setEnabled(available)
        self.group_gap.setEnabled(available and self.grouping_checkbox.isChecked())

    def set_grouping_state(self, enabled, *, message="Grouping off", busy=False):
        with QSignalBlocker(self.grouping_checkbox):
            self.grouping_checkbox.setChecked(bool(enabled))
        self.group_gap.setEnabled(self.grouping_checkbox.isEnabled() and enabled and not busy)
        self.group_status.setText(message)

    def show_selection_label_menu(self, run, refs, targets):
        """Visible bulk labelling, independent of the inspected table row."""
        menu = QMenu(self.label_button)
        menu.addSection(f"Label {len(refs):,} selected detections")

        def add_actions(parent, classifier_id):
            for text, label in (
                ("Positive", BinaryLabel.POSITIVE),
                ("Negative", BinaryLabel.NEGATIVE),
                ("Remove label", None),
            ):
                action = parent.addAction(text)
                action.triggered.connect(
                    lambda _checked=False, cid=classifier_id, choice=label:
                        self.human_labels_requested.emit(
                            refs, run.run_id, run.embedding_space, cid, choice
                        )
                )

        default = next((item for item in targets if item[0] == run.classifier_id), None)
        if default is not None:
            menu.addSection(default[1])
            add_actions(menu, default[0])
        others = [item for item in targets if item[0] != run.classifier_id]
        if others:
            submenu = menu.addMenu("Other classifiers…") if default else menu
            for classifier_id, name in others:
                add_actions(submenu.addMenu(name), classifier_id)
        if not targets:
            action = menu.addAction("No compatible classifiers")
            action.setEnabled(False)
        menu.exec(self.label_button.mapToGlobal(self.label_button.rect().bottomLeft()))

    def _context_menu_at(self, position):
        """Ask the controller which classifiers are compatible with this row."""
        index = self.table.indexAt(position)
        if not index.isValid():
            return
        # Preserve any Ctrl/Shift multi-selection. Right-click changes the
        # inspected row only; it must not silently replace the bulk selection.
        with QSignalBlocker(self.table.selectionModel()):
            self.table.selectionModel().setCurrentIndex(
                index, QItemSelectionModel.SelectionFlag.NoUpdate
            )
        self.context_menu_requested.emit(
            index.row(), self.table.viewport().mapToGlobal(position)
        )

    def _emit_page_selection(self, *_):
        selected = tuple(
            self.table_model.rows[index.row()].ref
            for index in self.table.selectionModel().selectedRows()
            if 0 <= index.row() < len(self.table_model.rows)
        )
        self.selected_page_refs_changed.emit(selected)

    def set_selection_state(self, count, *, available=True, busy=False):
        self.selection_count_label.setText(
            f"{count:,} selected" + (" · selecting…" if busy else "")
        )
        self.clear_selection_button.setEnabled(count > 0 or busy)
        self.label_button.setEnabled(available and count > 0 and not busy)
        self.select_all_matching_button.setEnabled(available and not busy)

    def set_context_group(self, group):
        self._context_group = group

    def set_context_label_refs(self, refs, *, outside_selection=False):
        """Snapshot which detections the existing label actions will affect."""
        self._context_label_refs = frozenset(refs)
        self._context_outside_selection = bool(outside_selection)

    def move_current_to_row(self, row):
        """Navigate the inspector without changing the bulk selection."""
        index = self.table_model.index(row, 0)
        if index.isValid():
            self.table.selectionModel().setCurrentIndex(
                index, QItemSelectionModel.SelectionFlag.NoUpdate
            )
            self.table.scrollTo(index)

    def restore_bulk_selection(self, rows, selected_refs):
        """Update highlights without replacing the current inspected row."""
        selected = set(selected_refs)
        selection_model = self.table.selectionModel()
        with QSignalBlocker(selection_model):
            selection_model.clearSelection()
            selection = QItemSelection()
            for row_index, detection in enumerate(rows):
                if detection.ref in selected:
                    selection.select(
                        self.table_model.index(row_index, 0),
                        self.table_model.index(row_index, len(self.table_model.FIELDS) - 1),
                    )
            if not selection.isEmpty():
                selection_model.select(
                    selection, QItemSelectionModel.SelectionFlag.Select
                )

    def show_human_label_menu(self, global_position, detection, run, targets):
        """The scan's classifier gets one-click actions; others use submenus."""
        menu = QMenu(self.table)
        context_refs = self._context_label_refs or frozenset((detection.ref,))
        multiple = len(context_refs) > 1
        group = self._context_group
        if group is not None:
            group_action = menu.addAction(
                f"Select entire temporal group {group.group_id} "
                f"({group.detection_count:,} detections)"
            )
            group_action.triggered.connect(
                lambda _checked=False, gid=group.group_id:
                    self.select_group_requested.emit(gid, False)
            )
            matching_action = menu.addAction("Select matching members of this group")
            matching_action.triggered.connect(
                lambda _checked=False, gid=group.group_id:
                    self.select_group_requested.emit(gid, True)
            )
            menu.addSeparator()
        clear_action = menu.addAction("Clear bulk selection")
        clear_action.triggered.connect(self.clear_selection_requested)
        menu.addSeparator()

        if multiple:
            menu.addSection(f"Labelling {len(context_refs):,} selected detections")
        elif self._context_outside_selection:
            menu.addSection("This detection only (outside selection)")

        def emit_label(classifier_id, label):
            if multiple:
                self.human_labels_requested.emit(
                    context_refs, run.run_id, run.embedding_space,
                    classifier_id, label,
                )
            else:
                self.human_label_requested.emit(
                    detection.ref, run.embedding_space, classifier_id, label
                )

        def add_actions(parent, classifier_id, existing_label):
            for text, label in (
                ("Positive", BinaryLabel.POSITIVE),
                ("Negative", BinaryLabel.NEGATIVE),
                ("Remove human label", None),
            ):
                action = parent.addAction(text)
                # For a mixed selection, the clicked row's label is not
                # representative of the entire batch.
                action.setCheckable(label is not None and not multiple)
                if not multiple and label is not None:
                    action.setChecked(existing_label is label)
                if label is None:
                    # A selected batch can contain labels even if the clicked
                    # detection is unlabelled; don't disable Remove in that case.
                    action.setEnabled(multiple or existing_label is not None)
                action.triggered.connect(
                    lambda _checked=False, cid=classifier_id, choice=label:
                        emit_label(cid, choice)
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
        if new_sort == "group" and not self.grouping_checkbox.isChecked():
            self.group_status.setText("Enable temporal grouping before sorting by Group.")
            self.set_sort_indicator(self._sort_by, self._descending)
            return
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

    def show_page(self, rows, metadata, *, selected_row=None, groups=None,
                  selected_refs=()):
        # Model resets and programmatic selection must not look like a user's
        # row click. The controller explicitly updates the inspector after
        # populating the page and only requests audio on user navigation.
        with QSignalBlocker(self.table.selectionModel()):
            self.table_model.set_page(rows, metadata, groups)
            selected = set(selected_refs)
            selection = QItemSelection()
            for row_index, detection in enumerate(rows):
                if detection.ref in selected:
                    selection.select(
                        self.table_model.index(row_index, 0),
                        self.table_model.index(row_index, len(self.table_model.FIELDS) - 1),
                    )
            if not selection.isEmpty():
                self.table.selectionModel().select(
                    selection, QItemSelectionModel.SelectionFlag.Select
                )
            if selected_row is not None and 0 <= selected_row < len(rows):
                index = self.table_model.index(selected_row, 0)
                self.table.selectionModel().setCurrentIndex(
                    index, QItemSelectionModel.SelectionFlag.NoUpdate
                )
                self.table.scrollTo(index)

    def set_navigation(self, position, total, page, page_count):
        self.position_label.setText(f"{position:,} / {total:,}   ·   Page {page + 1} / {max(1, page_count)}")
        self.count_label.setText(f"{total:,} matching detections")
        self.prev_button.setEnabled(position > 1)
        self.next_button.setEnabled(0 < position < total)
        self.prev_page_button.setEnabled(page > 0)
        self.next_page_button.setEnabled(page + 1 < page_count)
        self.open_button.setEnabled(position > 0)

    def show_details(self, run, detection, metadata, *, group=None):
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
        if group is not None:
            lines.extend((
                "", f"Temporal group: {group.group_id}",
                f"Group detections: {group.detection_count:,}",
                f"Group interval (nominal): {group.start_s:.2f}–{group.end_s:.2f} s",
            ))
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

    def choose_csv_export(self, run, matching_count, selected_count=0, *, filters_active=False):
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
        selected_radio = QRadioButton(
            f"Selected detections ({selected_count:,}, across all pages)"
        )
        selected_radio.setEnabled(selected_count > 0)
        if selected_count > 0:
            selected_radio.setChecked(True)
        elif filters_active:
            filtered_radio.setChecked(True)
        else:
            all_radio.setChecked(True)
        contents.addWidget(all_radio)
        contents.addWidget(filtered_radio)
        contents.addWidget(selected_radio)
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
        scope = ("selected" if selected_radio.isChecked() else
                 "filtered" if filtered_radio.isChecked() else "all")
        return destination, scope

    def set_export_state(self, *, available, running, cancelling=False):
        self.export_button.setEnabled(bool(available and not running))
        self.cancel_export_button.setEnabled(bool(running and not cancelling))
        self.cancel_export_button.setVisible(bool(running))

    def show_export_status(self, message):
        self.export_status_label.setText(message)
        self.export_status_label.setVisible(bool(message))
