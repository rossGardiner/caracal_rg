"""Qt-independent state and paging rules for the Detections workspace."""

from dataclasses import dataclass, field

from src.Detection import Detection, DetectionRun
from src.EmbeddingCorpus import EmbeddingRef


PAGE_SIZE = 200


@dataclass
class DetectionsModel:
    classifier_id: str | None = None
    run: DetectionRun | None = None
    min_score: float | None = None
    recording_id: str | None = None
    sort_by: str = "score"
    descending: bool = True
    total: int = 0
    page: int = 0
    rows: tuple[Detection, ...] = field(default_factory=tuple)
    selected_index: int | None = None
    scanning: bool = False
    status: str = "Select a classifier to browse saved detections."
    grouping_enabled: bool = False
    grouping_gap_s: float = 10.0
    grouping: object | None = None
    # The current inspected row is independent of the set selected for future
    # bulk operations. A selection is always scoped to a single saved run.
    selected_refs: set[EmbeddingRef] = field(default_factory=set)

    def replace_visible_selection(self, visible_refs, selected_refs):
        """Update just the current page; retain selections on other pages."""
        visible = set(visible_refs)
        selected = set(selected_refs)
        if not selected <= visible:
            raise ValueError("selected refs must belong to the visible page")
        self.selected_refs.difference_update(visible)
        self.selected_refs.update(selected)

    def label_targets_for(self, clicked_ref):
        """Right-click labels the selection only when clicked inside it.

        Preserve the existing selection across pages and filters, while
        preventing an unselected right-click from silently labelling unrelated
        rows. The set is a snapshot for use while the context menu is open.
        """
        if not isinstance(clicked_ref, EmbeddingRef):
            raise TypeError("clicked_ref must be an EmbeddingRef")
        if clicked_ref in self.selected_refs and len(self.selected_refs) > 1:
            return frozenset(self.selected_refs)
        return frozenset((clicked_ref,))

    def add_selection(self, refs):
        self.selected_refs.update(refs)

    def clear_selection(self):
        self.selected_refs.clear()

    @property
    def selection_count(self):
        return len(self.selected_refs)

    def clear_grouping(self):
        self.grouping_enabled = False
        self.grouping = None
        if self.sort_by == "group":
            self.sort_by, self.descending = "score", True

    @property
    def selected(self):
        if self.selected_index is None:
            return None
        local_index = self.selected_index - self.page * PAGE_SIZE
        if 0 <= local_index < len(self.rows):
            return self.rows[local_index]
        return None

    def set_page(self, page, rows, total):
        page = int(page)
        total = int(total)
        rows = tuple(rows)
        if page < 0 or total < 0:
            raise ValueError("Invalid detection page")
        if len(rows) > PAGE_SIZE:
            raise ValueError("Detection page exceeds PAGE_SIZE")
        if any(not isinstance(row, Detection) for row in rows):
            raise TypeError("Detection page must contain Detection objects")
        self.page, self.rows, self.total = page, rows, total
        self.selected_index = None

    def select_local_row(self, row):
        if not 0 <= row < len(self.rows):
            self.selected_index = None
            return None
        self.selected_index = self.page * PAGE_SIZE + row
        return self.rows[row]

    @property
    def position(self):
        return 0 if self.selected_index is None else self.selected_index + 1

    def clear_results(self):
        self.clear_selection()
        self.clear_grouping()
        self.run = None
        self.total = 0
        self.page = 0
        self.rows = ()
        self.selected_index = None
        self.min_score = None
        self.recording_id = None
