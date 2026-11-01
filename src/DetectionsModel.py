"""Qt-independent state and paging rules for the Detections workspace."""

from dataclasses import dataclass, field

from src.Detection import Detection, DetectionRun


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
        self.run = None
        self.total = 0
        self.page = 0
        self.rows = ()
        self.selected_index = None
        self.min_score = None
        self.recording_id = None
