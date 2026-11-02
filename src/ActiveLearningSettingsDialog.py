"""Small modal editor for a classifier's existing training/review parameters."""

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.ActiveLearner import ActiveLearningParameters
from src.ActiveLearningClassifierDefinition import ActiveLearningClassifierDefinition
from src.BinaryEmbeddingClassifier import BinaryClassifierParameters


class ActiveLearningSettingsDialog(QDialog):
    """Edit training and candidate-selection settings for one classifier.

    The dialog collects values but never mutates the classifier definition or
    its fitted weights. Persistence and invalidation belong to the controller.
    """

    def __init__(self, definition: ActiveLearningClassifierDefinition, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure classifier — {definition.name}")
        self.setMinimumWidth(440)
        self.setModal(True)
        self.parameters = None

        model = definition.classifier_parameters
        policy = definition.active_learning_parameters
        tabs = QTabWidget()

        training_tab = QWidget()
        training = QFormLayout(training_tab)
        self.hidden_enabled = QCheckBox("Use hidden layer (MLP)")
        self.hidden_enabled.setChecked(model.hidden_dimension is not None)
        self.hidden_dimension = self._integer(model.hidden_dimension or 128, 1, 1000000)
        self.hidden_dimension.setEnabled(self.hidden_enabled.isChecked())
        self.hidden_enabled.toggled.connect(self.hidden_dimension.setEnabled)
        training.addRow("Architecture:", self.hidden_enabled)
        training.addRow("Hidden units:", self.hidden_dimension)
        self.epochs = self._integer(model.epochs, 1, 1000000)
        training.addRow("Epochs:", self.epochs)
        self.learning_rate = self._decimal(model.learning_rate, 0.00000001, 10.0, 8, 0.0001)
        training.addRow("Learning rate:", self.learning_rate)
        self.batch_size = self._integer(model.batch_size, 1, 1000000)
        training.addRow("Training batch size:", self.batch_size)
        self.negative_ratio = self._integer(model.negative_to_positive_ratio, 1, 1000000)
        training.addRow("Negatives per positive:", self.negative_ratio)
        self.model_seed = self._integer(model.random_seed, -2147483648, 2147483647)
        training.addRow("Training seed:", self.model_seed)
        self.device = QLineEdit(model.device)
        self.device.setToolTip("PyTorch device, e.g. cpu, cuda or mps (if available)")
        training.addRow("Device:", self.device)
        tabs.addTab(training_tab, "Model training")

        review_tab = QWidget()
        review = QFormLayout(review_tab)
        self.queue_size = self._integer(policy.review_queue_size, 1, 1000000)
        review.addRow("Review queue size:", self.queue_size)
        help_text = QLabel(
            "Selection fractions are relative weights; they are normalised "
            "automatically. At least one must be greater than zero."
        )
        help_text.setWordWrap(True)
        review.addRow(help_text)
        self.uncertainty_fraction = self._decimal(policy.uncertainty_fraction, 0, 100, 3, 0.05)
        self.positive_fraction = self._decimal(policy.likely_positive_fraction, 0, 100, 3, 0.05)
        self.random_fraction = self._decimal(policy.random_fraction, 0, 100, 3, 0.05)
        review.addRow("Uncertain weight:", self.uncertainty_fraction)
        review.addRow("Likely-positive weight:", self.positive_fraction)
        review.addRow("Random weight:", self.random_fraction)
        self.boundary = self._decimal(policy.uncertainty_decision_boundary, 0, 1, 3, 0.05)
        review.addRow("Decision boundary:", self.boundary)
        self.scoring_batch = self._integer(policy.scoring_batch_size, 1, 1000000)
        review.addRow("Scoring batch size:", self.scoring_batch)
        self.review_seed = self._integer(policy.random_seed, -2147483648, 2147483647)
        review.addRow("Sampling seed:", self.review_seed)
        tabs.addTab(review_tab, "Review selection")

        note = QLabel(
            "Settings are saved for this classifier. Changing model training "
            "settings invalidates its fitted weights. Any change clears the "
            "current review queue; prepare the pipeline again to apply it."
        )
        note.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(note)
        layout.addWidget(buttons)

    @staticmethod
    def _integer(value, minimum, maximum):
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _decimal(value, minimum, maximum, decimals, step):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def accept(self):
        try:
            model = BinaryClassifierParameters(
                hidden_dimension=(
                    self.hidden_dimension.value() if self.hidden_enabled.isChecked()
                    else None
                ),
                epochs=self.epochs.value(),
                learning_rate=self.learning_rate.value(),
                batch_size=self.batch_size.value(),
                negative_to_positive_ratio=self.negative_ratio.value(),
                random_seed=self.model_seed.value(),
                device=self.device.text().strip(),
            )
            policy = ActiveLearningParameters(
                review_queue_size=self.queue_size.value(),
                uncertainty_fraction=self.uncertainty_fraction.value(),
                likely_positive_fraction=self.positive_fraction.value(),
                random_fraction=self.random_fraction.value(),
                uncertainty_decision_boundary=self.boundary.value(),
                scoring_batch_size=self.scoring_batch.value(),
                random_seed=self.review_seed.value(),
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid classifier settings", str(exc))
            return

        self.parameters = (model, policy)
        super().accept()
