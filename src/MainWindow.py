"""Top-level application shell with individually dockable workspaces.

Each MVC view lives in a movable dock whose content scrolls independently
when the user shrinks the main window or splits several workspaces.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QToolBar,
    QWidget,
)

from src.PipelineView import PipelineView


class MainWindow(QMainWindow):
    """Application shell containing the application's independent MVC views.

    The shell owns layout only. Explorer, pipeline, active-learning, and
    terminal behaviour remains inside their existing MVC objects.
    """

    def __init__(
        self,
        explorer_view: QWidget,
        pipeline_view: QWidget | None = None,
        active_learning_view: QWidget | None = None,
        terminal_view: QWidget | None = None,
        parent=None,
    ):
        super().__init__(parent)

        if not isinstance(explorer_view, QWidget):
            raise TypeError(
                "explorer_view must be a QWidget"
            )

        if pipeline_view is None:
            pipeline_view = PipelineView()

        if not isinstance(pipeline_view, QWidget):
            raise TypeError(
                "pipeline_view must be a QWidget"
            )

        if (
            active_learning_view is not None
            and not isinstance(active_learning_view, QWidget)
        ):
            raise TypeError(
                "active_learning_view must be a QWidget or None"
            )

        if (
            terminal_view is not None
            and not isinstance(terminal_view, QWidget)
        ):
            raise TypeError(
                "terminal_view must be a QWidget or None"
            )

        self.explorer_view = explorer_view
        self.pipeline_view = pipeline_view
        self.active_learning_view = active_learning_view
        self.terminal_view = terminal_view

        self._view_docks: dict[int, QDockWidget] = {}
        self._workspace_docks: list[QDockWidget] = []

        # Native QDockWidgets, but without tabification (which hides the
        # draggable title bars) or floating windows (which can end up behind
        # the application). Avoid animations when resizing heavy Explorer panes.
        self.setDockOptions(QMainWindow.DockOption.AllowNestedDocks)

        self.explorer_dock = self._add_workspace(
            title="Explorer",
            object_name="explorer_workspace_dock",
            view=self.explorer_view,
        )
        self.pipeline_dock = self._add_workspace(
            title="Pipelines",
            object_name="pipelines_workspace_dock",
            view=self.pipeline_view,
        )

        self.active_learning_dock = None
        if self.active_learning_view is not None:
            self.active_learning_dock = self._add_workspace(
                title="Active Learning",
                object_name="active_learning_workspace_dock",
                view=self.active_learning_view,
            )

        self.terminal_dock = None
        if self.terminal_view is not None:
            self.terminal_dock = self._add_workspace(
                title="Terminal",
                object_name="terminal_workspace_dock",
                view=self.terminal_view,
            )

        self._create_view_menu()
        self.reset_workspace_layout()

        self.setWindowTitle(
            "Caracal"
        )

        # The preferred opening size must also fit on smaller displays.
        # Dock combinations must never force the shell to grow to fit their
        # content; the scroll areas absorb that overflow instead.
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(1450, max(1, int(available.width() * 0.9))),
                min(900, max(1, int(available.height() * 0.9))),
            )
        else:
            self.resize(1450, 900)

    def _add_workspace(
        self,
        title: str,
        object_name: str,
        view: QWidget,
    ) -> QDockWidget:
        """Wrap one top-level application view in a movable dock."""

        dock = QDockWidget(
            title,
            self,
        )
        dock.setObjectName(
            object_name
        )
        dock.setAllowedAreas(
            Qt.DockWidgetArea.AllDockWidgetAreas
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        # A dock's child determines its minimum size. Give the dock a
        # shrinkable scroll area instead of the potentially large MVC view;
        # otherwise one view can prevent the whole window from being resized.
        scroll_area = QScrollArea(dock)
        scroll_area.setObjectName(f"{object_name}_scroll_area")
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # Keep a small, usable dock footprint. ``Ignored`` plus a zero
        # minimum lets Qt collapse a dock until its title bar/drop target is
        # effectively unusable. ``Expanding`` still allows the scroll area
        # to shrink below the view's preferred size; its scrollbars handle
        # the overflowing content instead.
        scroll_area.setMinimumSize(160, 100)
        scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        scroll_area.setWidget(view)
        dock.setWidget(scroll_area)

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            dock,
        )

        self._workspace_docks.append(
            dock
        )
        self._view_docks[id(view)] = dock

        return dock

    def _create_view_menu(self):
        """Keep every workspace accessible even when its dock is hidden."""

        view_menu = self.menuBar().addMenu("&View")
        workspace_toolbar = QToolBar("Workspaces", self)
        workspace_toolbar.setObjectName("workspace_switcher_toolbar")
        workspace_toolbar.setMovable(False)
        workspace_toolbar.setFloatable(False)
        workspace_toolbar.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.addToolBar(workspace_toolbar)
        self.workspace_toolbar = workspace_toolbar

        for dock in self._workspace_docks:
            action = dock.toggleViewAction()
            view_menu.addAction(action)
            workspace_toolbar.addAction(action)

        view_menu.addSeparator()

        reset_action = QAction(
            "Reset workspace layout",
            self,
        )
        reset_action.triggered.connect(
            self.reset_workspace_layout
        )
        view_menu.addAction(
            reset_action
        )
        self.reset_layout_action = reset_action

    def reset_workspace_layout(self):
        """Show Explorer and Active Learning side by side with title bars.

        Pipelines and Terminal start hidden to avoid squeezing the two larger
        views; users can reveal them from the Workspaces toolbar or View menu.
        """
        for dock in self._workspace_docks:
            dock.setFloating(False)
            self.removeDockWidget(dock)

        # Register the docks before splitting to keep Qt's dock area bookkeeping
        # consistent when the user resets after moving/hiding workspaces.
        for dock in self._workspace_docks:
            self.addDockWidget(
                Qt.DockWidgetArea.LeftDockWidgetArea,
                dock,
            )

        secondary = (
            self.active_learning_dock
            or self.pipeline_dock
            or self.terminal_dock
        )
        self.splitDockWidget(
            self.explorer_dock,
            secondary,
            Qt.Orientation.Horizontal,
        )

        # Hidden docks are pre-positioned; revealing one creates a split,
        # rather than a tab group or a separate top-level window.
        if self.pipeline_dock is not secondary:
            self.splitDockWidget(
                secondary,
                self.pipeline_dock,
                Qt.Orientation.Vertical,
            )
        if self.terminal_dock is not None and self.terminal_dock is not secondary:
            self.splitDockWidget(
                self.explorer_dock,
                self.terminal_dock,
                Qt.Orientation.Vertical,
            )

        for dock in self._workspace_docks:
            dock.setVisible(
                dock is self.explorer_dock or dock is secondary
            )

        QTimer.singleShot(0, self._size_default_workspaces)

    def _size_default_workspaces(self):
        """Give Explorer slightly more width when its partner is visible."""
        secondary = (
            self.active_learning_dock
            or self.pipeline_dock
            or self.terminal_dock
        )
        if secondary.isVisible() and self.explorer_dock.isVisible():
            self.resizeDocks(
                [self.explorer_dock, secondary],
                [850, 650],
                Qt.Orientation.Horizontal,
            )

    def focus_view(
        self,
        view: QWidget,
    ):
        """Reveal and select one composed application view."""

        dock = self._view_docks.get(
            id(view)
        )
        if dock is None:
            raise ValueError(
                "view is not managed by this MainWindow"
            )

        dock.show()
        dock.raise_()
        view.setFocus(
            Qt.FocusReason.OtherFocusReason
        )

    def closeEvent(
        self,
        event,
    ):
        """Allow composed views to release background resources cleanly."""

        shutdown = getattr(
            self.explorer_view,
            "shutdown",
            None,
        )

        if callable(shutdown):
            shutdown()

        super().closeEvent(
            event
        )
