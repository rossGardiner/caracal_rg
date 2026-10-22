"""Top-level application shell with individually dockable workspaces.

Each MVC view lives in a movable dock whose content scrolls independently
when the user shrinks the main window or splits several workspaces.
"""

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QWidget,
)

from src.PipelineView import PipelineView


class _WorkspaceTitleBar(QWidget):
    """Keep native dock dragging; explicitly detach on double-click.

    Mouse events other than the double-click are ignored so QDockWidget
    continues handling its normal dock-and-split drag gestures.
    """

    def __init__(self, dock: QDockWidget, detach):
        super().__init__(dock)
        self._detach = detach

        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 1, 3, 1)
        layout.setSpacing(6)

        title = QLabel(dock.windowTitle(), self)
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(title, 1)

        close = QToolButton(self)
        close.setText("×")
        close.setToolTip("Hide workspace")
        close.clicked.connect(dock.hide)
        layout.addWidget(close)
        self._close_button = close

    def set_detached(self, detached: bool):
        """Hide the dock's close control while in its own window."""
        self._close_button.setVisible(not detached)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._detach()
            event.accept()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()


class _DetachedWorkspaceWindow(QMainWindow):
    """A regular OS-managed window, not a floating QDockWidget tool window."""

    def __init__(self, owner: "MainWindow", dock: QDockWidget):
        # No QWidget parent: this is a real top-level window, with its own
        # taskbar/window-manager presence and independent monitor placement.
        super().__init__(None, Qt.WindowType.Window)
        self._owner = owner
        self._dock = dock
        self.setWindowTitle(f"{dock.windowTitle()} — Caracal")

        toolbar = QToolBar("Workspace", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        dock_back = QAction("Dock back", self)
        dock_back.triggered.connect(lambda: owner._redock_workspace(dock))
        toolbar.addAction(dock_back)

    def closeEvent(self, event):
        if not self._owner._closing:
            # Closing a popped-out workspace returns it to the main window;
            # the view and its controller are not destroyed.
            self._owner._redock_workspace(self._dock)
        super().closeEvent(event)


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
        self._workspace_actions: dict[QDockWidget, QAction] = {}
        self._detached_windows: dict[QDockWidget, _DetachedWorkspaceWindow] = {}
        self._dock_return_areas: dict[QDockWidget, Qt.DockWidgetArea] = {}
        self._closing = False

        # Keep ordinary docking/splitting native. Pop-out workspaces use real
        # top-level windows rather than floating QDockWidget tool windows.
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
        dock.setTitleBarWidget(
            _WorkspaceTitleBar(dock, lambda d=dock: self._detach_workspace(d))
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
            action = QAction(dock.windowTitle(), self)
            action.setCheckable(True)
            action.toggled.connect(
                lambda visible, d=dock: self._set_workspace_visible(d, visible)
            )
            dock.visibilityChanged.connect(
                lambda _visible, d=dock: self._sync_workspace_action(d)
            )
            self._workspace_actions[dock] = action
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

    def _sync_workspace_action(self, dock: QDockWidget):
        action = self._workspace_actions.get(dock)
        if action is None:
            return
        detached = self._detached_windows.get(dock)
        visible = detached.isVisible() if detached is not None else dock.isVisible()
        action.blockSignals(True)
        action.setChecked(visible)
        action.blockSignals(False)

    def _set_workspace_visible(self, dock: QDockWidget, visible: bool):
        detached = self._detached_windows.get(dock)
        if detached is not None:
            if visible:
                detached.showNormal()
                detached.raise_()
                detached.activateWindow()
            else:
                detached.hide()
        else:
            dock.setVisible(visible)
            if visible:
                dock.raise_()
        self._sync_workspace_action(dock)

    def _detach_workspace(self, dock: QDockWidget):
        """Move an existing workspace into its own ordinary top-level window."""
        existing = self._detached_windows.get(dock)
        if existing is not None:
            self._redock_workspace(dock)
            return

        # Move the entire QDockWidget, including its QScrollArea and MVC view.
        # QDockWidget has no takeWidget() API: setWidget(None) is *not* a safe
        # ownership transfer and caused blank/destroyed detached workspaces.
        origin = dock.mapToGlobal(QPoint(0, 0))
        width, height = dock.width(), dock.height()
        area = self.dockWidgetArea(dock)
        self._dock_return_areas[dock] = (
            area if area != Qt.DockWidgetArea.NoDockWidgetArea
            else Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.removeDockWidget(dock)  # hides, but does not delete the dock

        window = _DetachedWorkspaceWindow(self, dock)
        window.setCentralWidget(dock)  # transfers the whole intact dock
        self._detached_windows[dock] = window

        # This dock is now regular content inside the standalone window.
        # Move the *outer* OS window, not the inner dock, when dragging.
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        title_bar = dock.titleBarWidget()
        if isinstance(title_bar, _WorkspaceTitleBar):
            title_bar.set_detached(True)
        dock.show()  # removeDockWidget() hid the dock

        screen = QGuiApplication.screenAt(origin) or self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(max(480, width), available.width())
            height = min(max(320, height), available.height())
            window.resize(width, height)
            window.move(
                max(available.left(), min(origin.x() + 24, available.right() - width + 1)),
                max(available.top(), min(origin.y() + 24, available.bottom() - height + 1)),
            )
        else:
            window.resize(max(480, width), max(320, height))
            window.move(origin + QPoint(24, 24))

        window.show()
        window.raise_()
        window.activateWindow()
        self._sync_workspace_action(dock)

    def _redock_workspace(self, dock: QDockWidget):
        """Return the intact dock and its existing view to the main window."""
        window = self._detached_windows.pop(dock, None)
        if window is None:
            return

        # takeCentralWidget() explicitly transfers ownership, unlike
        # QDockWidget.setWidget(None). The scroll area never leaves its dock.
        returned_dock = window.takeCentralWidget()
        if returned_dock is not dock:
            raise RuntimeError("Detached window lost its workspace dock")

        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        title_bar = dock.titleBarWidget()
        if isinstance(title_bar, _WorkspaceTitleBar):
            title_bar.set_detached(False)

        area = self._dock_return_areas.pop(
            dock, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(area, dock)
        dock.show()
        dock.raise_()
        window.hide()
        window.deleteLater()
        self._sync_workspace_action(dock)

    def reset_workspace_layout(self):
        """Show Explorer and Active Learning side by side with title bars.

        Pipelines and Terminal start hidden to avoid squeezing the two larger
        views; users can reveal them from the Workspaces toolbar or View menu.
        """
        for dock in tuple(self._detached_windows):
            self._redock_workspace(dock)

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
        for dock in self._workspace_docks:
            self._sync_workspace_action(dock)

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

        detached = self._detached_windows.get(dock)
        if detached is not None:
            detached.showNormal()
            detached.raise_()
            detached.activateWindow()
        else:
            dock.show()
            dock.raise_()
        self._sync_workspace_action(dock)
        view.setFocus(
            Qt.FocusReason.OtherFocusReason
        )

    def closeEvent(
        self,
        event,
    ):
        """Allow composed views to release background resources cleanly."""

        self._closing = True
        for window in tuple(self._detached_windows.values()):
            window.close()

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
