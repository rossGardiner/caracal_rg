"""Interactive audio and embedding explorer entry point."""

import argparse
import base64
from functools import partial
from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QByteArray, QProcess, QTimer, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

from src.ApplicationTheme import apply_application_theme
from src.ActiveLearner import ReviewCandidate, ReviewReason
from src.EmbeddingCorpus import EmbeddingRef
from src.ActiveLearningModel import ActiveLearningStatus
from src.LatestJobRunner import LatestJobRunner
from src.ProjectFile import ProjectFile
from src.ActiveLearningClassifierStore import ActiveLearningClassifierStore
from src.ActiveLearningController import ActiveLearningController
from src.ActiveLearningModel import ActiveLearningModel
from src.ActiveLearningView import ActiveLearningView
from src.QuickReviewController import QuickReviewController
from src.QuickReviewView import QuickReviewView
from src.ExplorerController import ExplorerController
from src.DetectionStore import DetectionStore
from src.DetectionsModel import DetectionsModel
from src.DetectionsView import DetectionsView
from src.DetectionsController import DetectionsController
from src.ExplorerView import ExplorerView
from src.ExplorerModel import ExplorerModel
from src.MainWindow import MainWindow
from src.PipelineController import PipelineController
from src.PipelineDefinitions import get_pipeline_definitions
from src.PipelineDefinitionStore import PipelineDefinitionStore
from src.PipelineFactory import PipelineFactory
from src.PipelineRunner import PipelineRunner
from src.PipelineRunModel import PipelineRunModel
from src.PipelineView import PipelineView
from src.QtSignalHandler import QtSignalHandler
from src.TerminalView import TerminalView


def main():
    """Open the application idle; build pipelines only after a user action."""

    args_parser = argparse.ArgumentParser(add_help=False)
    args_parser.add_argument("--project", default=None)
    options, qt_args = args_parser.parse_known_args(sys.argv[1:])

    # A project is always isolated from the user's regular saved definitions.
    # Nothing is imported over global configuration. An opened project lives
    # in a session directory and is persisted only with Save Project.
    project_directory = None
    project_path = Path(options.project).expanduser().resolve() if options.project else None
    project_payload = None
    if project_path is not None:
        project_directory = tempfile.TemporaryDirectory(prefix="caracal-open-project-")
        try:
            project_payload = ProjectFile.open_into(project_path, project_directory.name)
        except Exception as exc:
            project_directory.cleanup()
            error_app = QApplication([sys.argv[0]])
            QMessageBox.critical(None, "Could not open project", f"{project_path}\n\n{exc}")
            return 1

    app = QApplication([sys.argv[0], *qt_args])
    apply_application_theme(app)
    session = project_payload["session"] if project_payload is not None else {}

    terminal_view = TerminalView()
    terminal_view.start_capture()

    builtin_pipeline_definitions = get_pipeline_definitions()
    pipeline_factory = PipelineFactory()
    definition_store = PipelineDefinitionStore(
        Path(project_directory.name) / "pipelines.json"
        if project_directory is not None else None
    )
    try:
        saved_pipeline_definitions = definition_store.load()
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"Could not load saved pipeline definitions from "
            f"{definition_store.path}: {exc}",
            file=sys.stderr,
        )
        saved_pipeline_definitions = ()

    builtin_names = {
        definition.name
        for definition in builtin_pipeline_definitions
    }
    if project_payload is not None:
        try:
            for definition in saved_pipeline_definitions:
                if definition.name in builtin_names:
                    raise ValueError(f"Project pipeline {definition.name!r} conflicts with a built-in pipeline")
                pipeline_factory.validate_definition(definition)
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(None, "Project incompatible", str(exc))
            terminal_view.shutdown()
            return 1
    user_pipeline_definitions = []
    for definition in saved_pipeline_definitions:
        if definition.name in builtin_names:
            print(
                f"Ignoring saved pipeline {definition.name!r}: "
                "the name is reserved by a built-in pipeline",
                file=sys.stderr,
            )
            continue

        try:
            pipeline_factory.validate_definition(definition)
        except (TypeError, ValueError) as exc:
            print(
                f"Ignoring saved pipeline {definition.name!r}: {exc}",
                file=sys.stderr,
            )
            continue

        user_pipeline_definitions.append(definition)

    user_pipeline_definitions = tuple(user_pipeline_definitions)
    pipeline_definitions = (
        tuple(builtin_pipeline_definitions)
        + user_pipeline_definitions
    )
    classifier_store = ActiveLearningClassifierStore(
        Path(project_directory.name) / "classifiers.json"
        if project_directory is not None else None
    )
    try:
        classifier_definitions = classifier_store.load()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        if project_payload is not None:
            QMessageBox.critical(None, "Project incompatible", str(exc))
            terminal_view.shutdown()
            return 1
        print(
            f"Could not load saved active-learning classifiers from "
            f"{classifier_store.path}: {exc}",
            file=sys.stderr,
        )
        classifier_definitions = ()

    signal_handler = QtSignalHandler(
        app
    )

    # Keep a local reference for the lifetime of the QApplication. Qt's
    # signal handler object is otherwise intentionally unused in Python code.
    _ = signal_handler

    explorer_model = ExplorerModel.empty()

    explorer_view = ExplorerView(
        model=explorer_model,
    )
    explorer_controller = ExplorerController(
        model=explorer_model,
        view=explorer_view,
        parent=explorer_view,
    )
    explorer_controller.initialize(
        packet_index=explorer_view.initial_packet_index()
    )

    pipeline_run_model = PipelineRunModel(
        available_pipelines=tuple(
            definition.name
            for definition in pipeline_definitions
        ),
    )
    pipeline_view = PipelineView()
    pipeline_runner = PipelineRunner()
    pipeline_controller = PipelineController(
        model=pipeline_run_model,
        view=pipeline_view,
        runner=pipeline_runner,
        pipeline_factory=pipeline_factory,
        builtin_pipeline_definitions=builtin_pipeline_definitions,
        user_pipeline_definitions=user_pipeline_definitions,
        definition_store=definition_store,
    )

    active_learning_model = ActiveLearningModel(
        available_pipelines=tuple(
            definition.name
            for definition in pipeline_controller.pipeline_definition_snapshot()
        ),
        classifier_definitions=classifier_definitions,
    )
    active_learning_view = ActiveLearningView()
    active_learning_controller = ActiveLearningController(
        model=active_learning_model,
        view=active_learning_view,
        pipeline_factory=pipeline_factory,
        pipeline_definitions=pipeline_controller.pipeline_definition_snapshot(),
        classifier_store=classifier_store,
        parent=active_learning_view,
    )
    pipeline_controller.pipeline_definitions_changed.connect(
        active_learning_controller.set_pipeline_definitions
    )

    explorer_controller.embedding_selection_changed.connect(
        active_learning_controller.set_explorer_selection
    )
    explorer_space, explorer_ref = (
        explorer_controller.current_embedding_selection()
    )
    active_learning_controller.set_explorer_selection(
        explorer_space,
        explorer_ref,
    )

    pipeline_controller.pipeline_activation_ready.connect(
        explorer_controller.activate_pipeline
    )
    explorer_controller.pipeline_activation_succeeded.connect(
        pipeline_controller.set_active_pipeline
    )
    explorer_controller.pipeline_activation_failed.connect(
        pipeline_controller.pipeline_activation_failed
    )

    detections_model = DetectionsModel()
    detections_view = DetectionsView()
    detections_store = DetectionStore(
        Path(project_directory.name) / ProjectFile.DETECTIONS
        if project_directory is not None else None
    )
    detections_controller = DetectionsController(
        model=detections_model,
        view=detections_view,
        store=detections_store,
        active_learning_model=active_learning_model,
        classifier_store=classifier_store,
        parent=detections_view,
        grouping_preference=session.get("grouping_enabled"),
        grouping_gap=session.get("grouping_gap_s"),
    )
    active_learning_controller.workspace_state_changed.connect(
        detections_controller.refresh_from_active_learning
    )

    def refresh_detection_recordings(*_args):
        space, _ref = explorer_controller.current_embedding_selection()
        detections_controller.set_explorer_context(
            space, explorer_model.audio_packets, explorer_model.chunk_grid
        )

    explorer_controller.pipeline_activation_succeeded.connect(
        refresh_detection_recordings
    )
    refresh_detection_recordings()

    window = MainWindow(
        explorer_view=explorer_view,
        pipeline_view=pipeline_view,
        active_learning_view=active_learning_view,
        detections_view=detections_view,
        terminal_view=terminal_view,
    )

    # Quick Review is a separate, non-modal pop-out. It reads the canonical
    # audio from the compatible Explorer dataset without changing its selection.
    quick_review_view = QuickReviewView(parent=window)
    quick_review_controller = QuickReviewController(
        view=quick_review_view,
        active_learning_controller=active_learning_controller,
        explorer_controller=explorer_controller,
        parent=quick_review_view,
    )
    active_learning_view.quick_review_requested.connect(quick_review_controller.show)

    def open_active_learning_example(ref):
        if explorer_controller.navigate_to_embedding(ref):
            window.focus_view(explorer_view)

    active_learning_controller.open_example_requested.connect(
        open_active_learning_example
    )

    def open_detection(space, ref):
        current_space, _ref = explorer_controller.current_embedding_selection()
        if current_space != space:
            detections_view.show_status(
                "Explorer is using an incompatible pipeline. Activate the "
                "matching pipeline before opening this detection."
            )
            return
        if explorer_controller.navigate_to_embedding(ref):
            # Deliberately leave keyboard focus in Detections for Next/Previous.
            window.explorer_dock.show()
        else:
            detections_view.show_status(
                "Explorer could not open this recording/chunk. Check the active dataset."
            )

    detections_controller.open_detection_requested.connect(open_detection)

    def label_detection(classifier_id, ref, embedding_space, label):
        _success, message = active_learning_controller.apply_detection_label(
            classifier_id, ref, embedding_space, label,
        )
        # Active Learning emits workspace_state_changed after a successful
        # save, so the Detections human-label column is refreshed immediately.
        detections_view.show_status(message)

    detections_controller.label_detection_requested.connect(label_detection)

    def label_detections(classifier_id, refs, embedding_space, label):
        _success, message = active_learning_controller.apply_detection_labels(
            classifier_id, refs, embedding_space, label,
        )
        detections_view.show_status(message)

    detections_controller.label_detections_requested.connect(label_detections)

    # Project snapshots combine editable JSON state with binary model weights
    # and a consistent SQLite backup. Archive creation runs off the UI thread.
    project_jobs = LatestJobRunner(name="caracal-project-save", parent=window)
    project_save_number = 0
    save_progress = None
    close_after_save = False
    close_allowed = False
    restarting = False
    open_after_save = None

    def collect_project_session():
        _space, ref = explorer_controller.current_embedding_selection()
        active = active_learning_model
        detections = detections_model
        return {
            "pipeline_selected": pipeline_run_model.selected_pipeline,
            "explorer_pipeline": (
                explorer_model.pipeline_name if explorer_model.pipeline is not None else None
            ),
            "explorer_ref": (
                {"recording_id": ref.recording_id, "chunk_index": ref.chunk_index}
                if ref is not None else None
            ),
            "classifier_selected": active.selected_classifier_id,
            "classifier_pipeline": active.selected_pipeline_name,
            "classifier_prepared": active.is_ready,
            "review_queue": [
                {"recording_id": item.ref.recording_id,
                 "chunk_index": item.ref.chunk_index,
                 "probability": item.probability,
                 "reason": item.reason.value}
                for item in active.review_queue
            ],
            "review_index": active.review_index,
            "review_skipped_refs": [
                {"recording_id": item.recording_id, "chunk_index": item.chunk_index}
                for item in sorted(active.review_skipped_refs)
            ],
            "detections": {
                "classifier_id": detections.classifier_id,
                "run_id": detections.run.run_id if detections.run is not None else None,
                "min_score": detections.min_score,
                "recording_id": detections.recording_id,
                "sort_by": detections.sort_by,
                "descending": detections.descending,
                "page": detections.page,
                "selected_refs": [
                    {"recording_id": item.recording_id, "chunk_index": item.chunk_index}
                    for item in sorted(detections.selected_refs)
                ],
            },
            "grouping_enabled": detections_controller._grouping_preferred,
            "grouping_gap_s": detections_model.grouping_gap_s,
            "window_geometry": base64.b64encode(bytes(window.saveGeometry())).decode("ascii"),
            "window_layout": base64.b64encode(bytes(window.saveState())).decode("ascii"),
        }

    def is_project_busy():
        return (
            pipeline_controller._run_is_active()
            or pipeline_run_model.activation_pending
            or active_learning_model.status is ActiveLearningStatus.PREPARING
            or active_learning_model.round_running
            or detections_model.scanning
            or detections_controller._exporting
        )

    def save_project(*, save_as=False):
        nonlocal project_path, project_save_number, save_progress
        if save_progress is not None:
            return
        if is_project_busy():
            QMessageBox.warning(window, "Project busy",
                "Finish the current preparation, scan, training, export or benchmark before saving.")
            return
        destination = project_path
        if save_as or destination is None:
            proposed = str(destination or (Path.home() / "Caracal Project.caracal"))
            filename, _ = QFileDialog.getSaveFileName(
                window, "Save Project", proposed, "Caracal project (*.caracal)")
            if not filename:
                return
            destination = Path(filename)
            if not str(destination).lower().endswith(ProjectFile.EXTENSION):
                destination = Path(str(destination) + ProjectFile.EXTENSION)
        try:
            # Snapshot all mutable UI state on the Qt thread. The worker must
            # never read a live Qt widget or controller.
            snapshot = collect_project_session()
            pipelines = tuple(
                definition.copy()
                for name, definition in pipeline_controller.pipeline_definitions.items()
                if name not in pipeline_controller._builtin_pipeline_names
            )
            classifiers = tuple(active_learning_model.classifier_definitions)
        except Exception as exc:
            QMessageBox.critical(window, "Save failed", str(exc))
            return
        project_save_number += 1
        number = project_save_number
        save_progress = QProgressDialog("Saving project…", "", 0, 0, window)
        save_progress.setCancelButton(None)
        save_progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        save_progress.setMinimumDuration(0)
        save_progress.show()
        window.save_project_action.setEnabled(False)
        window.save_as_project_action.setEnabled(False)
        window.open_project_action.setEnabled(False)
        project_jobs.submit(number, partial(
            ProjectFile.save, destination, pipelines=pipelines,
            classifiers=classifiers, classifier_store=classifier_store,
            detection_store=detections_store, session=snapshot,
        ))

    def finish_project_save():
        nonlocal save_progress
        if save_progress is not None:
            save_progress.close()
            save_progress.deleteLater()
            save_progress = None
        window.save_project_action.setEnabled(True)
        window.save_as_project_action.setEnabled(True)
        window.open_project_action.setEnabled(True)

    def project_saved(number, path):
        nonlocal project_path, close_after_save, close_allowed, open_after_save
        if number != project_save_number:
            return
        finish_project_save()
        project_path = Path(path)
        window.setWindowTitle(f"{project_path.stem} — Caracal")
        window.statusBar().showMessage(f"Saved {project_path.name}", 5000)
        if open_after_save is not None:
            next_path = open_after_save
            open_after_save = None
            restart_with_project(next_path)
        if close_after_save:
            close_after_save = False
            close_allowed = True
            window.close()

    def project_failed(number, message):
        nonlocal close_after_save, open_after_save
        if number != project_save_number:
            return
        close_after_save = False
        open_after_save = None
        finish_project_save()
        QMessageBox.critical(window, "Save failed", message)

    project_jobs.ready.connect(project_saved)
    project_jobs.failed.connect(project_failed)
    window.save_project_action.triggered.connect(lambda: save_project())
    window.save_as_project_action.triggered.connect(lambda: save_project(save_as=True))

    def restart_with_project(filename):
        nonlocal restarting
        # Re-launch with the archive in an isolated session. This avoids
        # swapping SQLite/worker references underneath live MVC controllers.
        if getattr(sys, "frozen", False):
            program = sys.executable
            arguments = ["--project", str(Path(filename).resolve())]
            working_dir = str(Path(sys.executable).resolve().parent)
        else:
            program = sys.executable
            arguments = ["-m", "src.main", "--project", str(Path(filename).resolve())]
            working_dir = str(Path(__file__).resolve().parent.parent)
        if not QProcess.startDetached(program, arguments, working_dir):
            QMessageBox.critical(window, "Could not open project", "Could not restart Caracal.")
            return
        restarting = True
        app.quit()

    def open_project():
        nonlocal open_after_save
        if save_progress is not None or is_project_busy():
            QMessageBox.warning(window, "Project busy", "Finish the current operation first.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            window, "Open Project", str(project_path or Path.home()),
            "Caracal project (*.caracal)")
        if not filename:
            return
        try:
            ProjectFile.inspect(filename)
        except Exception as exc:
            QMessageBox.critical(window, "Could not open project", str(exc))
            return
        if project_path is not None:
            dialog = QMessageBox(window)
            dialog.setWindowTitle("Open Project")
            dialog.setText("Save the current project before opening another one?")
            dialog.setStandardButtons(
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            dialog.setDefaultButton(QMessageBox.StandardButton.Save)
            choice = dialog.exec()
            if choice == QMessageBox.StandardButton.Cancel:
                return
            if choice == QMessageBox.StandardButton.Save:
                open_after_save = filename
                save_project()
                if save_progress is None:
                    open_after_save = None
                return
        else:
            choice = QMessageBox.question(
                window, "Open Project",
                "Opening a project starts a new Caracal window and closes this one. "
                "Unsaved changes in this workspace will be lost. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
        restart_with_project(filename)

    window.open_project_action.triggered.connect(open_project)

    def before_close_project():
        """An open archive is not an auto-save directory; prevent lost edits."""
        nonlocal close_after_save
        if restarting or close_allowed:
            return True
        if save_progress is not None:
            return False
        if project_path is None:
            # Without an explicitly named project the default workspace uses
            # Caracal's existing per-user auto-saving stores.
            return True
        if is_project_busy():
            QMessageBox.warning(window, "Project busy", "Finish the current operation before closing the project.")
            return False
        dialog = QMessageBox(window)
        dialog.setWindowTitle("Close Project")
        dialog.setText("Save this project before closing Caracal?")
        dialog.setInformativeText("Changes not saved to the .caracal file will be lost.")
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Save)
        chosen = dialog.exec()
        if chosen == QMessageBox.StandardButton.Discard:
            return True
        if chosen == QMessageBox.StandardButton.Save:
            close_after_save = True
            save_project()
            if save_progress is None:  # cancelled the Save As dialog
                close_after_save = False
        return False

    window.project_before_close = before_close_project

    if project_payload is not None:
        window.setWindowTitle(f"{project_path.stem} — Caracal")

        def restore_project_session():
            # GUI settings and selected objects are restored by IDs, never by
            # display indices that might have changed between versions.
            chosen_pipeline = session.get("pipeline_selected")
            if chosen_pipeline in pipeline_controller.pipeline_definitions:
                pipeline_controller.select_pipeline(chosen_pipeline)
            chosen_classifier = session.get("classifier_selected")
            if any(item.classifier_id == chosen_classifier for item in active_learning_model.classifier_definitions):
                active_learning_controller.select_classifier(chosen_classifier)
                chosen_training_pipeline = session.get("classifier_pipeline")
                if not active_learning_model.pipeline_locked and chosen_training_pipeline in active_learning_model.available_pipelines:
                    active_learning_controller.select_pipeline(chosen_training_pipeline)

            detection_state = session.get("detections", {})
            if isinstance(detection_state, dict):
                cls_id = detection_state.get("classifier_id")
                if any(item.classifier_id == cls_id for item in active_learning_model.classifier_definitions):
                    detections_controller.select_classifier(cls_id)
                    run_id = detection_state.get("run_id")
                    if run_id and detections_store.get_run(run_id) is not None:
                        detections_controller.select_run(run_id)
                    if detections_model.run is not None:
                        threshold = detection_state.get("min_score")
                        if threshold is not None:
                            recording = detection_state.get("recording_id") or ""
                            detections_view.min_score.setValue(float(threshold))
                            detections_view.recording_filter.setText(recording)
                            detections_controller.apply_filters(float(threshold), recording)
                        sort_by = detection_state.get("sort_by")
                        if sort_by in detections_view.table_model.FIELDS:
                            detections_controller.set_sort(sort_by, bool(detection_state.get("descending", True)))
                        detections_model.selected_refs = {
                            EmbeddingRef(item["recording_id"], item["chunk_index"])
                            for item in detection_state.get("selected_refs", [])
                        }
                        detections_controller._show_selection_count()
                        page = max(0, int(detection_state.get("page", 0)))
                        if page:
                            detections_controller._load_page(page)

            for key, restore in (
                ("window_geometry", window.restoreGeometry),
                ("window_layout", window.restoreState),
            ):
                encoded = session.get(key)
                if isinstance(encoded, str) and len(encoded) < 200000:
                    try:
                        restore(QByteArray.fromBase64(encoded.encode("ascii")))
                    except (ValueError, TypeError):
                        pass

            # Only restore the review queue after its matching embedding space
            # has been prepared. Cached vectors and media stay external.
            review_state = session.get("review_queue", [])
            review_pending = bool(review_state) and bool(session.get("classifier_prepared"))

            def restore_review():
                nonlocal review_pending
                if not review_pending or not active_learning_model.is_ready:
                    return
                review_pending = False
                try:
                    queue = tuple(
                        ReviewCandidate(
                            ref=EmbeddingRef(item["recording_id"], item["chunk_index"]),
                            probability=item["probability"],
                            reason=ReviewReason(item["reason"]),
                        ) for item in review_state
                    )
                    active_learning_model.review_queue = queue
                    active_learning_model.review_index = min(
                        max(0, int(session.get("review_index", 0))), max(0, len(queue) - 1)
                    )
                    active_learning_model.review_skipped_refs = {
                        EmbeddingRef(item["recording_id"], item["chunk_index"])
                        for item in session.get("review_skipped_refs", [])
                    }
                    active_learning_controller.refresh_view()
                except (ValueError, KeyError, TypeError) as exc:
                    print(f"Could not restore review queue: {exc}", file=sys.stderr)

            active_learning_controller.workspace_state_changed.connect(restore_review)
            if session.get("classifier_prepared") and active_learning_model.selected_pipeline_name:
                active_learning_controller.prepare_selected_pipeline()

            explorer_pipeline = session.get("explorer_pipeline")
            if explorer_pipeline in pipeline_controller.pipeline_definitions:
                previous_selection = pipeline_run_model.selected_pipeline
                def restore_explorer(name, _hash):
                    if name != explorer_pipeline:
                        return
                    ref_data = session.get("explorer_ref")
                    if isinstance(ref_data, dict):
                        explorer_controller.navigate_to_embedding(EmbeddingRef(
                            ref_data["recording_id"], ref_data["chunk_index"]
                        ))
                    if previous_selection in pipeline_controller.pipeline_definitions:
                        pipeline_controller.select_pipeline(previous_selection)
                    explorer_controller.pipeline_activation_succeeded.disconnect(restore_explorer)
                explorer_controller.pipeline_activation_succeeded.connect(restore_explorer)
                pipeline_controller.select_pipeline(explorer_pipeline)
                pipeline_controller.use_in_explorer()

        QTimer.singleShot(0, restore_project_session)

    app.aboutToQuit.connect(project_jobs.shutdown)
    app.aboutToQuit.connect(quick_review_controller.shutdown)
    app.aboutToQuit.connect(
        explorer_controller.shutdown
    )
    app.aboutToQuit.connect(detections_controller.shutdown)
    app.aboutToQuit.connect(
        pipeline_controller.shutdown
    )
    app.aboutToQuit.connect(
        active_learning_controller.shutdown
    )
    app.aboutToQuit.connect(
        terminal_view.shutdown
    )

    # Keep the MVC objects alive for the duration of the Qt event loop.
    application_objects = (
        explorer_model,
        explorer_controller,
        pipeline_run_model,
        pipeline_runner,
        pipeline_controller,
        active_learning_model,
        active_learning_controller,
        quick_review_view,
        quick_review_controller,
        detections_model,
        detections_controller,
        detections_store,
        project_jobs,
        project_directory,
        terminal_view,
    )
    _ = application_objects

    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(
        main()
    )
