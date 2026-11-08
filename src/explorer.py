"""Interactive audio and embedding explorer entry point.

A single QMainWindow hosts Getting Started and successive project sessions.
Every project has isolated stores and workers, retired safely on Close/Open.
"""

import argparse
import base64
from functools import partial
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from PySide6.QtCore import QByteArray, QTimer, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

from src.ApplicationTheme import apply_application_theme
from src.MainWindow import MainWindow
from src.ProjectFile import ProjectFile
from src.ApplicationIdentity import APP_NAME
from src.QtSignalHandler import QtSignalHandler


def main():
    parser = argparse.ArgumentParser(add_help=False)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--project", default=None)
    choices.add_argument("--new-project", action="store_true")
    options, qt_args = parser.parse_known_args(sys.argv[1:])

    app = QApplication([sys.argv[0], *qt_args])
    apply_application_theme(app)
    signal_handler = QtSignalHandler(app)  # Create once, not for every project.
    window = MainWindow()
    current = None
    retired = []
    allow_close = False

    from src.LatestJobRunner import LatestJobRunner

    def poll_retired():
        """Keep old Qt objects and temporary files alive until workers exit."""
        for session in tuple(retired):
            if any(thread.is_alive() for thread in session.threads()):
                continue
            for dock in session.docks:
                dock.deleteLater()
            if session.quick_review_view is not None:
                session.quick_review_view.deleteLater()
            session.project_directory.cleanup()
            retired.remove(session)
        if not retired:
            retire_timer.stop()

    retire_timer = QTimer(window)
    retire_timer.setInterval(100)
    retire_timer.timeout.connect(poll_retired)

    def dispose_current():
        nonlocal current
        if current is None:
            return
        old = current
        current = None
        old.shutdown()  # Stop signal production and request worker cancellation.
        old.docks = window.clear_project_views()
        retired.append(old)
        retire_timer.start()
        poll_retired()

    def activate_project(project_path=None):
        """Install a new project in the existing window; never start another process."""
        nonlocal current
        if project_path == "welcome":
            dispose_current()
            return
        if project_path is not None:
            try:
                ProjectFile.inspect(project_path)
            except Exception as exc:
                QMessageBox.critical(window, "Could not open project", str(exc))
                return
        dispose_current()
        try:
            current = build_project(project_path)
        except Exception as exc:
            QMessageBox.critical(window, "Could not open project", str(exc))
            window.show_welcome()

    def request_transition(destination, title):
        """Ask about saving an open project before replacing its views."""
        if current is None:
            activate_project(destination)
            return
        if current.busy():
            QMessageBox.warning(window, "Project busy", "Finish the current operation first.")
            return
        dialog = QMessageBox(window)
        dialog.setWindowTitle(title)
        dialog.setText("Save the current project before continuing?")
        dialog.setInformativeText("Discard will lose changes not saved to a project file.")
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = dialog.exec()
        if choice == QMessageBox.StandardButton.Discard:
            QTimer.singleShot(0, lambda: activate_project(destination))
        elif choice == QMessageBox.StandardButton.Save:
            if current.save():
                current.after_save(lambda: activate_project(destination))

    def open_project():
        filename, _ = QFileDialog.getOpenFileName(
            window, "Open Project", str(Path.home()), ProjectFile.OPEN_FILTER)
        if not filename:
            return
        path = Path(filename).expanduser().resolve()
        if current is not None and current.path() == path:
            window.statusBar().showMessage("That project is already open.", 4000)
            return
        # Validate before prompting/switching, preserving an existing project
        # when the selected archive is corrupt.
        try:
            ProjectFile.inspect(path)
        except Exception as exc:
            QMessageBox.critical(window, "Could not open project", str(exc))
            return
        request_transition(path, "Open Project")

    def after_close():
        nonlocal allow_close
        dispose_current()
        allow_close = True
        window.close()

    def before_close():
        if allow_close or current is None:
            dispose_current()
            return True
        if current.busy():
            QMessageBox.warning(window, "Project busy", "Finish the current operation first.")
            return False
        dialog = QMessageBox(window)
        dialog.setWindowTitle(f"Close {APP_NAME}")
        dialog.setText("Save the current project before closing?")
        dialog.setInformativeText("Changes not saved to a project file will be lost.")
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = dialog.exec()
        if choice == QMessageBox.StandardButton.Discard:
            QTimer.singleShot(0, after_close)
        elif choice == QMessageBox.StandardButton.Save and current.save():
            current.after_save(after_close)
        return False

    window.project_before_close = before_close
    window.new_project_action.triggered.connect(
        lambda: request_transition(None, "New Project"))
    window.close_project_action.triggered.connect(
        lambda: request_transition("welcome", "Close Project"))
    window.open_project_action.triggered.connect(open_project)
    window.welcome_page.new_project_requested.connect(
        lambda: activate_project())
    window.welcome_page.open_project_requested.connect(open_project)
    window.save_project_action.triggered.connect(lambda: current.save() if current else None)
    window.save_as_project_action.triggered.connect(
        lambda: current.save(save_as=True) if current else None)

    def build_project(project_path):
        """Own all project-local stores, workers and Qt views in one session."""
        from src.ActiveLearner import ReviewCandidate, ReviewReason
        from src.EmbeddingCorpus import EmbeddingRef
        from src.ActiveLearningModel import ActiveLearningStatus
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
        from src.PipelineController import PipelineController
        from src.PipelineDefinitions import get_pipeline_definitions
        from src.PipelineDefinitionStore import PipelineDefinitionStore
        from src.PipelineFactory import PipelineFactory
        from src.PipelineRunner import PipelineRunner
        from src.PipelineRunModel import PipelineRunModel
        from src.PipelineView import PipelineView
        from src.TerminalView import TerminalView


        # Both an opened archive and an untitled NEW project use an isolated
        # temporary directory. An untitled project therefore starts completely
        # empty and cannot accidentally read or overwrite globally saved work.
        project_directory = tempfile.TemporaryDirectory(prefix="audio-project-")
        try:
            project_path = Path(project_path).expanduser().resolve() if project_path is not None else None
            project_payload = None
            if project_path is not None:
                try:
                    project_payload = ProjectFile.open_into(project_path, project_directory.name)
                except Exception as exc:
                    raise ValueError(f"{project_path}: {exc}") from exc

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
                    raise ValueError(f"Project incompatible: {exc}") from exc
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
                    raise ValueError(f"Project incompatible: {exc}") from exc
                print(
                    f"Could not load saved active-learning classifiers from "
                    f"{classifier_store.path}: {exc}",
                    file=sys.stderr,
                )
                classifier_definitions = ()

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
                grouping_preference=session.get("grouping_enabled", True),
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

            window.attach_project_views(
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
            project_jobs = LatestJobRunner(name="audio-project-save", parent=window)
            project_save_number = 0
            save_progress = None
            pending_after_save = None
            session_alive = True

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
                if not session_alive or save_progress is not None:
                    return False
                if is_project_busy():
                    QMessageBox.warning(window, "Project busy",
                        "Finish the current preparation, scan, training, export or benchmark before saving.")
                    return False
                destination = project_path
                # Never overwrite a legacy archive with a new-format manifest
                # and an old extension. Offer a new-name save instead.
                if destination is not None and destination.suffix.lower() != ProjectFile.EXTENSION:
                    save_as = True
                if save_as or destination is None:
                    suggestion = (destination.with_suffix(ProjectFile.EXTENSION)
                                  if destination is not None else
                                  Path.home() / f"Untitled Project{ProjectFile.EXTENSION}")
                    filename, _ = QFileDialog.getSaveFileName(
                        window, "Save Project", str(suggestion), ProjectFile.SAVE_FILTER)
                    if not filename:
                        return False
                    destination = Path(filename)
                    if destination.suffix.lower() != ProjectFile.EXTENSION:
                        if destination.suffix.lower() == ProjectFile.LEGACY_EXTENSION:
                            destination = destination.with_suffix(ProjectFile.EXTENSION)
                        else:
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
                    return False
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
                window.new_project_action.setEnabled(False)
                window.close_project_action.setEnabled(False)
                project_jobs.submit(number, partial(
                    ProjectFile.save, destination, pipelines=pipelines,
                    classifiers=classifiers, classifier_store=classifier_store,
                    detection_store=detections_store, session=snapshot,
                ))
                return True

            def finish_project_save():
                nonlocal save_progress
                if save_progress is not None:
                    save_progress.close()
                    save_progress.deleteLater()
                    save_progress = None
                window.save_project_action.setEnabled(True)
                window.save_as_project_action.setEnabled(True)
                window.open_project_action.setEnabled(True)
                window.new_project_action.setEnabled(True)
                window.close_project_action.setEnabled(True)

            def project_saved(number, path):
                nonlocal project_path, pending_after_save
                if number != project_save_number:
                    return
                finish_project_save()
                project_path = Path(path)
                window.setWindowTitle(f"{project_path.stem} — {APP_NAME}")
                window.statusBar().showMessage(f"Saved {project_path.name}", 5000)
                if pending_after_save is not None:
                    callback = pending_after_save
                    pending_after_save = None
                    QTimer.singleShot(0, callback)

            def project_failed(number, message):
                nonlocal pending_after_save
                if number != project_save_number:
                    return
                pending_after_save = None
                finish_project_save()
                QMessageBox.critical(window, "Save failed", message)

            project_jobs.ready.connect(project_saved)
            project_jobs.failed.connect(project_failed)

            if project_payload is None:
                window.setWindowTitle(f"Untitled Project — {APP_NAME}")
            if project_payload is not None:
                window.setWindowTitle(f"{project_path.stem} — {APP_NAME}")

                def restore_project_session():
                    if not session_alive:
                        return
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
                        if not session_alive or not review_pending or not active_learning_model.is_ready:
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


            def set_after_save(callback):
                nonlocal pending_after_save
                pending_after_save = callback

            def shutdown_session():
                nonlocal session_alive
                if not session_alive:
                    return
                session_alive = False
                project_jobs.shutdown()
                quick_review_controller.shutdown()
                explorer_controller.shutdown()
                explorer_view.shutdown()  # Stop playback when closing a project, not only on app exit.
                detections_controller.shutdown()
                pipeline_controller.shutdown()
                active_learning_controller.shutdown()
                terminal_view.shutdown()
                quick_review_view.hide()

            # Hold the complete session alive while asynchronous workers finish.
            # The window keeps no project state after clear_project_views().
            controllers = (explorer_controller, pipeline_controller,
                           active_learning_controller, detections_controller,
                           quick_review_controller)
            workers = [project_jobs]
            for controller in controllers:
                workers.extend(value for value in vars(controller).values()
                               if isinstance(value, LatestJobRunner))
            def running_threads():
                threads = [runner._thread for runner in workers]
                if pipeline_runner._thread is not None:
                    threads.append(pipeline_runner._thread)
                return threads

            return SimpleNamespace(
                save=save_project, busy=lambda: save_progress is not None or is_project_busy(),
                path=lambda: project_path, after_save=set_after_save,
                shutdown=shutdown_session, project_directory=project_directory,
                quick_review_view=quick_review_view, threads=running_threads,
                controllers=controllers, views=(explorer_view, pipeline_view,
                    active_learning_view, detections_view, terminal_view),
                other_objects=(project_jobs, detections_store, pipeline_runner),
                docks=(),
            )

        except Exception:
            # A malformed project or a constructor failure must not leave a
            # half-built workspace attached to the main window.
            state = dict(locals())
            controllers = [state[name] for name in (
                "quick_review_controller", "explorer_controller",
                "detections_controller", "pipeline_controller",
                "active_learning_controller") if name in state]
            job_objects = [state[name] for name in ("project_jobs",) if name in state]
            for controller in controllers:
                job_objects.extend(value for value in vars(controller).values()
                                   if isinstance(value, LatestJobRunner))
            for controller in controllers:
                try:
                    controller.shutdown()
                except Exception:
                    pass
            for job in job_objects:
                job.shutdown()
            runner = state.get("pipeline_runner")
            if runner is not None:
                runner.shutdown()
            explorer = state.get("explorer_view")
            if explorer is not None:
                explorer.shutdown()
            terminal = state.get("terminal_view")
            if terminal is not None:
                terminal.shutdown()
            quick = state.get("quick_review_view")
            if quick is not None:
                quick.hide()
            docks = window.clear_project_views() if window._workspace_docks else ()

            def failed_threads():
                threads = [job._thread for job in job_objects]
                if runner is not None and runner._thread is not None:
                    threads.append(runner._thread)
                return threads

            # Retire any in-flight IO rather than deleting its SQLite files.
            failed = SimpleNamespace(
                threads=failed_threads, docks=docks, quick_review_view=quick,
                project_directory=project_directory, refs=state,
            )
            retired.append(failed)
            retire_timer.start()
            poll_retired()
            raise

    # Close Project returns to this same window's Getting Started page.
    # "welcome" is only an internal route, never a filesystem path.
    window.show()
    if options.project is not None:
        QTimer.singleShot(0, lambda: activate_project(Path(options.project).expanduser().resolve()))
    elif options.new_project:
        QTimer.singleShot(0, lambda: activate_project())
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
