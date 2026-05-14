import os
import re
import json
import shutil
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, QUrl, Signal, Qt
from PySide6.QtGui import QDesktopServices, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from cryptography.fernet import Fernet, InvalidToken

import ocr_pipeline
from ocr_pipeline import PipelineConfig
from model_options import get_model_options

CONFIG_PATH = Path(__file__).with_name("config.json")
KEY_PATH = Path(__file__).with_name("config.key")


def load_config():
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def save_config(data):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_or_create_key() -> bytes:
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()
    key = Fernet.generate_key()
    KEY_PATH.write_bytes(key)
    return key


def encrypt_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    fernet = Fernet(load_or_create_key())
    return fernet.encrypt(api_key.encode("utf-8")).decode("ascii")


def decrypt_api_key(config: dict) -> str:
    encrypted = config.get("api_key_encrypted", "")
    if encrypted:
        try:
            fernet = Fernet(load_or_create_key())
            return fernet.decrypt(encrypted.encode("ascii")).decode("utf-8")
        except (InvalidToken, OSError, ValueError):
            return ""
    return os.environ.get("GOOGLE_VISION_API_KEY", "")


def encrypt_text(value: str) -> str:
    if not value:
        return ""
    fernet = Fernet(load_or_create_key())
    return fernet.encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_text(value: str) -> str:
    if not value:
        return ""
    try:
        fernet = Fernet(load_or_create_key())
        return fernet.decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, OSError, ValueError):
        return ""


class DropList(QListWidget):
    paths_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                paths.append(local)
        if paths:
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class SignalStream:
    def __init__(self, signal):
        self.signal = signal
        self._buffer = ""

    def write(self, text):
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.signal.emit(line)

    def flush(self):
        if self._buffer:
            self.signal.emit(self._buffer)
            self._buffer = ""


class OcrWorker(QThread):
    log = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, image_paths, output_dir, api_key, credentials_json, model, base_url,
                 batch_size, api_delay, temperature, max_output_tokens,
                 streaming, send_thinking, gemini_thinking_level, openai_thinking_effort,
                 smart_chunking, chunks_only, flatten_white,
                 skip_pinyin, skip_romanization, enable_dedupe,
                 system_prompt, user_prompt, job_name, use_existing_chunks, no_resume, create_epub,
                 cleanup_rules):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.api_key = api_key
        self.credentials_json = credentials_json
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.api_delay = api_delay
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.streaming = streaming
        self.send_thinking = send_thinking
        self.gemini_thinking_level = gemini_thinking_level
        self.openai_thinking_effort = openai_thinking_effort
        self.smart_chunking = smart_chunking
        self.chunks_only = chunks_only
        self.flatten_white = flatten_white
        self.skip_pinyin = skip_pinyin
        self.skip_romanization = skip_romanization
        self.enable_dedupe = enable_dedupe
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.job_name = job_name
        self.use_existing_chunks = use_existing_chunks
        self.no_resume = no_resume
        self.create_epub = create_epub
        self.cleanup_rules = cleanup_rules

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_config = ocr_pipeline._config

        # Build a fresh config from GUI values
        ocr_pipeline._config = PipelineConfig(
            smart_line_chunking=self.smart_chunking,
            flatten_to_white_background=self.flatten_white,
            skip_pinyin_lines=self.skip_pinyin,
            skip_romanization_lines=self.skip_romanization,
            enable_dedupe=self.enable_dedupe,
            max_workers=self.batch_size,
            api_call_delay_seconds=self.api_delay,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            streaming=self.streaming,
            send_thinking_parameters=self.send_thinking,
            gemini_thinking_level=self.gemini_thinking_level,
            openai_thinking_effort=self.openai_thinking_effort,
            use_existing_chunks=self.use_existing_chunks,
            resume_processing=not self.no_resume,
            export_epub=self.create_epub,
            cleanup_rules=list(self.cleanup_rules),
            vision_ocr_prompt=self.system_prompt or ocr_pipeline.DEFAULT_VISION_OCR_PROMPT,
            vision_ocr_user_prompt=self.user_prompt or ocr_pipeline.DEFAULT_VISION_OCR_USER_PROMPT,
        )

        sys.stdout = SignalStream(self.log)
        sys.stderr = SignalStream(self.log)
        try:
            for index, image_path in enumerate(self.image_paths, start=1):
                if self.isInterruptionRequested():
                    self.log.emit("Stopped before next image.")
                    break
                out_dir = self.output_dir or ocr_pipeline._config.default_output_dir
                self.log.emit(f"[{index}/{len(self.image_paths)}] {image_path}")
                if self.chunks_only:
                    ocr_pipeline.process_image_chunk_only(image_path, out_dir, self.job_name)
                else:
                    ocr_pipeline.process_image(
                        image_path,
                        out_dir,
                        self.api_key,
                        self.credentials_json,
                        self.model,
                        self.base_url,
                        job_name=self.job_name,
                    )

            if self.create_epub and self.job_name and not self.chunks_only:
                results_dir = ocr_pipeline._results_dir(out_dir, self.job_name)
                if os.path.isdir(results_dir):
                    html_files = [f for f in os.listdir(results_dir) if f.endswith(".html")]
                    if html_files:
                        epub_path = ocr_pipeline._export_epub(results_dir, self.job_name)
                        self.log.emit(f"Exported EPUB -> {epub_path} ({len(html_files)} chapters)")

            self.finished_ok.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            ocr_pipeline._config = old_config


class CustomModelsDialog(QDialog):
    """Dialog for managing user-added custom model names."""

    def __init__(self, models: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Custom Models")
        self.resize(400, 300)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Custom model names (appear in the model dropdown):"))

        self.list_widget = QListWidget()
        for m in models:
            self.list_widget.addItem(m)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_model)
        btn_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_model(self):
        text, ok = QInputDialog.getText(self, "Add Custom Model", "Model name:")
        if ok and text.strip():
            self.list_widget.addItem(text.strip())

    def remove_selected(self):
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def get_models(self) -> list[str]:
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]


class ResultsBrowserDialog(QDialog):
    """Browse OCR Results, preview output, and search across files."""

    def __init__(self, output_dir: str, parent=None):
        super().__init__(parent)
        self.output_dir = output_dir
        self.setWindowTitle("OCR Results Browser")
        self.resize(850, 550)

        self._results_cache: dict[str, str] = {}  # file_path -> content

        layout = QVBoxLayout(self)
        split = QHBoxLayout()

        # Left: tree
        left = QVBoxLayout()
        left.addWidget(QLabel("Jobs / Files:"))
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item_clicked)
        left.addWidget(self.tree)
        split.addLayout(left, stretch=1)

        # Right: preview
        right = QVBoxLayout()
        right.addWidget(QLabel("Preview:"))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        right.addWidget(self.preview)
        split.addLayout(right, stretch=3)

        layout.addLayout(split)

        # Bottom bar: search + actions
        bottom = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search in preview...")
        self.search_edit.returnPressed.connect(self._find_in_preview)
        bottom.addWidget(self.search_edit, stretch=1)

        find_btn = QPushButton("Find")
        find_btn.clicked.connect(self._find_in_preview)
        bottom.addWidget(find_btn)

        scan_btn = QPushButton("Scan All")
        scan_btn.clicked.connect(self._scan_all)
        bottom.addWidget(scan_btn)

        open_btn = QPushButton("Open File")
        open_btn.clicked.connect(self._open_current)
        bottom.addWidget(open_btn)

        open_dir_btn = QPushButton("Open Folder")
        open_dir_btn.clicked.connect(self._open_folder)
        bottom.addWidget(open_dir_btn)

        bottom.addStretch()
        ok_btn = QPushButton("Close")
        ok_btn.clicked.connect(self.accept)
        bottom.addWidget(ok_btn)
        layout.addLayout(bottom)

        self._populate_tree()

    def _populate_tree(self):
        self.tree.clear()
        self._results_cache.clear()

        results_root = os.path.join(self.output_dir, "OCR Results")
        if not os.path.isdir(results_root):
            self.tree.addTopLevelItem(QTreeWidgetItem(["(No results found)"]))
            return

        for job_name in sorted(os.listdir(results_root)):
            job_dir = os.path.join(results_root, job_name)
            if not os.path.isdir(job_dir):
                continue
            html_files = sorted(f for f in os.listdir(job_dir) if f.endswith(".html"))
            if not html_files:
                continue

            job_item = QTreeWidgetItem([job_name])
            job_item.setData(0, Qt.UserRole, job_dir)
            for hf in html_files:
                file_path = os.path.join(job_dir, hf)
                child = QTreeWidgetItem([os.path.splitext(hf)[0]])
                child.setData(0, Qt.UserRole, file_path)
                job_item.addChild(child)
            self.tree.addTopLevelItem(job_item)

    def _load_file_content(self, file_path: str) -> str:
        if file_path not in self._results_cache:
            try:
                with open(file_path, encoding="utf-8") as f:
                    self._results_cache[file_path] = f.read()
            except OSError:
                self._results_cache[file_path] = "(Could not read file)"
        return self._results_cache[file_path]

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int):
        file_path = item.data(0, Qt.UserRole)
        if not file_path or not file_path.endswith(".html"):
            return
        content = self._load_file_content(file_path)
        self.preview.setHtml(content)

    def _find_in_preview(self):
        term = self.search_edit.text().strip()
        if not term:
            return
        cursor = self.preview.textCursor()
        cursor.movePosition(cursor.Start)
        self.preview.setTextCursor(cursor)
        if not self.preview.find(term):
            # wrap around
            cursor.movePosition(cursor.Start)
            self.preview.setTextCursor(cursor)
            self.preview.find(term)

    def _scan_all(self):
        term = self.search_edit.text().strip()
        if not term:
            return

        results_root = os.path.join(self.output_dir, "OCR Results")
        matches = []
        for job_name in sorted(os.listdir(results_root)) if os.path.isdir(results_root) else []:
            job_dir = os.path.join(results_root, job_name)
            if not os.path.isdir(job_dir):
                continue
            for f in sorted(os.listdir(job_dir)):
                if not f.endswith(".html"):
                    continue
                path = os.path.join(job_dir, f)
                content = self._load_file_content(path)
                # Strip HTML tags for text search
                text = re.sub(r"<[^>]+>", "", content)
                if term.lower() in text.lower():
                    matches.append(f"{job_name}/{f}")

        if matches:
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Matches for \"{term}\"")
            dlg.resize(500, 300)
            layout = QVBoxLayout(dlg)
            label = QLabel(f"{len(matches)} file(s) contain \"{term}\":")
            layout.addWidget(label)
            text = QTextEdit()
            text.setReadOnly(True)
            text.setPlainText("\n".join(matches))
            layout.addWidget(text)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn)
            dlg.exec()
        else:
            QMessageBox.information(self, "Scan Results", f"No files contain \"{term}\".")

    def _open_current(self):
        item = self.tree.currentItem()
        if not item:
            return
        file_path = item.data(0, Qt.UserRole)
        if file_path and os.path.isfile(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

    def _open_folder(self):
        item = self.tree.currentItem()
        if not item:
            return
        folder = item.data(0, Qt.UserRole)
        if not folder:
            return
        if os.path.isfile(folder):
            folder = os.path.dirname(folder)
        if os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))


class ChunkPreviewDialog(QDialog):
    """Preview chunk boundaries overlaid on the source image."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Chunk Preview — {os.path.basename(image_path)}")
        self.resize(650, 700)

        self.image_path = image_path
        self.chunk_height = ocr_pipeline._config.target_chunk_height
        self._full_img = None

        layout = QVBoxLayout(self)

        # Slider for chunk height
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Chunk height:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(200, 4000)
        self.slider.setValue(self.chunk_height)
        self.slider.setTickInterval(200)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self._update_preview)
        slider_row.addWidget(self.slider, stretch=1)
        self.height_label = QLabel(f"{self.chunk_height} px")
        self.height_label.setFixedWidth(60)
        slider_row.addWidget(self.height_label)
        layout.addLayout(slider_row)

        # Image display
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll, stretch=1)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self._update_preview(self.chunk_height)

    def _build_overlay(self, target_height: int) -> QPixmap:
        """Render the source image with red horizontal lines at chunk boundaries."""
        try:
            from PIL import Image, ImageDraw
            img = Image.open(self.image_path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
        except Exception:
            return QPixmap()

        w, h = img.size
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay)

        # Compute cut points using the same logic as the pipeline
        y = target_height
        while y < h:
            draw.line([(0, y), (w, y)], fill="red", width=2)
            y += target_height

        # Convert to QPixmap
        data = overlay.tobytes("raw", "RGB" if overlay.mode == "RGB" else "RGBA")
        fmt = QImage.Format_RGB888 if overlay.mode == "RGB" else QImage.Format_RGBA8888
        qimg = QImage(data, w, h, fmt)
        return QPixmap.fromImage(qimg)

    def _update_preview(self, value: int):
        self.chunk_height = value
        self.height_label.setText(f"{value} px")
        pixmap = self._build_overlay(value)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap)


class CleanupRulesDialog(QDialog):
    """Manage regex text cleanup rules and apply them retroactively."""

    COL_PATTERN = 0
    COL_REPLACEMENT = 1
    COL_ENABLED = 2

    def __init__(self, rules: list[dict], output_dir: str, job_name: str, parent=None):
        super().__init__(parent)
        self.rules = list(rules)  # copy
        self.output_dir = output_dir
        self.job_name = job_name
        self.setWindowTitle("Text Cleanup Rules")
        self.resize(700, 400)

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Pattern", "Replacement", "On", ""])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(self.COL_PATTERN, 200)
        self.table.setColumnWidth(self.COL_REPLACEMENT, 200)
        self.table.setColumnWidth(self.COL_ENABLED, 40)
        self.table.setColumnWidth(3, 80)
        self._rebuild_table()
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Rule")
        add_btn.clicked.connect(self._add_rule)
        btn_row.addWidget(add_btn)

        load_btn = QPushButton("Load...")
        load_btn.clicked.connect(self._load_rules)
        btn_row.addWidget(load_btn)

        save_btn = QPushButton("Save As...")
        save_btn.clicked.connect(self._save_rules)
        btn_row.addWidget(save_btn)

        apply_btn = QPushButton("Apply to Existing Results")
        apply_btn.clicked.connect(self._apply_to_results)
        btn_row.addWidget(apply_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _rebuild_table(self):
        self.table.setRowCount(0)
        for i, rule in enumerate(self.rules):
            self.table.insertRow(i)
            pat = QTableWidgetItem(rule.get("pattern", ""))
            rep = QTableWidgetItem(rule.get("replacement", ""))
            chk = QTableWidgetItem()
            chk.setFlags(chk.flags() | Qt.ItemIsUserCheckable)
            chk.setCheckState(Qt.Checked if rule.get("enabled", True) else Qt.Unchecked)
            self.table.setItem(i, self.COL_PATTERN, pat)
            self.table.setItem(i, self.COL_REPLACEMENT, rep)
            self.table.setItem(i, self.COL_ENABLED, chk)
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda checked, row=i: self._remove_rule(row))
            self.table.setCellWidget(i, 3, remove_btn)

    def _sync_table_to_rules(self):
        self.rules.clear()
        for i in range(self.table.rowCount()):
            pat = self.table.item(i, self.COL_PATTERN)
            rep = self.table.item(i, self.COL_REPLACEMENT)
            chk = self.table.item(i, self.COL_ENABLED)
            self.rules.append({
                "pattern": pat.text() if pat else "",
                "replacement": rep.text() if rep else "",
                "enabled": chk.checkState() == Qt.Checked if chk else True,
            })

    def _add_rule(self):
        self._sync_table_to_rules()
        self.rules.append({"pattern": "", "replacement": "", "enabled": True})
        self._rebuild_table()

    def _remove_rule(self, row):
        self._sync_table_to_rules()
        if 0 <= row < len(self.rules):
            self.rules.pop(row)
        self._rebuild_table()

    def _load_rules(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Cleanup Rules", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.rules = data
            elif isinstance(data, dict) and "replacements" in data:
                self.rules = [{"pattern": p, "replacement": r, "enabled": True}
                              for p, r in data["replacements"]]
            self._rebuild_table()
        except (json.JSONDecodeError, OSError) as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _save_rules(self):
        self._sync_table_to_rules()
        path, _ = QFileDialog.getSaveFileName(self, "Save Cleanup Rules", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"replacements": [[r["pattern"], r["replacement"]] for r in self.rules if r.get("enabled", True)]},
                          f, ensure_ascii=False, indent=2)
        except OSError as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _apply_to_results(self):
        self._sync_table_to_rules()
        if not self.rules:
            QMessageBox.information(self, "No Rules", "Add at least one rule first.")
            return

        results_dir = ocr_pipeline._results_dir(self.output_dir, self.job_name)
        if not os.path.isdir(results_dir):
            QMessageBox.warning(self, "No Results", f"No results directory: {results_dir}")
            return

        html_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".html")
                            and not f.endswith("_backup.html"))
        if not html_files:
            QMessageBox.information(self, "No Files", "No .html files to process.")
            return

        reply = QMessageBox.question(self, "Apply Cleanup",
                                     f"Apply {len(self.rules)} rule(s) to {len(html_files)} file(s)?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        for hf in html_files:
            path = os.path.join(results_dir, hf)
            backup = path.replace(".html", "_backup.html")
            shutil.copy2(path, backup)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            blocks = ocr_pipeline._parse_blocks_from_html(html)
            cleaned = ocr_pipeline._apply_cleanup(blocks, self.rules)
            new_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{os.path.splitext(hf)[0]}</title></head>
<body>{"".join(cleaned)}</body></html>"""
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_html)
        QMessageBox.information(self, "Done", f"Cleaned {len(html_files)} file(s). Backups saved as *_backup.html.")


class ModelCompareDialog(QDialog):
    """Run the same image through two models and compare results side-by-side."""

    def __init__(self, image_path: str, model_items: list[str], api_key: str, base_url: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.api_key = api_key
        self.base_url = base_url
        self.setWindowTitle(f"Model Comparison — {os.path.basename(image_path)}")
        self.resize(850, 550)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"Image: {os.path.basename(image_path)}"))

        # Model selectors
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model A:"))
        self.model_a = QComboBox()
        self.model_a.setEditable(True)
        self.model_a.addItems(model_items)
        self.model_a.setCurrentText(model_items[0] if model_items else "")
        model_row.addWidget(self.model_a, stretch=1)

        model_row.addWidget(QLabel("Model B:"))
        self.model_b = QComboBox()
        self.model_b.setEditable(True)
        self.model_b.addItems(model_items)
        default_b = model_items[1] if len(model_items) > 1 else model_items[0]
        self.model_b.setCurrentText(default_b)
        model_row.addWidget(self.model_b, stretch=1)
        layout.addLayout(model_row)

        run_btn = QPushButton("Run Comparison")
        run_btn.clicked.connect(self._run_comparison)
        layout.addWidget(run_btn)

        # Side-by-side results
        split = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Model A output:"))
        self.result_a = QTextEdit()
        self.result_a.setReadOnly(True)
        left.addWidget(self.result_a)
        split.addLayout(left)

        right = QVBoxLayout()
        right.addWidget(QLabel("Model B output:"))
        self.result_b = QTextEdit()
        self.result_b.setReadOnly(True)
        right.addWidget(self.result_b)
        split.addLayout(right)
        layout.addLayout(split)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _run_comparison(self):
        self.result_a.setPlainText("Running...")
        self.result_b.setPlainText("Running...")
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        model_a = self.model_a.currentText().strip()
        model_b = self.model_b.currentText().strip()

        def run_one(model_name: str) -> list[str]:
            try:
                return ocr_pipeline.ocr_image(
                    self.image_path, self.api_key, model=model_name, base_url=self.base_url
                )
            except Exception as e:
                return [f"<p>Error: {e}</p>"]

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(run_one, model_a)
            future_b = pool.submit(run_one, model_b)
            blocks_a = future_a.result()
            blocks_b = future_b.result()

        text_a = ocr_pipeline._html_blocks_to_plain_text(blocks_a)
        text_b = ocr_pipeline._html_blocks_to_plain_text(blocks_b)

        self.result_a.setPlainText(text_a)
        self.result_b.setPlainText(text_b)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.setWindowTitle("ocrrr")
        self.resize(
            int(self.config.get("window_width", 900)),
            int(self.config.get("window_height", 750)),
        )
        self.worker = None
        self._custom_models = self.config.get("custom_models", [])
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._schedule_save)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        title = QLabel("Drop image files or folders below")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        self.path_list = DropList()
        self.path_list.paths_dropped.connect(self.add_paths)
        layout.addWidget(self.path_list, stretch=2)

        buttons = QHBoxLayout()
        add_files = QPushButton("Add Files")
        add_files.clicked.connect(self.pick_files)
        buttons.addWidget(add_files)

        add_folder = QPushButton("Add Folder")
        add_folder.clicked.connect(self.pick_folder)
        buttons.addWidget(add_folder)

        remove_selected_btn = QPushButton("Remove Selected")
        remove_selected_btn.clicked.connect(self.remove_selected)
        buttons.addWidget(remove_selected_btn)

        clear = QPushButton("Clear")
        clear.clicked.connect(self.path_list.clear)
        buttons.addWidget(clear)
        buttons.addStretch()
        layout.addLayout(buttons)

        # ---- Model ----
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.currentTextChanged.connect(self._schedule_save)
        self.model_combo.currentTextChanged.connect(self.update_key_state)
        model_row.addWidget(self.model_combo, stretch=1)

        manage_custom = QPushButton("Manage Custom Models")
        manage_custom.clicked.connect(self.manage_custom_models)
        model_row.addWidget(manage_custom)
        layout.addLayout(model_row)

        # ---- Profiles ----
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(150)
        self.profile_combo.currentTextChanged.connect(self._load_profile)
        profile_row.addWidget(self.profile_combo)
        save_profile_btn = QPushButton("Save As...")
        save_profile_btn.clicked.connect(self._save_profile)
        profile_row.addWidget(save_profile_btn)
        delete_profile_btn = QPushButton("Delete")
        delete_profile_btn.clicked.connect(self._delete_profile)
        profile_row.addWidget(delete_profile_btn)
        profile_row.addStretch()
        layout.addLayout(profile_row)

        # ---- API key ----
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API key:"))
        self.api_key_edit = QLineEdit(decrypt_api_key(self.config))
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.textChanged.connect(self._schedule_save)
        key_row.addWidget(self.api_key_edit, stretch=1)
        layout.addLayout(key_row)

        # ---- URL route ----
        route_row = QHBoxLayout()
        route_row.addWidget(QLabel("URL route:"))
        self.base_url_edit = QLineEdit(str(self.config.get("base_url", "")))
        self.base_url_edit.setPlaceholderText("Blank = auto routing")
        self.base_url_edit.textChanged.connect(self._schedule_save)
        self.base_url_edit.textChanged.connect(self.update_key_state)
        route_row.addWidget(self.base_url_edit, stretch=1)
        layout.addLayout(route_row)

        # ---- Credentials ----
        credentials_row = QHBoxLayout()
        credentials_row.addWidget(QLabel("Service account JSON:"))
        saved_credentials = decrypt_text(str(self.config.get("credentials_json_encrypted", "")))
        self.credentials_edit = QLineEdit(saved_credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
        self.credentials_edit.textChanged.connect(self._schedule_save)
        credentials_row.addWidget(self.credentials_edit, stretch=1)
        browse_credentials = QPushButton("Browse")
        browse_credentials.clicked.connect(self.pick_credentials)
        credentials_row.addWidget(browse_credentials)
        layout.addLayout(credentials_row)

        # ---- Output folder ----
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output folder:"))
        self.output_edit = QLineEdit(str(self.config.get("output_dir", "")))
        self.output_edit.setPlaceholderText("Blank = gui.py folder")
        self.output_edit.textChanged.connect(self._schedule_save)
        output_row.addWidget(self.output_edit, stretch=1)
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self.pick_output)
        output_row.addWidget(browse_output)
        layout.addLayout(output_row)

        # ---- Job name ----
        job_row = QHBoxLayout()
        job_row.addWidget(QLabel("Job name:"))
        self.job_name_edit = QLineEdit(str(self.config.get("job_name", "")))
        self.job_name_edit.setPlaceholderText("Derived from image folder")
        self.job_name_edit.textChanged.connect(self._schedule_save)
        job_row.addWidget(self.job_name_edit, stretch=1)
        view_progress_btn = QPushButton("View Progress")
        view_progress_btn.clicked.connect(self.view_progress)
        job_row.addWidget(view_progress_btn)
        layout.addLayout(job_row)

        # ---- Tuning ----
        tuning_row = QHBoxLayout()
        tuning_row.addWidget(QLabel("Batch size:"))
        self.batch_size_edit = QLineEdit(str(self.config.get("batch_size", ocr_pipeline._config.max_workers)))
        self.batch_size_edit.setFixedWidth(60)
        self.batch_size_edit.textChanged.connect(self._schedule_save)
        tuning_row.addWidget(self.batch_size_edit)
        tuning_row.addWidget(QLabel("API delay:"))
        self.api_delay_edit = QLineEdit(str(self.config.get("api_delay", ocr_pipeline._config.api_call_delay_seconds)))
        self.api_delay_edit.setFixedWidth(60)
        self.api_delay_edit.textChanged.connect(self._schedule_save)
        tuning_row.addWidget(self.api_delay_edit)
        tuning_row.addWidget(QLabel("Temperature:"))
        self.temperature_edit = QLineEdit(str(self.config.get("temperature", ocr_pipeline._config.temperature)))
        self.temperature_edit.setFixedWidth(60)
        self.temperature_edit.textChanged.connect(self._schedule_save)
        tuning_row.addWidget(self.temperature_edit)
        tuning_row.addWidget(QLabel("Max tokens:"))
        self.max_output_tokens_edit = QLineEdit(str(self.config.get("max_output_tokens", ocr_pipeline._config.max_output_tokens)))
        self.max_output_tokens_edit.setFixedWidth(70)
        self.max_output_tokens_edit.textChanged.connect(self._schedule_save)
        tuning_row.addWidget(self.max_output_tokens_edit)
        tuning_row.addStretch()
        layout.addLayout(tuning_row)

        # ---- Thinking ----
        thinking_row = QHBoxLayout()
        self.send_thinking = QCheckBox("Send thinking params")
        self.send_thinking.setChecked(bool(self.config.get("send_thinking", False)))
        self.send_thinking.toggled.connect(self._schedule_save)
        self.send_thinking.toggled.connect(self.update_key_state)
        thinking_row.addWidget(self.send_thinking)
        thinking_row.addWidget(QLabel("Gemini:"))
        self.gemini_thinking_combo = QComboBox()
        self.gemini_thinking_combo.addItems(["minimal", "low", "medium", "high"])
        self.gemini_thinking_combo.setCurrentText(str(self.config.get("gemini_thinking_level", ocr_pipeline._config.gemini_thinking_level)))
        self.gemini_thinking_combo.currentTextChanged.connect(self._schedule_save)
        self.gemini_thinking_combo.setFixedWidth(100)
        thinking_row.addWidget(self.gemini_thinking_combo)
        thinking_row.addWidget(QLabel("OpenAI:"))
        self.openai_thinking_combo = QComboBox()
        self.openai_thinking_combo.addItems(["none", "low", "medium", "high", "xhigh"])
        self.openai_thinking_combo.setCurrentText(str(self.config.get("openai_thinking_effort", ocr_pipeline._config.openai_thinking_effort)))
        self.openai_thinking_combo.currentTextChanged.connect(self._schedule_save)
        self.openai_thinking_combo.setFixedWidth(100)
        thinking_row.addWidget(self.openai_thinking_combo)
        thinking_row.addStretch()
        layout.addLayout(thinking_row)

        # ---- Options ----
        options = QHBoxLayout()
        self.smart_chunking = QCheckBox("Smart line chunking")
        self.smart_chunking.setChecked(bool(self.config.get("smart_chunking", True)))
        self.smart_chunking.toggled.connect(self._schedule_save)
        options.addWidget(self.smart_chunking)
        self.flatten_white = QCheckBox("White background")
        self.flatten_white.setChecked(bool(self.config.get("white_background", True)))
        self.flatten_white.toggled.connect(self._schedule_save)
        options.addWidget(self.flatten_white)
        self.skip_pinyin = QCheckBox("Skip pinyin")
        self.skip_pinyin.setChecked(bool(self.config.get("skip_pinyin", False)))
        self.skip_pinyin.toggled.connect(self._schedule_save)
        options.addWidget(self.skip_pinyin)
        self.skip_romanization = QCheckBox("Skip romanization")
        self.skip_romanization.setChecked(bool(self.config.get("skip_romanization", False)))
        self.skip_romanization.toggled.connect(self._schedule_save)
        options.addWidget(self.skip_romanization)
        self.dedupe_cut_overlap = QCheckBox("Dedupe cut overlap")
        self.dedupe_cut_overlap.setChecked(bool(self.config.get("dedupe_cut_overlap", False)))
        self.dedupe_cut_overlap.toggled.connect(self._schedule_save)
        options.addWidget(self.dedupe_cut_overlap)
        self.streaming = QCheckBox("Streaming")
        self.streaming.setChecked(bool(self.config.get("streaming", True)))
        self.streaming.toggled.connect(self._schedule_save)
        options.addWidget(self.streaming)
        self.chunks_only = QCheckBox("Chunks only")
        self.chunks_only.setChecked(bool(self.config.get("chunks_only", False)))
        self.chunks_only.toggled.connect(self.update_key_state)
        self.chunks_only.toggled.connect(self._schedule_save)
        options.addWidget(self.chunks_only)
        self.use_chunks = QCheckBox("Use existing chunks")
        self.use_chunks.setChecked(bool(self.config.get("use_existing_chunks", False)))
        self.use_chunks.toggled.connect(self._schedule_save)
        options.addWidget(self.use_chunks)
        self.no_resume = QCheckBox("Reprocess all")
        self.no_resume.setToolTip("Process all images even if output already exists")
        self.no_resume.setChecked(bool(self.config.get("no_resume", False)))
        self.no_resume.toggled.connect(self._schedule_save)
        options.addWidget(self.no_resume)
        self.create_epub = QCheckBox("Create EPUB")
        self.create_epub.setChecked(bool(self.config.get("create_epub", False)))
        self.create_epub.toggled.connect(self._schedule_save)
        options.addWidget(self.create_epub)
        options.addStretch()
        layout.addLayout(options)

        # ---- Prompts ----
        layout.addWidget(QLabel("Prompts (leave blank to use built-in defaults):"))

        prompt_sys_row = QHBoxLayout()
        prompt_sys_row.addWidget(QLabel("System prompt:"))
        prompt_sys_row.addStretch()
        reset_sys = QPushButton("Reset to default")
        reset_sys.clicked.connect(self.reset_system_prompt)
        prompt_sys_row.addWidget(reset_sys)
        layout.addLayout(prompt_sys_row)

        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setAcceptRichText(False)
        saved_sys = self.config.get("system_prompt", "")
        if not saved_sys:
            saved_sys = ocr_pipeline.DEFAULT_VISION_OCR_PROMPT
        self.system_prompt_edit.setPlainText(saved_sys)
        self.system_prompt_edit.textChanged.connect(self._schedule_save)
        self.system_prompt_edit.setMaximumHeight(80)
        layout.addWidget(self.system_prompt_edit)

        prompt_usr_row = QHBoxLayout()
        prompt_usr_row.addWidget(QLabel("User prompt:"))
        prompt_usr_row.addStretch()
        reset_usr = QPushButton("Reset to default")
        reset_usr.clicked.connect(self.reset_user_prompt)
        prompt_usr_row.addWidget(reset_usr)
        layout.addLayout(prompt_usr_row)

        self.user_prompt_edit = QTextEdit()
        self.user_prompt_edit.setAcceptRichText(False)
        saved_usr = self.config.get("user_prompt", "")
        if not saved_usr:
            saved_usr = ocr_pipeline.DEFAULT_VISION_OCR_USER_PROMPT
        self.user_prompt_edit.setPlainText(saved_usr)
        self.user_prompt_edit.textChanged.connect(self._schedule_save)
        self.user_prompt_edit.setMaximumHeight(60)
        layout.addWidget(self.user_prompt_edit)

        # ---- Run / Stop ----
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run OCR")
        self.run_button.clicked.connect(self.run_ocr)
        run_row.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop After Current Image")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_worker)
        run_row.addWidget(self.stop_button)
        view_results_btn = QPushButton("View Results")
        view_results_btn.clicked.connect(self._open_results_browser)
        run_row.addWidget(view_results_btn)
        preview_btn = QPushButton("Preview Chunks")
        preview_btn.clicked.connect(self._open_chunk_preview)
        run_row.addWidget(preview_btn)
        cleanup_btn = QPushButton("Cleanup Rules")
        cleanup_btn.clicked.connect(self._open_cleanup_rules)
        run_row.addWidget(cleanup_btn)
        compare_btn = QPushButton("Compare")
        compare_btn.clicked.connect(self._open_model_compare)
        run_row.addWidget(compare_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=2)

        self._rebuild_model_combo()
        self._rebuild_profile_combo()
        self.update_key_state()
        self.save_settings()

    # ---- Model combo management ----

    def _rebuild_model_combo(self):
        """Refresh model combo items, preserving current selection."""
        current = self.model_combo.currentText() or str(self.config.get("model", ocr_pipeline._config.default_ocr_model))
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        builtins = [ocr_pipeline._config.google_cloud_vision_model] + get_model_options()
        all_items = list(dict.fromkeys(builtins + self._custom_models))
        self.model_combo.addItems(all_items)
        self.model_combo.setCurrentText(current)
        completer = QCompleter(all_items, self.model_combo)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.model_combo.setCompleter(completer)
        self.model_combo.blockSignals(False)

    def manage_custom_models(self):
        dialog = CustomModelsDialog(list(self._custom_models), self)
        if dialog.exec():
            self._custom_models = dialog.get_models()
            self._rebuild_model_combo()
            self.save_settings()

    # ---- Prompt resets ----

    def _profile_settings_keys(self):
        """Settings that belong in a profile (not API keys, paths, etc.)."""
        return (
            "model", "base_url", "batch_size", "api_delay", "temperature",
            "max_output_tokens", "send_thinking", "gemini_thinking_level",
            "openai_thinking_effort", "smart_chunking", "white_background",
            "skip_pinyin", "skip_romanization", "dedupe_cut_overlap",
            "streaming", "chunks_only", "use_existing_chunks",
            "system_prompt", "user_prompt",
        )

    def _collect_profile(self) -> dict:
        """Snapshot current widget values into a dict matching config keys."""
        return {
            "model": self.model_combo.currentText().strip(),
            "base_url": self.base_url_edit.text().strip(),
            "batch_size": self.batch_size_edit.text().strip(),
            "api_delay": self.api_delay_edit.text().strip(),
            "temperature": self.temperature_edit.text().strip(),
            "max_output_tokens": self.max_output_tokens_edit.text().strip(),
            "send_thinking": self.send_thinking.isChecked(),
            "gemini_thinking_level": self.gemini_thinking_combo.currentText().strip(),
            "openai_thinking_effort": self.openai_thinking_combo.currentText().strip(),
            "smart_chunking": self.smart_chunking.isChecked(),
            "white_background": self.flatten_white.isChecked(),
            "skip_pinyin": self.skip_pinyin.isChecked(),
            "skip_romanization": self.skip_romanization.isChecked(),
            "dedupe_cut_overlap": self.dedupe_cut_overlap.isChecked(),
            "streaming": self.streaming.isChecked(),
            "chunks_only": self.chunks_only.isChecked(),
            "use_existing_chunks": self.use_chunks.isChecked(),
            "system_prompt": self.system_prompt_edit.toPlainText().strip(),
            "user_prompt": self.user_prompt_edit.toPlainText().strip(),
        }

    def _apply_profile(self, profile: dict):
        """Apply saved profile values to widgets."""
        self.model_combo.setCurrentText(str(profile.get("model", "")))
        self.base_url_edit.setText(str(profile.get("base_url", "")))
        self.batch_size_edit.setText(str(profile.get("batch_size", "")))
        self.api_delay_edit.setText(str(profile.get("api_delay", "")))
        self.temperature_edit.setText(str(profile.get("temperature", "")))
        self.max_output_tokens_edit.setText(str(profile.get("max_output_tokens", "")))
        self.send_thinking.setChecked(bool(profile.get("send_thinking", False)))
        self.gemini_thinking_combo.setCurrentText(str(profile.get("gemini_thinking_level", "minimal")))
        self.openai_thinking_combo.setCurrentText(str(profile.get("openai_thinking_effort", "none")))
        self.smart_chunking.setChecked(bool(profile.get("smart_chunking", True)))
        self.flatten_white.setChecked(bool(profile.get("white_background", True)))
        self.skip_pinyin.setChecked(bool(profile.get("skip_pinyin", False)))
        self.skip_romanization.setChecked(bool(profile.get("skip_romanization", False)))
        self.dedupe_cut_overlap.setChecked(bool(profile.get("dedupe_cut_overlap", False)))
        self.streaming.setChecked(bool(profile.get("streaming", True)))
        self.chunks_only.setChecked(bool(profile.get("chunks_only", False)))
        self.use_chunks.setChecked(bool(profile.get("use_existing_chunks", False)))
        self.system_prompt_edit.setPlainText(str(profile.get("system_prompt", "")))
        self.user_prompt_edit.setPlainText(str(profile.get("user_prompt", "")))

    def _rebuild_profile_combo(self):
        current = self.profile_combo.currentText() if self.profile_combo.count() else ""
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        profiles = self.config.get("profiles", {})
        names = sorted(profiles.keys())
        self.profile_combo.addItem("— none —")
        self.profile_combo.insertSeparator(1)
        self.profile_combo.addItems(names)
        if current and current in names:
            self.profile_combo.setCurrentText(current)
        else:
            self.profile_combo.setCurrentIndex(0)
        self.profile_combo.blockSignals(False)

    def _load_profile(self, name):
        if not name or name == "— none —":
            return
        profiles = self.config.get("profiles", {})
        if name in profiles:
            self._apply_profile(profiles[name])
            self._schedule_save()

    def _save_profile(self):
        name, ok = QInputDialog.getText(self, "Save Profile", "Profile name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        profiles = self.config.get("profiles", {})
        profiles[name] = self._collect_profile()
        self.config["profiles"] = profiles
        self._rebuild_profile_combo()
        self.profile_combo.setCurrentText(name)
        save_config(self.config)

    def _delete_profile(self):
        name = self.profile_combo.currentText()
        if not name or name == "— none —":
            return
        reply = QMessageBox.question(self, "Delete Profile", f"Delete profile \"{name}\"?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        profiles = self.config.get("profiles", {})
        profiles.pop(name, None)
        self.config["profiles"] = profiles
        self._rebuild_profile_combo()
        save_config(self.config)

    def _open_results_browser(self):
        output_dir = self.output_edit.text().strip() or ocr_pipeline._config.default_output_dir
        dialog = ResultsBrowserDialog(output_dir, self)
        dialog.exec()

    def _open_model_compare(self):
        image_paths = [self.path_list.item(i).text() for i in range(self.path_list.count())]
        if not image_paths:
            QMessageBox.warning(self, "No images", "Add at least one image first.")
            return
        api_key = self.api_key_edit.text().strip()
        if not api_key:
            api_key = os.environ.get("GOOGLE_VISION_API_KEY", "")
        model_items = [ocr_pipeline._config.google_cloud_vision_model] + get_model_options()
        dialog = ModelCompareDialog(image_paths[0], model_items, api_key,
                                    self.base_url_edit.text().strip(), self)
        dialog.exec()

    def _open_cleanup_rules(self):
        output_dir = self.output_edit.text().strip() or ocr_pipeline._config.default_output_dir
        job_name = self.job_name_edit.text().strip()
        rules = list(self.config.get("cleanup_rules", []))
        dialog = CleanupRulesDialog(rules, output_dir, job_name, self)
        if dialog.exec():
            self.config["cleanup_rules"] = dialog.rules
            save_config(self.config)

    def _open_chunk_preview(self):
        image_paths = [self.path_list.item(i).text() for i in range(self.path_list.count())]
        if not image_paths:
            QMessageBox.warning(self, "No images", "Add at least one image first.")
            return
        dialog = ChunkPreviewDialog(image_paths[0], self)
        dialog.exec()

    def view_progress(self):
        job_name = self.job_name_edit.text().strip()
        if not job_name:
            QMessageBox.information(self, "Progress", "Enter a job name first.")
            return
        output_dir = self.output_edit.text().strip() or ocr_pipeline._config.default_output_dir
        results_dir = ocr_pipeline._results_dir(output_dir, job_name)
        progress_path = os.path.join(results_dir, "progress.json")

        if not os.path.exists(progress_path):
            QMessageBox.information(self, "Progress", f"No progress recorded yet for \"{job_name}\".\n\nResults will appear in:\n{results_dir}")
            return

        try:
            with open(progress_path, encoding="utf-8") as f:
                data = json.load(f)
            completed = sorted(data.get("completed", []))
        except (json.JSONDecodeError, OSError):
            QMessageBox.warning(self, "Progress", f"Could not read progress file:\n{progress_path}")
            return

        total = self.path_list.count() or len(completed) or 0

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Progress: {job_name}")
        dialog.resize(420, 350)

        dlg_layout = QVBoxLayout(dialog)
        summary = QLabel(f"{len(completed)} image(s) completed" + (f" (of {total} loaded)" if total else ""))
        summary.setStyleSheet("font-weight: 600; font-size: 13px;")
        dlg_layout.addWidget(summary)

        text = QTextEdit()
        text.setReadOnly(True)
        lines = [f"✓ {b}" for b in completed] if completed else ["(none yet)"]
        text.setPlainText("\n".join(lines))
        dlg_layout.addWidget(text)

        open_btn = QPushButton("Open Results Folder")
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(results_dir)))
        dlg_layout.addWidget(open_btn)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        dlg_layout.addWidget(ok_btn)

        dialog.exec()

    def reset_system_prompt(self):
        self.system_prompt_edit.setPlainText(ocr_pipeline.DEFAULT_VISION_OCR_PROMPT)

    def reset_user_prompt(self):
        self.user_prompt_edit.setPlainText(ocr_pipeline.DEFAULT_VISION_OCR_USER_PROMPT)

    # ---- File / path management ----

    def add_paths(self, paths):
        images = []
        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for name in sorted(files):
                        full = os.path.join(root, name)
                        if name.lower().endswith(ocr_pipeline._config.image_exts):
                            images.append(full)
            elif path.lower().endswith(ocr_pipeline._config.image_exts):
                images.append(path)

        existing = {self.path_list.item(i).text() for i in range(self.path_list.count())}
        for image in sorted(dict.fromkeys(images)):
            if image not in existing:
                self.path_list.addItem(QListWidgetItem(image))
                existing.add(image)

        # Auto-populate job name from common parent directory
        all_paths = [self.path_list.item(i).text() for i in range(self.path_list.count())]
        if all_paths:
            parents = {os.path.basename(os.path.dirname(p)) for p in all_paths}
            if len(parents) == 1:
                common = parents.pop()
                if common and not self.job_name_edit.text():
                    self.job_name_edit.setText(common)

        if not images:
            self.append_log("No supported image files found.")

    def pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.gif)",
        )
        self.add_paths(files)

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            self.add_paths([folder])

    def pick_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.output_edit.setText(folder)

    def pick_credentials(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select service-account JSON",
            "",
            "JSON files (*.json)",
        )
        if path:
            self.credentials_edit.setText(path)

    def remove_selected(self):
        for item in self.path_list.selectedItems():
            self.path_list.takeItem(self.path_list.row(item))

    # ---- Run OCR ----

    def run_ocr(self):
        api_key = self.api_key_edit.text().strip()
        credentials_json = self.credentials_edit.text().strip()
        model = self.model_combo.currentText().strip() or ocr_pipeline._config.default_ocr_model
        base_url = self.base_url_edit.text().strip()
        chunks_only = self.chunks_only.isChecked()

        # Auto-persist user-typed model names that aren't already known
        builtins = {ocr_pipeline._config.google_cloud_vision_model} | set(get_model_options())
        if model and model not in builtins and model not in self._custom_models:
            self._custom_models.append(model)
            self._rebuild_model_combo()
            self.save_settings()

        if model != ocr_pipeline._config.google_cloud_vision_model:
            credentials_json = ""
        try:
            batch_size = max(1, int(self.batch_size_edit.text().strip() or ocr_pipeline._config.max_workers))
        except ValueError:
            batch_size = ocr_pipeline._config.max_workers
        try:
            api_delay = max(0.0, float(self.api_delay_edit.text().strip() or ocr_pipeline._config.api_call_delay_seconds))
        except ValueError:
            api_delay = ocr_pipeline._config.api_call_delay_seconds
        try:
            temperature = float(self.temperature_edit.text().strip() or ocr_pipeline._config.temperature)
        except ValueError:
            temperature = ocr_pipeline._config.temperature
        try:
            max_output_tokens = max(1, int(self.max_output_tokens_edit.text().strip() or ocr_pipeline._config.max_output_tokens))
        except ValueError:
            max_output_tokens = ocr_pipeline._config.max_output_tokens

        is_local = model.lower().startswith("lmstudio/") or model.lower().startswith("ollama/")
        if not api_key and not credentials_json and not chunks_only and not is_local:
            QMessageBox.warning(
                self,
                "Missing credentials",
                "Enter an API key, select a service-account JSON file for Google Cloud Vision, or enable Chunks only.",
            )
            return

        image_paths = [self.path_list.item(i).text() for i in range(self.path_list.count())]
        if not image_paths:
            QMessageBox.warning(self, "No images", "Add or drop at least one image.")
            return

        output_dir = self.output_edit.text().strip()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.log.clear()
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = OcrWorker(
            image_paths,
            output_dir,
            api_key,
            credentials_json,
            model,
            base_url,
            batch_size,
            api_delay,
            temperature,
            max_output_tokens,
            self.streaming.isChecked(),
            self.send_thinking.isChecked(),
            self.gemini_thinking_combo.currentText().strip(),
            self.openai_thinking_combo.currentText().strip(),
            self.smart_chunking.isChecked(),
            chunks_only,
            self.flatten_white.isChecked(),
            self.skip_pinyin.isChecked(),
            self.skip_romanization.isChecked(),
            self.dedupe_cut_overlap.isChecked(),
            self.system_prompt_edit.toPlainText().strip(),
            self.user_prompt_edit.toPlainText().strip(),
            self.job_name_edit.text().strip() or "",
            self.use_chunks.isChecked(),
            self.no_resume.isChecked(),
            self.create_epub.isChecked(),
            list(self.config.get("cleanup_rules", [])),
        )
        self.worker.log.connect(self.append_log)
        self.worker.finished_ok.connect(self.worker_finished)
        self.worker.failed.connect(self.worker_failed)
        self.worker.start()

    def update_key_state(self):
        chunks_only = self.chunks_only.isChecked()
        model = self.model_combo.currentText().strip() if hasattr(self, "model_combo") else ""
        is_google_cloud = model == ocr_pipeline._config.google_cloud_vision_model
        thinking_enabled = self.send_thinking.isChecked() if hasattr(self, "send_thinking") else False
        base_url = self.base_url_edit.text().strip() if hasattr(self, "base_url_edit") else ""
        is_native_gemini = bool(model.lower().startswith("gemini")) and not base_url
        is_native_claude = bool(model.lower().startswith("claude")) and not base_url
        is_lmstudio = bool(model.lower().startswith("lmstudio/"))
        is_ollama = bool(model.lower().startswith("ollama/"))
        is_local = is_lmstudio or is_ollama
        is_thinking_applicable = not is_native_claude and not is_local
        self.api_key_edit.setEnabled(not chunks_only and not is_local)
        self.credentials_edit.setEnabled(not chunks_only and is_google_cloud)
        if hasattr(self, "gemini_thinking_combo"):
            self.gemini_thinking_combo.setEnabled(not chunks_only and thinking_enabled and is_native_gemini)
        if hasattr(self, "openai_thinking_combo"):
            self.openai_thinking_combo.setEnabled(not chunks_only and thinking_enabled and not is_google_cloud and is_thinking_applicable and not is_native_gemini)
        if is_local:
            self.api_key_edit.setPlaceholderText("Optional for local endpoints")
            self.credentials_edit.setPlaceholderText("Not used by local models")
        elif chunks_only:
            self.api_key_edit.setPlaceholderText("Not needed for chunk-only mode")
            self.credentials_edit.setPlaceholderText("Not needed for chunk-only mode")
        elif is_google_cloud:
            self.api_key_edit.setPlaceholderText("Google Cloud Vision API key, or use service JSON")
            self.credentials_edit.setPlaceholderText("Optional service-account JSON")
        else:
            self.api_key_edit.setPlaceholderText("Gemini, OpenAI, or Anthropic API key")
            self.credentials_edit.setPlaceholderText("Only used by google-cloud-vision")

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.append_log("Stop requested. Waiting for current image/request to finish.")

    def worker_finished(self):
        self.append_log("Done.")
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    def worker_failed(self, message):
        self.append_log(message)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None
        QMessageBox.critical(self, "OCR failed", message)

    def append_log(self, message):
        self.log.append(message)

    def _schedule_save(self):
        """Debounced save – coalesces rapid signal bursts into a single write."""
        self._save_timer.start()

    def save_settings(self):
        if not all(hasattr(self, name) for name in (
            "output_edit", "model_combo", "base_url_edit", "batch_size_edit", "api_delay_edit", "temperature_edit",
            "max_output_tokens_edit",
            "send_thinking", "gemini_thinking_combo", "openai_thinking_combo",
            "smart_chunking", "flatten_white", "skip_pinyin", "skip_romanization", "dedupe_cut_overlap", "streaming", "chunks_only",
            "api_key_edit", "credentials_edit",
            "system_prompt_edit", "user_prompt_edit",
            "job_name_edit", "use_chunks", "no_resume", "create_epub",
        )):
            return
        api_key = self.api_key_edit.text().strip()
        credentials_json = self.credentials_edit.text().strip()

        # Normalize prompts: save empty string when text matches default
        sys_prompt = self.system_prompt_edit.toPlainText().strip()
        if sys_prompt == ocr_pipeline.DEFAULT_VISION_OCR_PROMPT:
            sys_prompt = ""
        usr_prompt = self.user_prompt_edit.toPlainText().strip()
        if usr_prompt == ocr_pipeline.DEFAULT_VISION_OCR_USER_PROMPT:
            usr_prompt = ""

        self.config.update({
            "output_dir": self.output_edit.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "base_url": self.base_url_edit.text().strip(),
            "batch_size": self.batch_size_edit.text().strip(),
            "api_delay": self.api_delay_edit.text().strip(),
            "temperature": self.temperature_edit.text().strip(),
            "max_output_tokens": self.max_output_tokens_edit.text().strip(),
            "send_thinking": self.send_thinking.isChecked(),
            "gemini_thinking_level": self.gemini_thinking_combo.currentText().strip(),
            "openai_thinking_effort": self.openai_thinking_combo.currentText().strip(),
            "smart_chunking": self.smart_chunking.isChecked(),
            "white_background": self.flatten_white.isChecked(),
            "skip_pinyin": self.skip_pinyin.isChecked(),
            "skip_romanization": self.skip_romanization.isChecked(),
            "dedupe_cut_overlap": self.dedupe_cut_overlap.isChecked(),
            "streaming": self.streaming.isChecked(),
            "chunks_only": self.chunks_only.isChecked(),
            "use_existing_chunks": self.use_chunks.isChecked(),
            "no_resume": self.no_resume.isChecked(),
            "create_epub": self.create_epub.isChecked(),
            "job_name": self.job_name_edit.text().strip(),
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt,
            "custom_models": self._custom_models,
            "api_key_encrypted": encrypt_api_key(api_key) if api_key else "",
            "credentials_json_encrypted": encrypt_text(credentials_json) if credentials_json else "",
            "window_width": self.width(),
            "window_height": self.height(),
        })
        save_config(self.config)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.save_settings()

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
