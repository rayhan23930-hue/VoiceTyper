"""
Free offline voice typing for Windows (MVP)
- Click the mic (or press Ctrl+Alt+Space), speak, click/press again to stop.
- Speech is transcribed offline with faster-whisper and pasted into the active app.
"""
import sys
import threading

import keyboard
import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
                               QPushButton, QWidget)

# ---------- Settings you can change ----------
MODEL_SIZE = "small"          # tiny / base / small / medium / large-v3-turbo
SAMPLE_RATE = 16000
HOTKEY = "ctrl+alt+space"
LANGUAGES = {"Bengali": "bn", "English": "en", "Auto-detect": None}
# ---------------------------------------------

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the Whisper model once. The first run downloads it (needs internet)."""
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        return _model


class Transcriber(QThread):
    status = Signal(str)
    text_ready = Signal(str)
    failed = Signal(str)

    def __init__(self, audio, language):
        super().__init__()
        self.audio = audio
        self.language = language

    def run(self):
        try:
            if _model is None:
                self.status.emit("⏳ Loading model (first run downloads it)...")
            model = get_model()
            self.status.emit("🤖 Transcribing...")
            segments, _info = model.transcribe(
                self.audio,
                language=self.language,
                vad_filter=True,
                beam_size=5,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            self.text_ready.emit(text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class VoiceWidget(QWidget):
    hotkey_pressed = Signal()

    def __init__(self):
        super().__init__()
        self.recording = False
        self.busy = False
        self.frames = []
        self.stream = None
        self.worker = None
        self.drag_pos = None

        # Frameless, always on top, never takes focus
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
                            Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.NoFocus)

        self.setStyleSheet("""
            QWidget { background-color: #1E1E1E; color: #FFFFFF; font-size: 13px; }
            QPushButton { background-color: #2D2D2D; border: none; border-radius: 6px;
                          padding: 6px 10px; }
            QPushButton:hover { background-color: #3A3A3A; }
            QComboBox { background-color: #2D2D2D; border: none; border-radius: 6px;
                        padding: 5px 8px; }
            QComboBox QAbstractItemView { background-color: #2D2D2D; color: #FFFFFF; }
        """)

        self.mic_btn = QPushButton("🎙️")
        self.mic_btn.setFixedSize(42, 34)
        self.mic_btn.setFocusPolicy(Qt.NoFocus)
        self.mic_btn.clicked.connect(self.toggle)

        self.lang_box = QComboBox()
        self.lang_box.addItems(LANGUAGES.keys())
        self.lang_box.setFocusPolicy(Qt.NoFocus)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setMinimumWidth(190)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setFocusPolicy(Qt.NoFocus)
        close_btn.clicked.connect(self.close_app)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self.mic_btn)
        layout.addWidget(self.lang_box)
        layout.addWidget(self.status_lbl)
        layout.addWidget(close_btn)

        # Global hotkey (works while another app is focused)
        self.hotkey_pressed.connect(self.toggle)
        try:
            keyboard.add_hotkey(HOTKEY, lambda: self.hotkey_pressed.emit())
        except Exception:  # noqa: BLE001
            self.set_status("Hotkey unavailable")

    # ----- window behaviour -----
    def showEvent(self, event):
        super().showEvent(event)
        if sys.platform == "win32":
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                                style | WS_EX_NOACTIVATE)
        # Place at bottom centre of the screen
        screen = QApplication.primaryScreen().availableGeometry()
        self.adjustSize()
        self.move(screen.center().x() - self.width() // 2,
                  screen.bottom() - self.height() - 40)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_pos)

    def mouseReleaseEvent(self, event):
        self.drag_pos = None

    def close_app(self):
        try:
            keyboard.unhook_all()
        except Exception:  # noqa: BLE001
            pass
        QApplication.quit()

    # ----- helpers -----
    def set_status(self, text):
        self.status_lbl.setText(text)

    def reset_ui(self):
        self.busy = False
        self.mic_btn.setText("🎙️")
        self.mic_btn.setStyleSheet("")
        self.set_status("Ready")

    # ----- recording flow -----
    def toggle(self):
        if self.busy:
            return
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def _audio_callback(self, indata, frames, time_info, status):
        self.frames.append(indata.copy())

    def start_recording(self):
        try:
            self.frames = []
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                         dtype="float32",
                                         callback=self._audio_callback)
            self.stream.start()
        except Exception as exc:  # noqa: BLE001
            self.set_status("Mic error")
            print("Microphone error:", exc)
            return
        self.recording = True
        self.mic_btn.setText("⏹")
        self.mic_btn.setStyleSheet("background-color: #C42B1C;")
        self.set_status("🎙️ Listening...")

    def stop_recording(self):
        self.recording = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:  # noqa: BLE001
            pass
        self.mic_btn.setText("🎙️")
        self.mic_btn.setStyleSheet("")

        if not self.frames:
            self.reset_ui()
            return
        audio = np.concatenate(self.frames).flatten().astype(np.float32)
        if len(audio) < SAMPLE_RATE * 0.3:
            self.set_status("Too short")
            QTimer.singleShot(1200, self.reset_ui)
            return

        self.busy = True
        language = LANGUAGES[self.lang_box.currentText()]
        self.worker = Transcriber(audio, language)
        self.worker.status.connect(self.set_status)
        self.worker.text_ready.connect(self.paste_text)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_error(self, message):
        print("Transcription error:", message)
        self.set_status("❌ Error (see console)")
        QTimer.singleShot(2500, self.reset_ui)

    # ----- typing into the active app -----
    def paste_text(self, text):
        if not text:
            self.set_status("No speech detected")
            QTimer.singleShot(1500, self.reset_ui)
            return
        clipboard = QApplication.clipboard()
        old_text = clipboard.text()
        clipboard.setText(text + " ")
        self.set_status("✍️ Typing...")
        QTimer.singleShot(80, lambda: self._send_paste(old_text))

    def _send_paste(self, old_text):
        keyboard.send("ctrl+v")

        def finish():
            QApplication.clipboard().setText(old_text)  # restore clipboard
            self.set_status("✅ Done!")
            QTimer.singleShot(1200, self.reset_ui)

        QTimer.singleShot(500, finish)


def main():
    app = QApplication(sys.argv)
    widget = VoiceWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
