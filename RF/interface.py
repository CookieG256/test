import base64
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

import cv2
import numpy as np

from recognition import FaceMatch

BACKGROUND = "#111C1D"
SIDEBAR = "#1B2928"
SURFACE = "#223230"
VIDEO_BACKGROUND = "#0B1111"
TEXT = "#F1F4ED"
MUTED = "#A0B2AA"
MINT = "#A9E6C5"
TEAL = "#52C8B0"
AMBER = "#E9B76C"
RED = "#E87970"


def draw_matches(
    frame: np.ndarray,
    face_locations: list[tuple[int, int, int, int]],
    matches: list[FaceMatch],
) -> np.ndarray:
    for (top, right, bottom, left), match in zip(face_locations, matches):
        color = (0, 0, 220) if not match.known else (
            (65, 185, 95) if match.role == "Estudiante" else (25, 145, 235)
        )
        label = f"{match.name} ({match.role})"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            frame,
            label,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return frame


def _to_photo(frame: np.ndarray, width: int, height: int) -> tk.PhotoImage:
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    resized = cv2.resize(
        frame,
        (max(1, int(frame_width * scale)), max(1, int(frame_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    success, encoded_frame = cv2.imencode(".png", rgb_frame)
    if not success:
        raise RuntimeError("No se pudo convertir el fotograma a PNG.")
    image_data = base64.b64encode(encoded_frame).decode("ascii")
    return tk.PhotoImage(data=image_data, format="png")


class AccessApp:
    def __init__(
        self,
        root: tk.Tk,
        start_camera: Callable[[], tuple[bool, str]],
        stop_camera: Callable[[], None],
        read_frame: Callable[[], tuple[np.ndarray | None, int, list[tuple[str, str, str]], str | None]],
    ) -> None:
        self.root = root
        self._start_camera = start_camera
        self._stop_camera = stop_camera
        self._read_frame = read_frame
        self._running = False
        self._after_id: str | None = None
        self._photo: tk.PhotoImage | None = None
        self._event_count = 0
        self._unknown_count = 0

        self.root.title("Pórtico | Control de acceso")
        self.root.geometry("1440x900")
        self.root.minsize(1080, 700)
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_styles()
        self._build_layout()
        self._update_clock()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Activity.Treeview",
            background=SURFACE,
            fieldbackground=SURFACE,
            foreground=TEXT,
            borderwidth=0,
            rowheight=37,
            font=("Bahnschrift", 10),
        )
        style.configure(
            "Activity.Treeview.Heading",
            background=SIDEBAR,
            foreground=MUTED,
            borderwidth=0,
            font=("Bahnschrift", 9, "bold"),
        )
        style.map("Activity.Treeview", background=[("selected", "#36574D")])

    def _build_layout(self) -> None:
        shell = tk.Frame(self.root, bg=BACKGROUND)
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(shell, bg=SIDEBAR, width=256, padx=24, pady=28)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="PÓRTICO",
            bg=SIDEBAR,
            fg=MINT,
            font=("Bahnschrift SemiBold", 22),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            sidebar,
            text="CONTROL DE ACCESO",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Bahnschrift", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(2, 32))

        self.camera_status = tk.Label(
            sidebar,
            text="●  CÁMARA APAGADA",
            bg="#293A37",
            fg=MUTED,
            font=("Bahnschrift", 9, "bold"),
            padx=12,
            pady=11,
            anchor="w",
        )
        self.camera_status.pack(fill="x", pady=(0, 18))

        self.start_button = tk.Button(
            sidebar,
            text="INICIAR MONITOREO",
            command=self.start,
            bg=TEAL,
            fg="#10201D",
            activebackground=MINT,
            activeforeground="#10201D",
            relief="flat",
            font=("Bahnschrift", 10, "bold"),
            cursor="hand2",
            padx=10,
            pady=13,
        )
        self.start_button.pack(fill="x")
        self.stop_button = tk.Button(
            sidebar,
            text="DETENER CÁMARA",
            command=self.stop,
            bg=SIDEBAR,
            fg=TEXT,
            activebackground=SURFACE,
            activeforeground=TEXT,
            relief="flat",
            font=("Bahnschrift", 10, "bold"),
            cursor="hand2",
            padx=10,
            pady=12,
            state="disabled",
        )
        self.stop_button.pack(fill="x", pady=(8, 0))

        tk.Frame(sidebar, bg="#344440", height=1).pack(fill="x", pady=18)
        tk.Label(
            sidebar,
            text="ALTA DE PERSONAS",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Bahnschrift", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 7))
        tk.Label(
            sidebar,
            text="Estudiantes: cadastros/alunos/\nProfesores: cadastros/professores/\nUna foto por persona. Reinicia la aplicación para cargar las altas.",
            bg=SIDEBAR,
            fg=TEXT,
            font=("Bahnschrift", 9),
            anchor="w",
            justify="left",
            wraplength=208,
        ).pack(fill="x")
        tk.Frame(sidebar, bg="#344440", height=1).pack(fill="x", pady=18)
        tk.Label(
            sidebar,
            text="EN ESTA SESIÓN",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Bahnschrift", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 14))
        self._make_stat(sidebar, "ROSTROS DETECTADOS", "faces", MINT)
        self._make_stat(sidebar, "REGISTROS", "events", TEAL)
        self._make_stat(sidebar, "DESCONOCIDOS", "unknowns", AMBER)

        tk.Label(
            sidebar,
            text="ESCUELA  /  ACCESO PRINCIPAL",
            bg=SIDEBAR,
            fg="#7F9289",
            font=("Bahnschrift", 8),
            anchor="w",
        ).pack(side="bottom", fill="x")

        content = tk.Frame(shell, bg=BACKGROUND, padx=30, pady=25)
        content.pack(side="left", fill="both", expand=True)
        header = tk.Frame(content, bg=BACKGROUND)
        header.pack(fill="x", pady=(0, 22))
        title_group = tk.Frame(header, bg=BACKGROUND)
        title_group.pack(side="left")
        tk.Label(
            title_group,
            text="Monitoreo",
            bg=BACKGROUND,
            fg=TEXT,
            font=("Bahnschrift SemiBold", 24),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            title_group,
            text="Reconocimiento y registro de accesos en tiempo real",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Bahnschrift", 10),
            anchor="w",
        ).pack(anchor="w", pady=(3, 0))
        self.clock_label = tk.Label(
            header,
            text="",
            bg=BACKGROUND,
            fg=MINT,
            font=("Bahnschrift", 12),
            anchor="e",
        )
        self.clock_label.pack(side="right", anchor="n", pady=5)

        body = tk.Frame(content, bg=BACKGROUND)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=285)
        body.grid_rowconfigure(0, weight=1)

        video_panel = tk.Frame(body, bg=VIDEO_BACKGROUND, highlightthickness=1, highlightbackground="#31423E")
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        video_header = tk.Frame(video_panel, bg=VIDEO_BACKGROUND, padx=17, pady=13)
        video_header.pack(fill="x")
        tk.Label(
            video_header,
            text="CÁMARA EN VIVO",
            bg=VIDEO_BACKGROUND,
            fg=TEXT,
            font=("Bahnschrift", 9, "bold"),
        ).pack(side="left")
        tk.Label(
            video_header,
            text="CAM 01",
            bg=VIDEO_BACKGROUND,
            fg=MUTED,
            font=("Bahnschrift", 9),
        ).pack(side="right")
        self.video_label = tk.Label(
            video_panel,
            text="La cámara está apagada\n\nInicia el monitoreo para ver la imagen.",
            bg=VIDEO_BACKGROUND,
            fg=MUTED,
            font=("Bahnschrift", 13),
            justify="center",
        )
        self.video_label.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        activity_panel = tk.Frame(body, bg=SURFACE, padx=16, pady=16)
        activity_panel.grid(row=0, column=1, sticky="nsew")
        tk.Label(
            activity_panel,
            text="Actividad reciente",
            bg=SURFACE,
            fg=TEXT,
            font=("Bahnschrift SemiBold", 16),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            activity_panel,
            text="Registros de esta sesión",
            bg=SURFACE,
            fg=MUTED,
            font=("Bahnschrift", 9),
            anchor="w",
        ).pack(fill="x", pady=(2, 14))
        self.activity = ttk.Treeview(
            activity_panel,
            columns=("time", "action", "person"),
            show="headings",
            style="Activity.Treeview",
            height=15,
        )
        self.activity.heading("time", text="HORA")
        self.activity.heading("action", text="ACCIÓN")
        self.activity.heading("person", text="PERSONA")
        self.activity.column("time", width=65, minwidth=65, stretch=False)
        self.activity.column("action", width=88, minwidth=75, stretch=False)
        self.activity.column("person", width=125, minwidth=90, stretch=True)
        self.activity.pack(fill="both", expand=True)
        self.activity.tag_configure("unknown", foreground=AMBER)
        self.activity.tag_configure("known", foreground=TEXT)

        footer = tk.Frame(content, bg=BACKGROUND)
        footer.pack(fill="x", pady=(14, 0))
        self.message_label = tk.Label(
            footer,
            text="Listo para comenzar",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Bahnschrift", 9),
            anchor="w",
        )
        self.message_label.pack(side="left")
        tk.Label(
            footer,
            text="ESCUELA  /  RECONOCIMIENTO FACIAL",
            bg=BACKGROUND,
            fg="#7F9289",
            font=("Bahnschrift", 8),
        ).pack(side="right")

    def _make_stat(self, parent: tk.Widget, title: str, key: str, color: str) -> None:
        block = tk.Frame(parent, bg=SIDEBAR)
        block.pack(fill="x", pady=(0, 15))
        tk.Label(
            block,
            text=title,
            bg=SIDEBAR,
            fg=MUTED,
            font=("Bahnschrift", 8, "bold"),
            anchor="w",
        ).pack(fill="x")
        value = tk.Label(
            block,
            text="0",
            bg=SIDEBAR,
            fg=color,
            font=("Bahnschrift SemiBold", 22),
            anchor="w",
        )
        value.pack(fill="x", pady=(2, 0))
        setattr(self, f"_{key}_label", value)

    def _update_clock(self) -> None:
        from datetime import datetime

        self.clock_label.configure(text=datetime.now().strftime("%d/%m/%Y   %H:%M:%S"))
        self.root.after(1000, self._update_clock)

    def start(self) -> None:
        if self._running:
            return
        started, message = self._start_camera()
        self.message_label.configure(text=message, fg=MUTED if started else RED)
        if not started:
            return

        self._running = True
        self.camera_status.configure(text="●  CÁMARA ACTIVA", fg=MINT)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._tick()

    def stop(self, message: str = "Monitoreo detenido") -> None:
        self._running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._stop_camera()
        self.camera_status.configure(text="●  CÁMARA APAGADA", fg=MUTED)
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.message_label.configure(text=message, fg=MUTED)
        self._faces_label.configure(text="0")

    def _tick(self) -> None:
        if not self._running:
            return
        frame, face_count, events, error = self._read_frame()
        if error:
            self.stop(error)
            return
        if frame is not None:
            width = max(1, self.video_label.winfo_width() - 12)
            height = max(1, self.video_label.winfo_height() - 12)
            self._photo = _to_photo(frame, width, height)
            self.video_label.configure(image=self._photo, text="")
            self._faces_label.configure(text=str(face_count))

        for time_text, action, person in events:
            is_unknown = action == "TENTATIVA"
            self.activity.insert(
                "",
                0,
                values=(time_text, action, person),
                tags=("unknown" if is_unknown else "known",),
            )
            self._event_count += 1
            if is_unknown:
                self._unknown_count += 1
        self._events_label.configure(text=str(self._event_count))
        self._unknowns_label.configure(text=str(self._unknown_count))
        self._after_id = self.root.after(20, self._tick)

    def close(self) -> None:
        self.stop("Aplicación cerrada")
        self.root.destroy()