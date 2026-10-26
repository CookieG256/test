from dataclasses import dataclass
from pathlib import Path

import face_recognition
import numpy as np

from config import FACE_MATCH_TOLERANCE, STUDENTS_DIR, TEACHERS_DIR


@dataclass(frozen=True)
class FaceMatch:
    name: str
    role: str
    known: bool


class FaceRecognizer:
    def __init__(self) -> None:
        encodings: list[np.ndarray] = []
        names: list[str] = []
        roles: list[str] = []
        self._load_directory(STUDENTS_DIR, "Estudiante", encodings, names, roles)
        self._load_directory(TEACHERS_DIR, "Profesor", encodings, names, roles)
        self.encodings = np.asarray(encodings) if encodings else np.empty((0, 128))
        self.names = names
        self.roles = roles
        print(f"[INFO] {len(self.names)} personas registradas en total.")

    @staticmethod
    def _load_directory(
        directory: Path,
        role: str,
        encodings: list[np.ndarray],
        names: list[str],
        roles: list[str],
    ) -> None:
        if not directory.exists():
            return

        for image_path in sorted(directory.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                image = face_recognition.load_image_file(str(image_path))
                face_encodings = face_recognition.face_encodings(image)
            except Exception as error:
                print(f"[AVISO] No se pudo cargar {image_path.name}: {error}")
                continue

            if face_encodings:
                encodings.append(face_encodings[0])
                names.append(image_path.stem.capitalize())
                roles.append(role)

    def identify(self, encoding: np.ndarray) -> FaceMatch:
        if not len(self.encodings):
            return FaceMatch("Desconocido", "Desconocido", False)

        distances = face_recognition.face_distance(self.encodings, encoding)
        best_index = int(np.argmin(distances))
        if distances[best_index] < FACE_MATCH_TOLERANCE:
            return FaceMatch(self.names[best_index], self.roles[best_index], True)
        return FaceMatch("Desconocido", "Desconocido", False)