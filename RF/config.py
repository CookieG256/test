from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STUDENTS_DIR = BASE_DIR / "cadastros" / "alunos"
TEACHERS_DIR = BASE_DIR / "cadastros" / "professores"
UNKNOWN_DIR = BASE_DIR / "desconhecidos_capturados"
DATABASE_PATH = BASE_DIR / "registro_escola.db"
CSV_PATH = BASE_DIR / "registro_escola.csv"

CAMERA_INDEX = 0
FRAME_SCALE = 0.33
RECOGNITION_INTERVAL_SECONDS = 0.20
FACE_MATCH_TOLERANCE = 0.55
ACCESS_COOLDOWN_SECONDS = 30
UNKNOWN_COOLDOWN_SECONDS = 10