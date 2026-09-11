from .base import *  # noqa: F401, F403

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# Sprint 1: usar SQLite para no depender de PostgreSQL durante el desarrollo.
# La configuración de PostgreSQL se mantiene intacta en `base.py` para usarla
# posteriormente (por ejemplo, en producción o cuando esté disponible).
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}