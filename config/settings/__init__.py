"""Point of entry for Django settings.

Importa la configuración por defecto (desarrollo). Para producción usa
`DJANGO_SETTINGS_MODULE=config.settings.production`.
"""
from .development import *  # noqa: F401, F403