"""Modelos y lógica de la aplicación de cuentas de Zpasio."""

from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _


class CustomUser(AbstractUser):
    """Usuario personalizado con un rol que distingue entrenadores y clientes."""

    class Role(models.TextChoices):
        ENTRENADOR = "ENTRENADOR", _("Entrenador")
        USUARIO = "USUARIO", _("Usuario")

    role = models.CharField(
        _("rol"),
        max_length=20,
        choices=Role.choices,
        default=Role.USUARIO,
        help_text=_("Rol del usuario dentro del sistema."),
    )

    class Meta:
        verbose_name = _("usuario")
        verbose_name_plural = _("usuarios")

    def __str__(self):
        return self.username

    @property
    def full_name(self) -> str:
        """Nombre completo combinando nombre y apellido."""
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def is_entrenador(self) -> bool:
        return self.role == self.Role.ENTRENADOR

    @property
    def is_usuario(self) -> bool:
        return self.role == self.Role.USUARIO