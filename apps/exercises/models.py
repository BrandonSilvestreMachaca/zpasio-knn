"""Modelos del módulo de gestión de ejercicios de Zpasio."""

from django.db import models
from django.utils.translation import gettext_lazy as _


class Exercise(models.Model):
    """Ejercicio del catálogo del gimnasio.

    Queda preparado para relacionarse con el módulo de rutinas en
    actividades posteriores (relación many-to-many vía tabla intermedia).
    """

    class MuscleGroup(models.TextChoices):
        PECHO = "PECHO", _("Pecho")
        ESPALDA = "ESPALDA", _("Espalda")
        PIERNAS = "PIERNAS", _("Piernas")
        HOMBROS = "HOMBROS", _("Hombros")
        BICEPS = "BICEPS", _("Bíceps")
        TRICEPS = "TRICEPS", _("Tríceps")
        ABDOMEN = "ABDOMEN", _("Abdomen/Core")
        CUERPO_COMPLETO = "CUERPO_COMPLETO", _("Cuerpo completo")
        OTRO = "OTRO", _("Otro")

    class Difficulty(models.TextChoices):
        PRINCIPIANTE = "PRINCIPIANTE", _("Principiante")
        INTERMEDIO = "INTERMEDIO", _("Intermedio")
        AVANZADO = "AVANZADO", _("Avanzado")

    name = models.CharField(_("nombre"), max_length=120)

    description = models.TextField(_("descripción"), blank=True)

    muscle_group = models.CharField(
        _("grupo muscular"),
        max_length=20,
        choices=MuscleGroup.choices,
    )

    difficulty = models.CharField(
        _("nivel de dificultad"),
        max_length=20,
        choices=Difficulty.choices,
    )

    created_at = models.DateTimeField(_("creado"), auto_now_add=True)

    is_active = models.BooleanField(
        _("activo"),
        default=True,
        help_text=_("Los ejercicios inactivos no se ofrecen en el catálogo."),
    )

    class Meta:
        verbose_name = _("ejercicio")
        verbose_name_plural = _("ejercicios")
        ordering = ("name",)

    def __str__(self):
        return self.name
