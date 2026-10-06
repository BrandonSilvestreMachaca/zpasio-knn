"""Modelos del módulo de evaluaciones físicas de Zpasio.

Una evaluación física es registrada por un entrenador sobre un usuario
(p. ej. la evaluación inicial) y contiene resultados de pruebas
objetivas (valor + unidad). El historial se conserva: cada evaluación
nueva es un registro aparte, nunca se sobrescribe una anterior.
"""

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _


class PhysicalEvaluation(models.Model):
    """Evaluación física realizada a un usuario por un entrenador."""

    class EvaluationType(models.TextChoices):
        INICIAL = "INICIAL", _("Inicial")
        SEGUIMIENTO = "SEGUIMIENTO", _("Seguimiento")
        FINAL = "FINAL", _("Final")

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="physical_evaluations",
        verbose_name=_("usuario"),
        help_text=_("Usuario evaluado."),
    )

    trainer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="evaluations_performed",
        verbose_name=_("entrenador"),
        help_text=_("Entrenador que registra la evaluación."),
    )

    evaluation_type = models.CharField(
        _("tipo de evaluación"),
        max_length=20,
        choices=EvaluationType.choices,
        default=EvaluationType.INICIAL,
    )

    conducted_at = models.DateTimeField(
        _("fecha de evaluación"),
        auto_now_add=True,
    )

    observations = models.TextField(
        _("observaciones"),
        blank=True,
    )

    class Meta:
        verbose_name = _("evaluación física")
        verbose_name_plural = _("evaluaciones físicas")
        ordering = ("-conducted_at",)

    def __str__(self):
        return (
            f"{self.get_evaluation_type_display()} de {self.user} — "
            f"{self.conducted_at:%d/%m/%Y}"
        )

    def clean(self):
        super().clean()
        if (
            self.trainer_id
            and getattr(self.trainer, "role", None) != "ENTRENADOR"
        ):
            raise ValidationError(
                {
                    "trainer": _(
                        "La evaluación debe registrarla un entrenador."
                    )
                }
            )
        if self.user_id and self.user_id == self.trainer_id:
            raise ValidationError(
                _("El entrenador no puede evaluarse a sí mismo.")
            )


class EvaluationResult(models.Model):
    """Resultado objetivo de una prueba dentro de una evaluación.

    Solo almacena mediciones observables: nombre de la prueba, valor y
    unidad (p. ej. 20 repeticiones, 60 segundos). No guarda puntuaciones
    subjetivas ni valores derivados.
    """

    evaluation = models.ForeignKey(
        PhysicalEvaluation,
        on_delete=models.CASCADE,
        related_name="results",
        verbose_name=_("evaluación"),
    )

    test_name = models.CharField(
        _("nombre de la prueba"),
        max_length=120,
    )

    value = models.DecimalField(
        _("valor obtenido"),
        max_digits=8,
        decimal_places=2,
    )

    unit = models.CharField(
        _("unidad de medida"),
        max_length=30,
        help_text=_("Por ejemplo: repeticiones, segundos, cm, kg."),
    )

    notes = models.CharField(
        _("observación"),
        max_length=200,
        blank=True,
    )

    class Meta:
        verbose_name = _("resultado de evaluación")
        verbose_name_plural = _("resultados de evaluación")
        ordering = ("id",)

    def __str__(self):
        return f"{self.test_name}: {self.value} {self.unit}"

    def clean(self):
        super().clean()
        if self.value is not None and self.value < 0:
            raise ValidationError(
                {"value": _("El resultado no puede ser negativo.")}
            )
