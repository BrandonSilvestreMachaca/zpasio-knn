"""Modelos de perfiles y datos físicos de Zpasio."""

from decimal import Decimal

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _


class Profile(models.Model):
    """Datos personales y de entrenamiento del usuario."""

    class Goal(models.TextChoices):
        PERDIDA_PESO = "PERDIDA_PESO", _("Pérdida de peso")
        GANANCIA_MUSCULAR = "GANANCIA_MUSCULAR", _("Ganancia muscular")
        MANTENIMIENTO = "MANTENIMIENTO", _("Mantenimiento")
        RESISTENCIA = "RESISTENCIA", _("Mejora de resistencia/condición física")

    class ExperienceLevel(models.TextChoices):
        PRINCIPIANTE = "PRINCIPIANTE", _("Principiante")
        INTERMEDIO = "INTERMEDIO", _("Intermedio")
        AVANZADO = "AVANZADO", _("Avanzado")

    class Gender(models.TextChoices):
        MASCULINO = "MASCULINO", _("Masculino")
        FEMENINO = "FEMENINO", _("Femenino")
        OTRO = "OTRO", _("Otro")
        PREFIERO_NO_DECIR = "PREFIERO_NO_DECIR", _("Prefiero no decirlo")

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
        verbose_name=_("usuario"),
    )

    phone = models.CharField(
        _("teléfono"),
        max_length=20,
        blank=True,
    )

    birth_date = models.DateField(
        _("fecha de nacimiento"),
        null=True,
        blank=True,
    )

    gender = models.CharField(
        _("género"),
        max_length=20,
        choices=Gender.choices,
        blank=True,
    )

    goal = models.CharField(
        _("objetivo"),
        max_length=30,
        choices=Goal.choices,
        blank=True,
    )

    experience_level = models.CharField(
        _("nivel de experiencia"),
        max_length=20,
        choices=ExperienceLevel.choices,
        blank=True,
    )

    created_at = models.DateTimeField(_("creado"), auto_now_add=True)
    updated_at = models.DateTimeField(_("actualizado"), auto_now=True)

    class Meta:
        verbose_name = _("perfil")
        verbose_name_plural = _("perfiles")

    def __str__(self):
        return f"Perfil de {self.user.username}"


class PhysicalData(models.Model):
    """Datos físicos medibles del usuario.

    En el Sprint 1 se almacena únicamente el registro *actual* mediante una
    relación OneToOne. La estructura `user` + `measured_at` está pensada para
    poder convertirse en un historial (ForeignKey) en sprints posteriores sin
    cambiar el resto del modelo.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="physical_data",
        verbose_name=_("usuario"),
    )

    weight_kg = models.DecimalField(
        _("peso (kg)"),
        max_digits=5,
        decimal_places=2,
    )

    height_cm = models.DecimalField(
        _("altura (cm)"),
        max_digits=5,
        decimal_places=2,
    )

    bmi = models.DecimalField(
        _("IMC"),
        max_digits=4,
        decimal_places=2,
        editable=False,
        help_text=_("Calculado automáticamente a partir de peso y altura."),
    )

    body_fat_percentage = models.DecimalField(
        _("porcentaje de grasa corporal"),
        max_digits=4,
        decimal_places=1,
        null=True,
        blank=True,
    )

    measured_at = models.DateTimeField(_("fecha de medición"), auto_now=True)

    class Meta:
        verbose_name = _("dato físico")
        verbose_name_plural = _("datos físicos")

    def __str__(self):
        return f"Datos físicos de {self.user.username}"

    def clean(self):
        super().clean()
        if self.weight_kg is not None and self.weight_kg <= 0:
            raise ValidationError(
                {"weight_kg": _("El peso debe ser un valor positivo.")}
            )
        if self.height_cm is not None and self.height_cm <= 0:
            raise ValidationError(
                {"height_cm": _("La altura debe ser un valor positivo.")}
            )

    def save(self, *args, **kwargs):
        self.bmi = self.calculate_bmi(self.weight_kg, self.height_cm)
        super().save(*args, **kwargs)

    @staticmethod
    def calculate_bmi(weight_kg: Decimal, height_cm: Decimal) -> Decimal:
        """Calcula el IMC = peso (kg) / altura (m)^2.

        Devuelve el valor redondeado a dos decimales, o None si alguno de los
        datos no es válido.
        """
        if not weight_kg or not height_cm:
            return None
        height_m = Decimal(height_cm) / Decimal("100")
        if height_m <= 0:
            return None
        bmi = Decimal(weight_kg) / (height_m * height_m)
        return bmi.quantize(Decimal("0.01"))