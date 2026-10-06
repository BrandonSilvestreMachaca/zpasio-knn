"""Modelos de perfiles y datos físicos de Zpasio."""

from datetime import date
from decimal import Decimal

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
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

    available_days_per_week = models.PositiveSmallIntegerField(
        _("días disponibles por semana"),
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(7)],
        help_text=_(
            "Días por semana que el usuario tiene disponibles para "
            "entrenar (entre 1 y 7)."
        ),
    )

    session_time_available_minutes = models.PositiveIntegerField(
        _("tiempo disponible por sesión (minutos)"),
        null=True,
        blank=True,
        validators=[MinValueValidator(1)],
        help_text=_("Minutos que el usuario puede dedicar a cada sesión."),
    )

    previous_training_days_per_week = models.PositiveSmallIntegerField(
        _("frecuencia previa de entrenamiento (días/semana)"),
        null=True,
        blank=True,
        validators=[MaxValueValidator(7)],
        help_text=_(
            "Días por semana que entrenaba antes de empezar (0 a 7)."
        ),
    )

    created_at = models.DateTimeField(_("creado"), auto_now_add=True)
    updated_at = models.DateTimeField(_("actualizado"), auto_now=True)

    class Meta:
        verbose_name = _("perfil")
        verbose_name_plural = _("perfiles")

    def __str__(self):
        return f"Perfil de {self.user.username}"

    @property
    def age(self):
        """Edad en años calculada a partir de la fecha de nacimiento."""
        if not self.birth_date:
            return None
        today = date.today()
        age = today.year - self.birth_date.year
        if (today.month, today.day) < (
            self.birth_date.month,
            self.birth_date.day,
        ):
            age -= 1
        return age


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

    waist_circumference_cm = models.DecimalField(
        _("circunferencia de cintura (cm)"),
        max_digits=5,
        decimal_places=1,
        null=True,
        blank=True,
        validators=[MinValueValidator(Decimal("0.01"))],
        help_text=_("Medida opcional de cintura en centímetros."),
    )

    body_fat_percentage = models.DecimalField(
        _("porcentaje de grasa corporal"),
        max_digits=4,
        decimal_places=1,
        null=True,
        blank=True,
        validators=[
            MinValueValidator(Decimal("0")),
            MaxValueValidator(Decimal("100")),
        ],
        help_text=_(
            "Dato medido (no calculado); no se utiliza para "
            "clasificaciones automáticas."
        ),
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

    # Clasificación del IMC (rango: etiqueta, clase CSS)
    BMI_CATEGORIES = (
        (Decimal("18.5"), "Bajo peso", "underweight"),
        (Decimal("25.0"), "Peso normal", "normal"),
        (Decimal("30.0"), "Sobrepeso", "overweight"),
        (None, "Obesidad", "obesity"),
    )

    @property
    def bmi_classification(self) -> str:
        """Etiqueta de clasificación del IMC según los rangos estándar OMS."""
        classification = self.classify_bmi(self.bmi)
        return classification[0] if classification else ""

    @property
    def bmi_classification_css_class(self) -> str:
        """Clase CSS asociada a la clasificación del IMC."""
        classification = self.classify_bmi(self.bmi)
        return classification[1] if classification else ""

    @staticmethod
    def classify_bmi(bmi: Decimal):
        """Clasifica numéricamente un IMC.

        Devuelve una tupla (etiqueta, clase_css) o None si el IMC no existe.
        Las comparaciones son numéricas para evitar problemas de límites.
        """
        if bmi is None:
            return None
        bmi = Decimal(bmi)
        for threshold, label, css_class in PhysicalData.BMI_CATEGORIES:
            if threshold is None or bmi < threshold:
                return label, css_class
        return None

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