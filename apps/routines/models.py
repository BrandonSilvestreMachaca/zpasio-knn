"""Modelos del módulo de gestión de rutinas de entrenamiento de Zpasio."""

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils.translation import gettext_lazy as _

from apps.exercises.models import Exercise
from apps.profiles.models import Profile


class Routine(models.Model):
    """Rutina de entrenamiento creada por un entrenador.

    Los objetivos y niveles reutilizan los mismos catálogos (`TextChoices`)
    que ya utiliza el perfil del usuario, para mantener consistencia con el
    futuro sistema de recomendación.
    """

    class Intensity(models.TextChoices):
        BAJA = "BAJA", _("Baja")
        MODERADA = "MODERADA", _("Moderada")
        ALTA = "ALTA", _("Alta")

    name = models.CharField(_("nombre"), max_length=150)

    description = models.TextField(_("descripción"), blank=True)

    goal = models.CharField(
        _("objetivo"),
        max_length=30,
        choices=Profile.Goal.choices,
    )

    level = models.CharField(
        _("nivel"),
        max_length=20,
        choices=Profile.ExperienceLevel.choices,
    )

    intensity = models.CharField(
        _("intensidad"),
        max_length=10,
        choices=Intensity.choices,
        default=Intensity.MODERADA,
    )

    days_per_week = models.PositiveSmallIntegerField(
        _("días por semana"),
        default=3,
        validators=[MinValueValidator(1), MaxValueValidator(7)],
        help_text=_("Frecuencia semanal sugerida (entre 1 y 7 días)."),
    )

    estimated_duration_minutes = models.PositiveIntegerField(
        _("duración estimada por sesión (minutos)"),
        default=45,
        validators=[MinValueValidator(1)],
        help_text=_("Minutos aproximados que dura una sesión de la rutina."),
    )

    exercises = models.ManyToManyField(
        Exercise,
        through="RoutineExercise",
        related_name="routines",
        verbose_name=_("ejercicios"),
    )

    is_active = models.BooleanField(
        _("activa"),
        default=True,
        help_text=_("Las rutinas inactivas no se muestran a los usuarios."),
    )

    created_at = models.DateTimeField(_("creada"), auto_now_add=True)
    updated_at = models.DateTimeField(_("actualizada"), auto_now=True)

    class Meta:
        verbose_name = _("rutina")
        verbose_name_plural = _("rutinas")
        ordering = ("name",)

    def __str__(self):
        return self.name

    def clean(self):
        super().clean()
        if self.days_per_week is not None and not (1 <= self.days_per_week <= 7):
            raise ValidationError(
                {
                    "days_per_week": _(
                        "Los días por semana deben estar entre 1 y 7."
                    )
                }
            )
        if (
            self.estimated_duration_minutes is not None
            and self.estimated_duration_minutes <= 0
        ):
            raise ValidationError(
                {
                    "estimated_duration_minutes": _(
                        "La duración estimada debe ser mayor que cero."
                    )
                }
            )

    @property
    def exercises_count(self) -> int:
        return self.exercises.count()

    @property
    def sessions_count(self) -> int:
        return self.sessions.count()


class RoutineSession(models.Model):
    """Sesión o día de entrenamiento dentro de una rutina.

    Estructuras como Push/Pull/Legs, Upper/Lower o Arnold Split agrupan
    sus ejercicios en varias sesiones (p. ej. "Push", "Pull", "Legs").
    Las rutinas simples pueden no tener sesiones y mostrar su lista plana
    de ejercicios.
    """

    routine = models.ForeignKey(
        Routine,
        on_delete=models.CASCADE,
        related_name="sessions",
        verbose_name=_("rutina"),
    )

    name = models.CharField(_("nombre"), max_length=100)

    order = models.PositiveIntegerField(_("orden"), default=1)

    class Meta:
        verbose_name = _("sesión de rutina")
        verbose_name_plural = _("sesiones de rutina")
        ordering = ("order",)
        constraints = [
            models.UniqueConstraint(
                fields=("routine", "name"),
                name="unique_session_name_per_routine",
                violation_error_message=_(
                    "Esta rutina ya tiene una sesión con ese nombre."
                ),
            ),
        ]

    def __str__(self):
        return f"{self.name} — {self.routine}"


class RoutineExercise(models.Model):
    """Tabla intermedia Rutina <-> Ejercicio.

    Almacena la configuración específica de un ejercicio dentro de una
    rutina: orden, series, repeticiones (ejercicios por conteo) o duración
    en segundos (ejercicios por tiempo, p. ej. plancha), y descanso.
    """

    routine = models.ForeignKey(
        Routine,
        on_delete=models.CASCADE,
        related_name="routine_exercises",
        verbose_name=_("rutina"),
    )

    exercise = models.ForeignKey(
        Exercise,
        on_delete=models.PROTECT,
        related_name="routine_exercises",
        verbose_name=_("ejercicio"),
    )

    session = models.ForeignKey(
        RoutineSession,
        on_delete=models.CASCADE,
        related_name="routine_exercises",
        verbose_name=_("sesión"),
        null=True,
        blank=True,
        help_text=_(
            "Sesión de la rutina a la que pertenece este ejercicio. "
            "Vacío para rutinas de una sola sesión."
        ),
    )

    order = models.PositiveIntegerField(_("orden"), default=1)

    sets = models.PositiveIntegerField(
        _("series"),
        validators=[MinValueValidator(1)],
    )

    repetitions = models.PositiveIntegerField(
        _("repeticiones"),
        null=True,
        blank=True,
        help_text=_("Déjalo vacío si el ejercicio se mide por tiempo."),
    )

    duration_seconds = models.PositiveIntegerField(
        _("duración (segundos)"),
        null=True,
        blank=True,
        help_text=_("Déjalo vacío si el ejercicio se mide por repeticiones."),
    )

    rest_seconds = models.PositiveIntegerField(_("descanso (segundos)"), default=0)

    class Meta:
        verbose_name = _("ejercicio de rutina")
        verbose_name_plural = _("ejercicios de rutina")
        ordering = ("order",)
        constraints = [
            models.UniqueConstraint(
                fields=("routine", "exercise"),
                name="unique_exercise_per_routine",
                violation_error_message=_(
                    "Este ejercicio ya está en la rutina."
                ),
            ),
        ]

    def __str__(self):
        return f"{self.exercise} en {self.routine}"

    def clean(self):
        super().clean()
        if (
            self.session is not None
            and self.session.routine_id != self.routine_id
        ):
            raise ValidationError(
                {
                    "session": _(
                        "La sesión debe pertenecer a la misma rutina."
                    )
                }
            )
        if self.repetitions is None and self.duration_seconds is None:
            raise ValidationError(
                {
                    "repetitions": _(
                        "Indica repeticiones o duración (al menos una)."
                    )
                }
            )
        if self.repetitions is not None and self.duration_seconds is not None:
            raise ValidationError(
                _("Usa repeticiones O duración, no ambas.")
            )


class UserRoutineAssignment(models.Model):
    """Asignación de una rutina a un usuario por parte de un entrenador.

    Registra qué rutina fue asignada y realizada por cada usuario,
    permitiendo mantener un historial completo de rutinas a lo largo del tiempo.
    """

    class Status(models.TextChoices):
        ASIGNADA = "ASIGNADA", _("Asignada")
        EN_PROGRESO = "EN_PROGRESO", _("En progreso")
        FINALIZADA = "FINALIZADA", _("Finalizada")
        CANCELADA = "CANCELADA", _("Cancelada")

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="routine_assignments",
        verbose_name=_("usuario"),
        help_text=_("Usuario al que se asigna la rutina."),
    )

    routine = models.ForeignKey(
        Routine,
        on_delete=models.PROTECT,
        related_name="user_assignments",
        verbose_name=_("rutina"),
        help_text=_("Rutina asignada al usuario."),
    )

    trainer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="assignments_created",
        verbose_name=_("entrenador"),
        help_text=_("Entrenador que realiza la asignación."),
    )

    start_date = models.DateField(
        _("fecha de inicio"),
        help_text=_("Fecha en la que comienza la asignación."),
    )

    end_date = models.DateField(
        _("fecha de finalización"),
        null=True,
        blank=True,
        help_text=_("Fecha en la que finaliza la asignación (opcional)."),
    )

    status = models.CharField(
        _("estado"),
        max_length=20,
        choices=Status.choices,
        default=Status.ASIGNADA,
    )

    initial_evaluation = models.ForeignKey(
        "assessments.PhysicalEvaluation",
        on_delete=models.SET_NULL,
        related_name="assignments_as_initial",
        null=True,
        blank=True,
        verbose_name=_("evaluación inicial"),
        help_text=_("Evaluación física inicial del usuario (opcional)."),
    )

    final_evaluation = models.ForeignKey(
        "assessments.PhysicalEvaluation",
        on_delete=models.SET_NULL,
        related_name="assignments_as_final",
        null=True,
        blank=True,
        verbose_name=_("evaluación final"),
        help_text=_("Evaluación física final del usuario (opcional)."),
    )

    planned_sessions = models.PositiveIntegerField(
        _("sesiones planificadas"),
        default=0,
        validators=[MinValueValidator(0)],
        help_text=_("Número total de sesiones planificadas."),
    )

    completed_sessions = models.PositiveIntegerField(
        _("sesiones realizadas"),
        default=0,
        validators=[MinValueValidator(0)],
        help_text=_("Número de sesiones que el usuario ha realizado."),
    )

    observations = models.TextField(
        _("observaciones"),
        blank=True,
        help_text=_("Notas adicionales del entrenador."),
    )

    created_at = models.DateTimeField(_("creada"), auto_now_add=True)
    updated_at = models.DateTimeField(_("actualizada"), auto_now=True)

    class Meta:
        verbose_name = _("asignación de rutina")
        verbose_name_plural = _("asignaciones de rutinas")
        ordering = ("-start_date", "-created_at")

    def __str__(self):
        return f"{self.routine.name} → {self.user} ({self.get_status_display()})"

    def clean(self):
        super().clean()
        # Validar que el entrenador tenga rol ENTRENADOR
        if self.trainer_id and getattr(self.trainer, "role", None) != "ENTRENADOR":
            raise ValidationError(
                {"trainer": _("El asignador debe ser un entrenador.")}
            )
        # Validar que el usuario asignado sea válido
        if self.user_id and getattr(self.user, "role", None) != "USUARIO":
            raise ValidationError(
                {"user": _("El usuario asignado debe tener rol USUARIO.")}
            )
        # Validar fecha final no anterior a fecha inicial
        if self.end_date and self.start_date and self.end_date < self.start_date:
            raise ValidationError(
                {"end_date": _("La fecha final no puede ser anterior a la fecha inicial.")}
            )
        # Validar evaluación inicial pertenece al usuario y es tipo INICIAL
        if self.initial_evaluation_id:
            if self.initial_evaluation.user_id != self.user_id:
                raise ValidationError(
                    {"initial_evaluation": _("La evaluación inicial debe pertenecer al mismo usuario.")}
                )
            if self.initial_evaluation.evaluation_type != "INICIAL":
                raise ValidationError(
                    {"initial_evaluation": _("La evaluación inicial debe ser de tipo INICIAL.")}
                )
        # Validar evaluación final pertenece al usuario y es tipo FINAL
        if self.final_evaluation_id:
            if self.final_evaluation.user_id != self.user_id:
                raise ValidationError(
                    {"final_evaluation": _("La evaluación final debe pertenecer al mismo usuario.")}
                )
            if self.final_evaluation.evaluation_type != "FINAL":
                raise ValidationError(
                    {"final_evaluation": _("La evaluación final debe ser de tipo FINAL.")}
                )
        # Validar sesiones realizadas no superen planificadas
        if (
            self.planned_sessions is not None
            and self.planned_sessions > 0
            and self.completed_sessions > self.planned_sessions
        ):
            raise ValidationError(
                {"completed_sessions": _("Las sesiones realizadas no pueden superar las planificadas.")}
            )

    @property
    def adherence_percentage(self):
        """Porcentaje de adherencia calculado dinámicamente.

        Devuelve None si no hay sesiones planificadas (evita división por cero).
        """
        if self.planned_sessions is None or self.planned_sessions == 0:
            return None
        return round((self.completed_sessions / self.planned_sessions) * 100, 1)

    @property
    def is_active(self) -> bool:
        """True si la asignación está en progreso o recién asignada."""
        return self.status in (self.Status.ASIGNADA, self.Status.EN_PROGRESO)
