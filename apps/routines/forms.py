"""Formularios del módulo de gestión de rutinas."""

from django import forms
from django.conf import settings
from django.forms import inlineformset_factory
from django.utils.translation import gettext_lazy as _

from apps.exercises.models import Exercise
from apps.profiles.models import Profile

from .models import Routine, RoutineExercise, UserRoutineAssignment


class RoutineForm(forms.ModelForm):
    """Datos generales de la rutina (solo entrenador)."""

    class Meta:
        model = Routine
        fields = (
            "name",
            "description",
            "goal",
            "level",
            "intensity",
            "days_per_week",
            "estimated_duration_minutes",
            "is_active",
        )
        widgets = {
            "description": forms.Textarea(attrs={"rows": 4}),
        }

    def clean_name(self):
        name = (self.cleaned_data.get("name") or "").strip()
        if not name:
            raise forms.ValidationError(_("El nombre de la rutina es obligatorio."))
        return name

    def clean_days_per_week(self):
        days = self.cleaned_data.get("days_per_week")
        if days is None:
            raise forms.ValidationError(
                _("Los días por semana son obligatorios.")
            )
        if not 1 <= days <= 7:
            raise forms.ValidationError(
                _("Los días por semana deben estar entre 1 y 7.")
            )
        return days

    def clean_estimated_duration_minutes(self):
        duration = self.cleaned_data.get("estimated_duration_minutes")
        if duration is None:
            raise forms.ValidationError(
                _("La duración estimada por sesión es obligatoria.")
            )
        if duration <= 0:
            raise forms.ValidationError(
                _("La duración estimada debe ser mayor que cero.")
            )
        return duration


class BaseRoutineExerciseFormSet(forms.BaseInlineFormSet):
    """Formset que relaciona ejercicios existentes con la rutina."""

    def clean(self):
        super().clean()
        if any(self.errors):
            return

        exercises = []
        valid_forms = 0
        for form in self.forms:
            if not hasattr(form, "cleaned_data") or form.cleaned_data.get("DELETE"):
                continue
            exercise = form.cleaned_data.get("exercise")
            if exercise is None:
                continue
            valid_forms += 1
            if exercise in exercises:
                form.add_error(
                    "exercise",
                    _("Este ejercicio ya está agregado a la rutina."),
                )
            exercises.append(exercise)

        if valid_forms == 0:
            raise forms.ValidationError(
                _("La rutina debe incluir al menos un ejercicio.")
            )


RoutineExerciseFormSet = inlineformset_factory(
    Routine,
    RoutineExercise,
    formset=BaseRoutineExerciseFormSet,
    fields=("exercise", "order", "sets", "repetitions", "duration_seconds",
            "rest_seconds"),
    extra=3,
    can_delete=True,
    min_num=0,
    validate_min=False,
)

# Solo se ofrecen ejercicios activos del catálogo existente para los nuevos
# renglones; al editar se conservan los ejercicios ya vinculados aunque hayan
# sido desactivados del catálogo.
def limit_exercise_queryset(formset):
    active_exercises = Exercise.objects.filter(is_active=True)
    for form in formset.forms:
        qs = active_exercises
        if form.instance.pk:
            qs = active_exercises | Exercise.objects.filter(
                pk=form.instance.exercise_id
            )
        form.fields["exercise"].queryset = qs.distinct()
    return formset


class UserRoutineAssignmentForm(forms.ModelForm):
    """Formulario para asignar una rutina a un usuario (solo entrenador)."""

    class Meta:
        model = UserRoutineAssignment
        fields = (
            "routine",
            "start_date",
            "end_date",
            "status",
            "initial_evaluation",
            "final_evaluation",
            "planned_sessions",
            "completed_sessions",
            "observations",
        )
        widgets = {
            "start_date": forms.DateInput(attrs={"type": "date"}),
            "end_date": forms.DateInput(attrs={"type": "date"}),
            "observations": forms.Textarea(attrs={"rows": 4}),
        }

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop("user", None)
        self.trainer = kwargs.pop("trainer", None)
        super().__init__(*args, **kwargs)

        # Filtrar rutinas activas
        self.fields["routine"].queryset = Routine.objects.filter(is_active=True)

        # Filtrar evaluaciones del usuario si se proporciona
        if self.user:
            self.fields["initial_evaluation"].queryset = self.user.physical_evaluations.filter(
                evaluation_type="INICIAL"
            )
            self.fields["final_evaluation"].queryset = self.user.physical_evaluations.filter(
                evaluation_type="FINAL"
            )
        else:
            self.fields["initial_evaluation"].queryset = self.fields["initial_evaluation"].queryset.none()
            self.fields["final_evaluation"].queryset = self.fields["final_evaluation"].queryset.none()

        # Hacer opcionales los campos de evaluación y fechas
        self.fields["end_date"].required = False
        self.fields["initial_evaluation"].required = False
        self.fields["final_evaluation"].required = False
        self.fields["observations"].required = False

    def clean(self):
        cleaned_data = super().clean()
        # Validaciones adicionales a nivel de formulario
        end_date = cleaned_data.get("end_date")
        start_date = cleaned_data.get("start_date")
        if end_date and start_date and end_date < start_date:
            self.add_error(
                "end_date",
                _("La fecha final no puede ser anterior a la fecha inicial."),
            )

        planned = cleaned_data.get("planned_sessions")
        completed = cleaned_data.get("completed_sessions")
        if planned is not None and planned > 0 and completed is not None and completed > planned:
            self.add_error(
                "completed_sessions",
                _("Las sesiones realizadas no pueden superar las planificadas."),
            )

        return cleaned_data
