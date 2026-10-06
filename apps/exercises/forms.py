"""Formularios del módulo de gestión de ejercicios."""

from django import forms
from django.utils.translation import gettext_lazy as _

from .models import Exercise


class ExerciseForm(forms.ModelForm):
    """Formulario de creación y edición de ejercicios (solo entrenador)."""

    class Meta:
        model = Exercise
        fields = ("name", "description", "muscle_group", "difficulty", "is_active")
        widgets = {
            "description": forms.Textarea(attrs={"rows": 4}),
        }

    def clean_name(self):
        name = (self.cleaned_data.get("name") or "").strip()
        if not name:
            raise forms.ValidationError(_("El nombre del ejercicio es obligatorio."))
        return name
