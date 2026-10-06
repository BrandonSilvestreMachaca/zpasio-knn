"""Formularios de perfil y datos físicos de Zpasio."""

from django import forms
from django.utils.translation import gettext_lazy as _

from .models import PhysicalData, Profile


class ProfileForm(forms.ModelForm):
    """Formulario de edición de datos personales y de entrenamiento."""

    class Meta:
        model = Profile
        fields = (
            "phone",
            "birth_date",
            "gender",
            "goal",
            "experience_level",
            "available_days_per_week",
            "session_time_available_minutes",
            "previous_training_days_per_week",
        )
        widgets = {
            "birth_date": forms.DateInput(
                attrs={"type": "date"},
                format="%Y-%m-%d",
            ),
        }

    def clean_available_days_per_week(self):
        days = self.cleaned_data.get("available_days_per_week")
        if days is not None and not 1 <= days <= 7:
            raise forms.ValidationError(
                _("Los días disponibles deben estar entre 1 y 7.")
            )
        return days

    def clean_session_time_available_minutes(self):
        minutes = self.cleaned_data.get("session_time_available_minutes")
        if minutes is not None and minutes <= 0:
            raise forms.ValidationError(
                _("El tiempo disponible por sesión debe ser mayor que cero.")
            )
        return minutes

    def clean_previous_training_days_per_week(self):
        days = self.cleaned_data.get("previous_training_days_per_week")
        if days is not None and not 0 <= days <= 7:
            raise forms.ValidationError(
                _("La frecuencia previa debe estar entre 0 y 7 días.")
            )
        return days


class PhysicalDataForm(forms.ModelForm):
    """Formulario de edición de datos físicos medibles.

    El IMC no se incluye como campo editable: se calcula automáticamente.
    """

    class Meta:
        model = PhysicalData
        fields = (
            "weight_kg",
            "height_cm",
            "waist_circumference_cm",
            "body_fat_percentage",
        )

    def clean_weight_kg(self):
        weight = self.cleaned_data.get("weight_kg")
        if weight is not None and weight <= 0:
            raise forms.ValidationError(_("El peso debe ser un valor positivo."))
        return weight

    def clean_height_cm(self):
        height = self.cleaned_data.get("height_cm")
        if height is not None and height <= 0:
            raise forms.ValidationError(_("La altura debe ser un valor positivo."))
        return height

    def clean_waist_circumference_cm(self):
        waist = self.cleaned_data.get("waist_circumference_cm")
        if waist is not None and waist <= 0:
            raise forms.ValidationError(
                _("La circunferencia de cintura debe ser un valor positivo.")
            )
        return waist

    def clean_body_fat_percentage(self):
        fat = self.cleaned_data.get("body_fat_percentage")
        if fat is not None and not 0 <= fat <= 100:
            raise forms.ValidationError(
                _("El porcentaje de grasa debe estar entre 0 y 100.")
            )
        return fat