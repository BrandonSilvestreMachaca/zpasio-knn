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
        )
        widgets = {
            "birth_date": forms.DateInput(
                attrs={"type": "date"},
                format="%Y-%m-%d",
            ),
        }


class PhysicalDataForm(forms.ModelForm):
    """Formulario de edición de datos físicos medibles.

    El IMC no se incluye como campo editable: se calcula automáticamente.
    """

    class Meta:
        model = PhysicalData
        fields = (
            "weight_kg",
            "height_cm",
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