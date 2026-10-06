"""Formularios del módulo de evaluaciones físicas."""

from django import forms
from django.forms import inlineformset_factory
from django.utils.translation import gettext_lazy as _

from .models import EvaluationResult, PhysicalEvaluation


class EvaluationForm(forms.ModelForm):
    """Datos generales de la evaluación (solo entrenador)."""

    class Meta:
        model = PhysicalEvaluation
        fields = ("evaluation_type", "observations")
        widgets = {
            "observations": forms.Textarea(attrs={"rows": 4}),
        }


class BaseEvaluationResultFormSet(forms.BaseInlineFormSet):
    """Los resultados son pruebas objetivas con valor y unidad."""

    def clean(self):
        super().clean()
        if any(self.errors):
            return

        valid_results = [
            form
            for form in self.forms
            if hasattr(form, "cleaned_data")
            and not form.cleaned_data.get("DELETE")
            and form.cleaned_data.get("test_name")
        ]
        if not valid_results:
            raise forms.ValidationError(
                _("La evaluación debe incluir al menos una prueba.")
            )
        for form in valid_results:
            value = form.cleaned_data.get("value")
            if value is not None and value < 0:
                form.add_error(
                    "value",
                    _("El resultado no puede ser negativo."),
                )


EvaluationResultFormSet = inlineformset_factory(
    PhysicalEvaluation,
    EvaluationResult,
    formset=BaseEvaluationResultFormSet,
    fields=("test_name", "value", "unit", "notes"),
    extra=4,
    can_delete=False,
    min_num=1,
    validate_min=False,
)
