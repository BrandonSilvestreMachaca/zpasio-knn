"""Vistas del módulo de evaluaciones físicas de Zpasio.

Política de permisos (aplicada en backend, no solo ocultando botones):
- ENTRENADOR: registra evaluaciones de cualquier usuario (rol USUARIO)
  y consulta todas las evaluaciones.
- USUARIO: consulta únicamente sus propias evaluaciones.
"""

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext_lazy as _

from apps.accounts.permissions import entrenador_required

from .forms import EvaluationForm, EvaluationResultFormSet
from .models import PhysicalEvaluation

User = get_user_model()


@login_required
def my_evaluations_view(request):
    """Historial de evaluaciones del usuario autenticado."""
    evaluations = (
        PhysicalEvaluation.objects.filter(user=request.user)
        .select_related("trainer")
        .prefetch_related("results")
    )
    return render(
        request,
        "assessments/evaluation_list.html",
        {"evaluations": evaluations},
    )


@entrenador_required
def evaluation_create_view(request, username):
    """Registra una evaluación física para un cliente (rol USUARIO)."""
    client = get_object_or_404(User, username=username, role=User.Role.USUARIO)
    if request.method == "POST":
        form = EvaluationForm(request.POST)
        formset = EvaluationResultFormSet(
            request.POST, instance=PhysicalEvaluation()
        )
        if form.is_valid() and formset.is_valid():
            with transaction.atomic():
                evaluation = form.save(commit=False)
                evaluation.user = client
                evaluation.trainer = request.user
                evaluation.save()
                formset.instance = evaluation
                formset.save()
            messages.success(
                request, _("Evaluación registrada correctamente.")
            )
            return redirect("profiles:client_detail", username=client.username)
    else:
        form = EvaluationForm()
        formset = EvaluationResultFormSet(instance=PhysicalEvaluation())
    return render(
        request,
        "assessments/evaluation_form.html",
        {"form": form, "formset": formset, "client": client},
    )


@login_required
def evaluation_detail_view(request, pk):
    """Detalle de una evaluación: entrenador o el propio evaluado."""
    evaluation = get_object_or_404(
        PhysicalEvaluation.objects.select_related("user", "trainer")
        .prefetch_related("results"),
        pk=pk,
    )
    if (
        not request.user.is_entrenador
        and evaluation.user_id != request.user.pk
    ):
        return HttpResponseForbidden()
    return render(
        request,
        "assessments/evaluation_detail.html",
        {"evaluation": evaluation},
    )
