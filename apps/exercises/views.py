"""Vistas del módulo de gestión de ejercicios de Zpasio.

Política de permisos:
- ENTRENADOR: acceso completo (listar, ver, crear, editar, eliminar).
- USUARIO: acceso de solo lectura (listar y ver detalle).
"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext_lazy as _

from apps.accounts.permissions import entrenador_required

from .forms import ExerciseForm
from .models import Exercise


@login_required
def exercise_list_view(request):
    """Listado del catálogo de ejercicios (entrenador y usuario)."""
    if request.user.is_entrenador:
        exercises = Exercise.objects.all()
    else:
        exercises = Exercise.objects.filter(is_active=True)
    return render(
        request,
        "exercises/exercise_list.html",
        {"exercises": exercises},
    )


@login_required
def exercise_detail_view(request, pk):
    """Detalle de un ejercicio (lectura para todos los roles)."""
    exercise = get_object_or_404(Exercise, pk=pk)
    return render(
        request,
        "exercises/exercise_detail.html",
        {"exercise": exercise},
    )


@entrenador_required
def exercise_create_view(request):
    """Creación de un ejercicio nuevo (solo entrenador)."""
    if request.method == "POST":
        form = ExerciseForm(request.POST)
        if form.is_valid():
            exercise = form.save()
            messages.success(request, _("Ejercicio creado correctamente."))
            return redirect("exercises:detail", pk=exercise.pk)
    else:
        form = ExerciseForm()
    return render(
        request,
        "exercises/exercise_form.html",
        {"form": form, "title": _("Nuevo ejercicio")},
    )


@entrenador_required
def exercise_update_view(request, pk):
    """Edición de un ejercicio existente (solo entrenador)."""
    exercise = get_object_or_404(Exercise, pk=pk)
    if request.method == "POST":
        form = ExerciseForm(request.POST, instance=exercise)
        if form.is_valid():
            form.save()
            messages.success(request, _("Ejercicio actualizado correctamente."))
            return redirect("exercises:detail", pk=exercise.pk)
    else:
        form = ExerciseForm(instance=exercise)
    return render(
        request,
        "exercises/exercise_form.html",
        {"form": form, "title": _("Editar ejercicio"), "exercise": exercise},
    )


@entrenador_required
def exercise_delete_view(request, pk):
    """Eliminación de un ejercicio mediante desactivación lógica.

    Decisión de diseño: en lugar de borrar físicamente el registro, se marca
    `is_active=False`. Así el ejercicio deja de mostrarse en el catálogo sin
    perder trazabilidad, lo que protegerá la integridad de las rutinas cuando
    se implemente el módulo de rutinas (FK -> Exercise).
    """
    exercise = get_object_or_404(Exercise, pk=pk)
    if request.method == "POST":
        if "confirm" in request.POST:
            exercise.is_active = False
            exercise.save(update_fields=["is_active"])
            messages.success(request, _("Ejercicio eliminado correctamente."))
            return redirect("exercises:list")
        # Cancelar
        return redirect("exercises:detail", pk=exercise.pk)
    return render(
        request,
        "exercises/exercise_confirm_delete.html",
        {"exercise": exercise},
    )
