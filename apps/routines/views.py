"""Vistas del módulo de gestión de rutinas de Zpasio.

Política de permisos:
- ENTRENADOR: acceso completo (listar, ver, crear, editar, eliminar).
- USUARIO: acceso de solo lectura (listar y ver rutinas activas).
"""

import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext_lazy as _

from apps.accounts.permissions import entrenador_required

from .forms import (
    RoutineExerciseFormSet,
    RoutineForm,
    UserRoutineAssignmentForm,
    limit_exercise_queryset,
)
from .models import Routine, UserRoutineAssignment

logger = logging.getLogger(__name__)


def _log_invalid_form(request, form, formset):
    """Registra errores de validación para diagnóstico (nivel DEBUG)."""
    logger.debug(
        "Formulario de rutina inválido (user=%s). form.errors=%s "
        "formset.errors=%s formset.non_form_errors=%s",
        request.user, form.errors.as_data(), formset.errors,
        formset.non_form_errors(),
    )


@login_required
def routine_list_view(request):
    """Listado de rutinas (entrenador las ve todas, usuario solo activas)."""
    if request.user.is_entrenador:
        routines = Routine.objects.prefetch_related("exercises")
    else:
        routines = Routine.objects.filter(is_active=True).prefetch_related(
            "exercises"
        )
    return render(request, "routines/routine_list.html", {"routines": routines})


@login_required
def routine_detail_view(request, pk):
    """Detalle de una rutina con sus ejercicios agrupados por sesión.

    Las rutinas con sesiones (PPL, Arnold Split...) se muestran
    agrupadas: cada sesión con su nombre y su lista ordenada de
    ejercicios. Las rutinas sin sesiones muestran la lista plana.
    """
    routine = get_object_or_404(
        Routine.objects.prefetch_related(
            "sessions__routine_exercises__exercise",
            "routine_exercises__exercise",
        ),
        pk=pk,
    )
    if not routine.is_active and not request.user.is_entrenador:
        from django.http import Http404

        raise Http404()

    session_blocks = [
        (session, session.routine_exercises.all())
        for session in routine.sessions.all()
    ]
    loose_items = [
        item
        for item in routine.routine_exercises.all()
        if item.session_id is None
    ]
    return render(
        request,
        "routines/routine_detail.html",
        {
            "routine": routine,
            "session_blocks": session_blocks,
            "loose_items": loose_items,
        },
    )


@entrenador_required
def routine_create_view(request):
    """Creación de una rutina con sus ejercicios (solo entrenador)."""
    if request.method == "POST":
        form = RoutineForm(request.POST)
        formset = RoutineExerciseFormSet(request.POST, instance=Routine())
        if form.is_valid() and formset.is_valid():
            with transaction.atomic():
                routine = form.save()
                formset.instance = routine
                formset.save()
            messages.success(request, _("Rutina creada correctamente."))
            return redirect("routines:detail", pk=routine.pk)
        _log_invalid_form(request, form, formset)
    else:
        form = RoutineForm()
        formset = RoutineExerciseFormSet(instance=Routine())
    limit_exercise_queryset(formset)
    return render(
        request,
        "routines/routine_form.html",
        {"form": form, "formset": formset, "title": _("Nueva rutina")},
    )


@entrenador_required
def routine_update_view(request, pk):
    """Edición de una rutina y sus ejercicios (solo entrenador)."""
    routine = get_object_or_404(Routine, pk=pk)
    if request.method == "POST":
        form = RoutineForm(request.POST, instance=routine)
        formset = RoutineExerciseFormSet(request.POST, instance=routine)
        if form.is_valid() and formset.is_valid():
            with transaction.atomic():
                form.save()
                formset.save()
            messages.success(request, _("Rutina actualizada correctamente."))
            return redirect("routines:detail", pk=routine.pk)
        _log_invalid_form(request, form, formset)
    else:
        form = RoutineForm(instance=routine)
        formset = RoutineExerciseFormSet(instance=routine)
    limit_exercise_queryset(formset)
    return render(
        request,
        "routines/routine_form.html",
        {
            "form": form,
            "formset": formset,
            "title": _("Editar rutina"),
            "routine": routine,
        },
    )


@entrenador_required
def routine_delete_view(request, pk):
    """Eliminación de una rutina mediante desactivación lógica.

    Igual que en el módulo de ejercicios, no se borra físicamente el
    registro: se marca `is_active=False` para preservar la trazabilidad.
    Requiere confirmación explícita.
    """
    routine = get_object_or_404(Routine, pk=pk)
    if request.method == "POST":
        if "confirm" in request.POST:
            routine.is_active = False
            routine.save(update_fields=["is_active", "updated_at"])
            messages.success(request, _("Rutina eliminada correctamente."))
            return redirect("routines:list")
        return redirect("routines:detail", pk=routine.pk)
    return render(
        request,
        "routines/routine_confirm_delete.html",
        {"routine": routine},
    )


# =============================================================================
# Asignaciones de rutina a usuarios (UserRoutineAssignment)
# =============================================================================


@login_required
def assignment_list_view(request):
    """Listado de asignaciones de rutinas.

    - ENTRENADOR: ve todas las asignaciones.
    - USUARIO: ve solo sus propias asignaciones.
    """
    if request.user.is_entrenador:
        assignments = UserRoutineAssignment.objects.select_related(
            "user", "routine", "trainer", "initial_evaluation", "final_evaluation"
        ).order_by("-start_date")
    else:
        assignments = UserRoutineAssignment.objects.filter(
            user=request.user
        ).select_related(
            "routine", "trainer", "initial_evaluation", "final_evaluation"
        ).order_by("-start_date")

    return render(
        request,
        "routines/assignment_list.html",
        {"assignments": assignments},
    )


@entrenador_required
def assignment_create_view(request, username):
    """Crea una asignación de rutina para un usuario (solo entrenador).

    El usuario se determina por el parámetro username en la URL.
    Opcionalmente se puede pre-seleccionar una rutina vía query parameter ?routine=<pk>.
    """
    from django.contrib.auth import get_user_model

    User = get_user_model()
    client = get_object_or_404(User, username=username, role=User.Role.USUARIO)

    # Pre-seleccionar rutina si viene en query parameter
    initial = {}
    routine_pk = request.GET.get("routine")
    if routine_pk:
        from .models import Routine
        routine = Routine.objects.filter(pk=routine_pk, is_active=True).first()
        if routine:
            initial["routine"] = routine.pk

    if request.method == "POST":
        form = UserRoutineAssignmentForm(
            request.POST, user=client, trainer=request.user
        )
        if form.is_valid():
            with transaction.atomic():
                form.save()
            messages.success(request, _("Rutina asignada correctamente."))
            return redirect("profiles:client_detail", username=client.username)
    else:
        form = UserRoutineAssignmentForm(user=client, trainer=request.user, initial=initial)

    return render(
        request,
        "routines/assignment_form.html",
        {"form": form, "client": client, "title": _("Asignar rutina")},
    )


@login_required
def assignment_detail_view(request, pk):
    """Detalle de una asignación de rutina.

    - ENTRENADOR: puede ver cualquier asignación.
    - USUARIO: solo puede ver sus propias asignaciones.
    """
    assignment = get_object_or_404(
        UserRoutineAssignment.objects.select_related(
            "user", "routine", "trainer", "initial_evaluation", "final_evaluation"
        ).prefetch_related("routine__sessions__routine_exercises__exercise"),
        pk=pk,
    )

    # Control de permisos
    if not request.user.is_entrenador and assignment.user_id != request.user.pk:
        from django.http import HttpResponseForbidden

        return HttpResponseForbidden()

    return render(
        request,
        "routines/assignment_detail.html",
        {"assignment": assignment},
    )


@entrenador_required
def assignment_update_view(request, pk):
    """Actualiza una asignación existente (solo entrenador)."""
    assignment = get_object_or_404(UserRoutineAssignment, pk=pk)

    if request.method == "POST":
        form = UserRoutineAssignmentForm(
            request.POST, instance=assignment, user=assignment.user, trainer=request.user
        )
        if form.is_valid():
            with transaction.atomic():
                form.save()
            messages.success(request, _("Asignación actualizada correctamente."))
            return redirect("routines:assignment_detail", pk=assignment.pk)
    else:
        form = UserRoutineAssignmentForm(
            instance=assignment, user=assignment.user, trainer=request.user
        )

    return render(
        request,
        "routines/assignment_form.html",
        {
            "form": form,
            "assignment": assignment,
            "client": assignment.user,
            "title": _("Editar asignación"),
        },
    )
