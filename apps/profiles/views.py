"""Vistas de perfil e IMC de Zpasio."""

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext_lazy as _

from apps.accounts.permissions import entrenador_required

from .forms import PhysicalDataForm, ProfileForm
from .models import PhysicalData, Profile

User = get_user_model()


@login_required
def profile_detail_view(request):
    """Muestra el perfil del usuario autenticado (con datos físicos e IMC)."""
    profile, _ = Profile.objects.get_or_create(user=request.user)
    physical_data = getattr(request.user, "physical_data", None)

    context = {
        "profile": profile,
        "physical_data": physical_data,
    }
    return render(request, "profiles/profile_detail.html", context)


@login_required
@transaction.atomic
def profile_edit_view(request):
    """Edición de datos personales del usuario autenticado."""
    profile, _created = Profile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, _("Perfil actualizado correctamente."))
            return redirect("profiles:detail")
    else:
        form = ProfileForm(instance=profile)

    return render(request, "profiles/profile_edit.html", {"form": form})


@login_required
@transaction.atomic
def physical_data_edit_view(request):
    """Edición de datos físicos del usuario autenticado."""
    physical_data = getattr(request.user, "physical_data", None)

    if request.method == "POST":
        form = PhysicalDataForm(request.POST, instance=physical_data)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.user = request.user
            obj.save()
            messages.success(request, _("Datos físicos actualizados correctamente."))
            return redirect("profiles:detail")
    else:
        form = PhysicalDataForm(instance=physical_data)

    return render(request, "profiles/physical_data_edit.html", {"form": form})


@entrenador_required
def entrenador_client_list_view(request):
    """Lista de usuarios/clientes visible para el entrenador."""
    clients = User.objects.filter(role=User.Role.USUARIO).order_by("username")
    return render(
        request,
        "profiles/client_list.html",
        {"clients": clients},
    )


@entrenador_required
def entrenador_client_detail_view(request, username):
    """Detalle del perfil de un cliente (solo para el entrenador)."""
    client = get_object_or_404(User, username=username, role=User.Role.USUARIO)
    profile = getattr(client, "profile", None)
    physical_data = getattr(client, "physical_data", None)

    context = {
        "client": client,
        "profile": profile,
        "physical_data": physical_data,
    }
    return render(request, "profiles/client_detail.html", context)