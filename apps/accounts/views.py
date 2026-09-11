"""Vistas de autenticación, registro y páginas base de Zpasio."""

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.utils.translation import gettext_lazy as _

from .forms import EmailOrUsernameAuthenticationForm, RegisterForm


class ZpasioLoginView(LoginView):
    """Inicio de sesión con su propia plantilla."""

    template_name = "accounts/login.html"
    authentication_form = EmailOrUsernameAuthenticationForm
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy("profiles:detail")


def register_view(request):
    """Registro de nuevos usuarios (clientes)."""
    if request.user.is_authenticated:
        return redirect("profiles:detail")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, _("Cuenta creada correctamente. ¡Bienvenido!"))
            return redirect("profiles:detail")
    else:
        form = RegisterForm()

    return render(request, "accounts/register.html", {"form": form})


@login_required
def home_view(request):
    """Página de inicio tras iniciar sesión."""
    return redirect("profiles:detail")


class ZpasioLogoutView(LogoutView):
    """Cierre de sesión."""

    next_page = "accounts:login"