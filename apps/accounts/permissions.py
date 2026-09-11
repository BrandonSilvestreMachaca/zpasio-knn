"""Decoradores y mixins para control de acceso por rol en Zpasio."""

from functools import wraps

from django.contrib.auth.mixins import UserPassesTestMixin
from django.http import HttpResponseForbidden


def role_required(*roles):
    """Restringe una vista a usuarios cuyo rol esté en `roles`."""

    def decorator(view_func):
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            if not request.user.is_authenticated:
                from django.contrib.auth.views import redirect_to_login

                return redirect_to_login(
                    request.get_full_path(),
                    "accounts:login",
                    "_next",
                )
            if request.user.role not in roles:
                return HttpResponseForbidden()
            return view_func(request, *args, **kwargs)

        return _wrapped

    return decorator


entrenador_required = role_required("ENTRENADOR")
usuario_required = role_required("USUARIO")


class RoleRequiredMixin(UserPassesTestMixin):
    """Mixin para vistas basadas en clase que exige un conjunto de roles."""

    allowed_roles = []

    def test_func(self):
        return self.request.user.is_authenticated and self.request.user.role in self.allowed_roles

    def handle_no_permission(self):
        if not self.request.user.is_authenticated:
            from django.contrib.auth.views import redirect_to_login

            return redirect_to_login(self.request.get_full_path(), "accounts:login")
        return HttpResponseForbidden()


class EntrenadorRequiredMixin(RoleRequiredMixin):
    allowed_roles = ["ENTRENADOR"]