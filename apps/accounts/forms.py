"""Formularios de autenticación y registro de Zpasio."""

from django import forms
from django.contrib.auth import authenticate
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.utils.translation import gettext_lazy as _

from .models import CustomUser


class EmailOrUsernameAuthenticationForm(AuthenticationForm):
    """Inicio de sesión usando el `username` o el `email` del usuario.

    Django, por defecto, solo compara contra ``USERNAME_FIELD`` (``username``).
    Aquí se acepta cualquiera de los dos identificadores, usando exclusivamente
    ``authenticate()`` nativo de Django para validar credenciales (nunca
    contraseña en texto plano).
    """

    username = forms.CharField(
        label=_("Usuario o correo electrónico"),
        widget=forms.TextInput(attrs={"autofocus": True, "autocomplete": "username"}),
    )

    def clean(self):
        username = self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")

        if username is not None and password:
            self.user_cache = authenticate(
                self.request,
                username=username,
                password=password,
            )
            if self.user_cache is None:
                # Reintento identificando por email (case-insensitive).
                user = CustomUser.objects.filter(email__iexact=username).first()
                if user is not None:
                    self.user_cache = authenticate(
                        self.request,
                        username=user.username,
                        password=password,
                    )
            if self.user_cache is None:
                raise self.get_invalid_login_error()
            else:
                self.confirm_login_allowed(self.user_cache)

        return self.cleaned_data


class RegisterForm(UserCreationForm):
    """Formulario de registro de nuevos usuarios (cliente)."""

    email = forms.EmailField(
        label=_("Correo electrónico"),
        required=True,
    )

    first_name = forms.CharField(
        label=_("Nombre"),
        max_length=150,
        required=True,
    )

    last_name = forms.CharField(
        label=_("Apellido"),
        max_length=150,
        required=True,
    )

    class Meta:
        model = CustomUser
        fields = (
            "username",
            "first_name",
            "last_name",
            "email",
            "password1",
            "password2",
        )

    def clean_email(self):
        email = self.cleaned_data.get("email")
        if email and CustomUser.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError(_("Ya existe un usuario con este correo electrónico."))
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"].lower()
        user.role = CustomUser.Role.USUARIO
        if commit:
            user.save()
        return user