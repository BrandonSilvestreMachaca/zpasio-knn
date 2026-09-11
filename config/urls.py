"""Configuración de URLs del proyecto Zpasio."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.accounts.urls")),
    path("perfil/", include("apps.profiles.urls")),
]