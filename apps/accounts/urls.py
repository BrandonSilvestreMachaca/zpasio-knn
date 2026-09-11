"""URLs de la aplicación de cuentas."""

from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("registro/", views.register_view, name="register"),
    path("login/", views.ZpasioLoginView.as_view(), name="login"),
    path("logout/", views.ZpasioLogoutView.as_view(), name="logout"),
]