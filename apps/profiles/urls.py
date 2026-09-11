"""URLs de la aplicación de perfiles."""

from django.urls import path

from . import views

app_name = "profiles"

urlpatterns = [
    path("", views.profile_detail_view, name="detail"),
    path("editar/", views.profile_edit_view, name="edit"),
    path("datos-fisicos/", views.physical_data_edit_view, name="physical_data_edit"),
    path("clientes/", views.entrenador_client_list_view, name="client_list"),
    path(
        "clientes/<str:username>/",
        views.entrenador_client_detail_view,
        name="client_detail",
    ),
]