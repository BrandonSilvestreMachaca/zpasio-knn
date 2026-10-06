"""URLs del módulo de gestión de rutinas."""

from django.urls import path

from . import views

app_name = "routines"

urlpatterns = [
    path("", views.routine_list_view, name="list"),
    path("nueva/", views.routine_create_view, name="create"),
    path("<int:pk>/", views.routine_detail_view, name="detail"),
    path("<int:pk>/editar/", views.routine_update_view, name="update"),
    path("<int:pk>/eliminar/", views.routine_delete_view, name="delete"),
    # Asignaciones de rutina a usuarios
    path("asignaciones/", views.assignment_list_view, name="assignment_list"),
    path(
        "asignaciones/nueva/<str:username>/",
        views.assignment_create_view,
        name="assignment_create",
    ),
    path(
        "asignaciones/<int:pk>/",
        views.assignment_detail_view,
        name="assignment_detail",
    ),
    path(
        "asignaciones/<int:pk>/editar/",
        views.assignment_update_view,
        name="assignment_update",
    ),
]
