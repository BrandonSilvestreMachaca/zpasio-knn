"""URLs del módulo de gestión de ejercicios."""

from django.urls import path

from . import views

app_name = "exercises"

urlpatterns = [
    path("", views.exercise_list_view, name="list"),
    path("nuevo/", views.exercise_create_view, name="create"),
    path("<int:pk>/", views.exercise_detail_view, name="detail"),
    path("<int:pk>/editar/", views.exercise_update_view, name="update"),
    path("<int:pk>/eliminar/", views.exercise_delete_view, name="delete"),
]
