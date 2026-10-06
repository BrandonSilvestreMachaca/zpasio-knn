"""URLs del módulo de evaluaciones físicas."""

from django.urls import path

from . import views

app_name = "assessments"

urlpatterns = [
    path("", views.my_evaluations_view, name="list"),
    path(
        "nueva/<str:username>/",
        views.evaluation_create_view,
        name="create",
    ),
    path("<int:pk>/", views.evaluation_detail_view, name="detail"),
]
