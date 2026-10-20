"""URLs de la aplicación de recomendaciones."""

from django.urls import path

from . import views

app_name = "recommendations"

urlpatterns = [
    path(
        "<str:username>/",
        views.recommendation_generate_view,
        name="generate",
    ),
]