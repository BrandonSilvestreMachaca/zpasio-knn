"""Vistas de la aplicación de recomendaciones KNN."""

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, render

from apps.accounts.permissions import entrenador_required

from .services import KNNRecommendationService, get_routine_by_knn_name

User = get_user_model()


@entrenador_required
def recommendation_generate_view(request, username):
    """Genera una recomendación KNN para un cliente (solo entrenadores)."""
    client = get_object_or_404(User, username=username, role=User.Role.USUARIO)

    service = KNNRecommendationService()
    user_features = service.extract_user_features(client)

    try:
        result = service.get_recommendation(user_features)
    except FileNotFoundError as e:
        result = {
            "success": False,
            "error": "model_missing",
            "message": f"Archivo del modelo o dataset no encontrado: {e}",
        }
    except RuntimeError as e:
        result = {
            "success": False,
            "error": "model_error",
            "message": f"Error cargando el modelo: {e}",
        }

    matched_routine = None
    if result.get("success") and result.get("recommended_routine"):
        matched_routine = get_routine_by_knn_name(result["recommended_routine"])

    context = {
        "client": client,
        "result": result,
        "features": user_features,
        "matched_routine": matched_routine,
    }
    return render(request, "recommendations/recommendation_detail.html", context)