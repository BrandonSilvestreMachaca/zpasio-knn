"""Mapeo explícito entre nombres de rutina del modelo KNN y rutinas en base de datos.

Este mapeo es controlado y explícito para evitar enlaces incorrectos
que podrían ocurrir con fuzzy matching genérico.
"""

from apps.routines.models import Routine


KNN_TO_ROUTINE_NAME_MAP = {
    "Upper/Lower": "Upper / Lower",
    "Torso/Pierna": "Torso / Pierna",
    "Push Pull Legs": "Push Pull Legs (PPL)",
    "Full Body": "Full Body",
    "Arnold Split": "Arnold Split",
}


def get_routine_by_knn_name(knn_routine_name: str) -> Routine | None:
    """Busca una Routine en BD que coincida con el nombre devuelto por KNN.

    Args:
        knn_routine_name: Nombre de rutina devuelto por el modelo KNN.

    Returns:
        Instancia de Routine si existe correspondencia exacta en el mapeo
        y la rutina está activa en BD, None en caso contrario.
    """
    db_name = KNN_TO_ROUTINE_NAME_MAP.get(knn_routine_name)
    if not db_name:
        return None

    return Routine.objects.filter(name=db_name, is_active=True).first()


def get_knn_names_with_matching_routine() -> list[str]:
    """Devuelve lista de nombres KNN que tienen correspondencia en BD."""
    return [
        knn_name
        for knn_name, db_name in KNN_TO_ROUTINE_NAME_MAP.items()
        if Routine.objects.filter(name=db_name, is_active=True).exists()
    ]


def get_all_mappings() -> dict[str, str]:
    """Devuelve el mapeo completo KNN -> nombre en BD."""
    return KNN_TO_ROUTINE_NAME_MAP.copy()