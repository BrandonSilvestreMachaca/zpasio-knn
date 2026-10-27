"""Servicio de recomendación KNN supervisado para Zpasio.

Carga el clasificador KNeighborsClassifier preentrenado y su preprocesador,
y genera recomendaciones basadas en la predicción de la etiqueta rutina_zpasio.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

logger = logging.getLogger(__name__)


class KNNRecommendationService:
    """Servicio para generar recomendaciones de rutinas usando KNN supervisado."""

    REQUIRED_NUMERIC_FIELDS = [
        "edad",
        "imc",
        "porcentaje_grasa",
        "dias_disponibles",
        "tiempo_sesion_minutos",
        "frecuencia_previa",
    ]

    REQUIRED_CATEGORICAL_FIELDS = [
        "sexo",
        "objetivo",
        "experiencia",
    ]

    REQUIRED_FIELDS = REQUIRED_NUMERIC_FIELDS + REQUIRED_CATEGORICAL_FIELDS

    K_NEIGHBORS = 15

    CATEGORY_MAPPING = {
        "sexo": {
            "MASCULINO": "Masculino",
            "FEMENINO": "Femenino",
            "OTRO": "Otro",
            "PREFIERO_NO_DECIR": "Prefiero no decirlo",
        },
        "objetivo": {
            "PERDIDA_PESO": "Pérdida de peso",
            "GANANCIA_MUSCULAR": "Ganancia muscular",
            "MANTENIMIENTO": "Mantenimiento",
            "RESISTENCIA": "Resistencia",
        },
        "experiencia": {
            "PRINCIPIANTE": "Principiante",
            "INTERMEDIO": "Intermedio",
            "AVANZADO": "Avanzado",
        },
    }

    def __init__(self):
        self._preprocesador = None
        self._classifier = None
        self._dataset = None
        self._models_loaded = False

    def _get_model_paths(self):
        """Obtiene las rutas de los archivos del modelo basadas en BASE_DIR."""
        base_dir = Path(settings.BASE_DIR)
        ml_dir = base_dir / "apps" / "recommendations" / "ml"
        return {
            "preprocesador": ml_dir / "preprocesador_classifier_candidate.joblib",
            "classifier": ml_dir / "knn_classifier_candidate.joblib",
            "dataset": ml_dir / "dataset_modelo_zpasio.csv",
        }

    def _load_models(self):
        """Carga el preprocesador, clasificador KNN y dataset."""
        if self._models_loaded:
            return

        paths = self._get_model_paths()

        missing_files = []
        for name, path in paths.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")

        if missing_files:
            raise FileNotFoundError(
                f"Archivos del modelo faltantes:\n" + "\n".join(missing_files)
            )

        try:
            self._preprocesador = joblib.load(paths["preprocesador"])
            self._classifier = joblib.load(paths["classifier"])
            self._dataset = pd.read_csv(paths["dataset"])
            self._models_loaded = True
            logger.info("Modelos KNN supervisados cargados correctamente")
        except Exception as e:
            logger.exception("Error cargando modelos KNN supervisados")
            raise RuntimeError(f"Error cargando modelos: {e}") from e

    def validate_user_data(self, user_data: dict) -> list[str]:
        """Valida que todos los campos requeridos estén presentes y no sean None.

        Args:
            user_data: Diccionario con los datos del usuario.

        Returns:
            Lista de nombres de campos faltantes (vacía si todos están presentes).
        """
        missing = []
        for field in self.REQUIRED_FIELDS:
            value = user_data.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                missing.append(field)
        return missing

    @classmethod
    def normalize_categories(cls, user_data: dict) -> dict:
        """Normaliza categorías de Django al formato esperado por el preprocesador KNN.

        Args:
            user_data: Diccionario con los datos del usuario (valores internos de Django).

        Returns:
            Diccionario con las categorías normalizadas.
        """
        normalized = user_data.copy()
        for field, mapping in cls.CATEGORY_MAPPING.items():
            value = normalized.get(field)
            if value in mapping:
                normalized[field] = mapping[value]
        return normalized

    def build_user_dataframe(self, user_data: dict) -> pd.DataFrame:
        """Construye un DataFrame de una fila con las features exactas del modelo.

        Args:
            user_data: Diccionario con los datos del usuario (claves = nombres de features).

        Returns:
            DataFrame con una fila y columnas en el orden esperado por el preprocesador.
        """
        normalized_data = self.normalize_categories(user_data)
        df = pd.DataFrame([normalized_data], columns=self.REQUIRED_FIELDS)
        return df

    def get_recommendation(self, user_data: dict) -> dict:
        """Genera una recomendación de rutina para el usuario dado usando el clasificador supervisado.

        Args:
            user_data: Diccionario con las features requeridas.

        Returns:
            Diccionario con el resultado de la recomendación.
        """
        self._load_models()

        missing_fields = self.validate_user_data(user_data)
        if missing_fields:
            field_labels = {
                "edad": "edad",
                "imc": "IMC",
                "porcentaje_grasa": "porcentaje de grasa",
                "dias_disponibles": "días disponibles",
                "tiempo_sesion_minutos": "tiempo disponible por sesión",
                "frecuencia_previa": "frecuencia previa",
                "sexo": "sexo",
                "objetivo": "objetivo",
                "experiencia": "experiencia",
            }
            missing_labels = [field_labels.get(f, f) for f in missing_fields]
            return {
                "success": False,
                "error": "missing_data",
                "message": (
                    "No se puede generar la recomendación porque faltan:\n- "
                    + "\n- ".join(missing_labels)
                ),
                "missing_fields": missing_fields,
            }

        user_df = self.build_user_dataframe(user_data)

        try:
            user_processed = self._preprocesador.transform(user_df)
        except Exception as e:
            logger.exception("Error en preprocesamiento")
            return {
                "success": False,
                "error": "preprocessing_error",
                "message": f"Error procesando los datos del usuario: {e}",
            }

        try:
            predicted_routine = self._classifier.predict(user_processed)[0]
            vote_proportions = self._classifier.predict_proba(user_processed)[0]
            class_labels = self._classifier.classes_

            distances, indices = self._classifier.kneighbors(
                user_processed, n_neighbors=self.K_NEIGHBORS
            )
        except Exception as e:
            logger.exception("Error consultando clasificador KNN")
            return {
                "success": False,
                "error": "knn_error",
                "message": f"Error consultando el clasificador: {e}",
            }

        distances = distances[0]
        indices = indices[0]

        if len(self._dataset) < self.K_NEIGHBORS:
            return {
                "success": False,
                "error": "insufficient_data",
                "message": (
                    f"El dataset tiene {len(self._dataset)} registros; "
                    f"se necesitan al menos {self.K_NEIGHBORS}."
                ),
            }

        neighbors = self._dataset.iloc[indices].copy()
        neighbors["distancia_knn"] = distances

        neighbor_routines = neighbors["rutina_zpasio"].value_counts().to_dict()

        routine_distances = {}
        for routine in class_labels:
            routine_mask = neighbors["rutina_zpasio"] == routine
            if routine_mask.any():
                routine_distances[routine] = round(neighbors.loc[routine_mask, "distancia_knn"].mean(), 4)
            else:
                routine_distances[routine] = None

        vote_details = []
        for i, routine in enumerate(class_labels):
            vote_details.append({
                "rutina_zpasio": routine,
                "proporcion_votos": round(vote_proportions[i], 4),
                "vecinos_cercanos": neighbor_routines.get(routine, 0),
                "distancia_promedio": routine_distances.get(routine),
            })

        vote_details.sort(key=lambda x: x["proporcion_votos"], reverse=True)

        matched_routine = predicted_routine
        avg_distance = routine_distances.get(matched_routine)
        cases = neighbor_routines.get(matched_routine, 0)

        return {
            "success": True,
            "recommended_routine": matched_routine,
            "cases": cases,
            "avg_distance": avg_distance,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": self.K_NEIGHBORS,
            "vote_proportions": {vd["rutina_zpasio"]: vd["proporcion_votos"] for vd in vote_details},
            "vote_details": vote_details,
            "neighbor_routines": neighbor_routines,
            "alternatives": vote_details[1:],
        }

    def extract_user_features(self, user) -> dict:
        """Extrae las features necesarias del modelo a partir de un usuario Django.

        Args:
            user: Instancia de CustomUser con profile y physical_data relacionados.

        Returns:
            Diccionario con las features en el formato esperado por el modelo.
        """
        profile = getattr(user, "profile", None)
        physical_data = getattr(user, "physical_data", None)

        if not profile or not physical_data:
            return {}

        features = {}

        features["edad"] = profile.age
        features["sexo"] = profile.gender
        features["objetivo"] = profile.goal
        features["experiencia"] = profile.experience_level
        features["dias_disponibles"] = profile.available_days_per_week
        features["tiempo_sesion_minutos"] = profile.session_time_available_minutes
        features["frecuencia_previa"] = profile.previous_training_days_per_week

        features["imc"] = float(physical_data.bmi) if physical_data.bmi else None
        features["porcentaje_grasa"] = (
            float(physical_data.body_fat_percentage)
            if physical_data.body_fat_percentage
            else None
        )

        return features