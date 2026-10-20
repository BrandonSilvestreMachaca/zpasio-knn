"""Pruebas para la aplicación de recomendaciones KNN."""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from django.contrib.auth import get_user_model
from django.test import TestCase, Client
from django.urls import reverse

from apps.profiles.models import Profile, PhysicalData
from apps.routines.models import Routine
from apps.recommendations.services import (
    KNNRecommendationService,
    get_routine_by_knn_name,
    get_knn_names_with_matching_routine,
    get_all_mappings,
    KNN_TO_ROUTINE_NAME_MAP,
)

User = get_user_model()


class KNNRecommendationServiceTests(TestCase):
    """Pruebas unitarias para KNNRecommendationService."""

    def setUp(self):
        self.service = KNNRecommendationService()

    @patch("apps.recommendations.services.knn_service.joblib.load")
    @patch("apps.recommendations.services.knn_service.pd.read_csv")
    def test_load_models_success(self, mock_read_csv, mock_joblib_load):
        """Verifica que los modelos se cargan correctamente."""
        mock_preprocesador = MagicMock()
        mock_classifier = MagicMock()
        mock_dataset = MagicMock()

        mock_joblib_load.side_effect = [mock_preprocesador, mock_classifier]
        mock_read_csv.return_value = mock_dataset

        self.service._load_models()

        self.assertTrue(self.service._models_loaded)
        self.assertEqual(self.service._preprocesador, mock_preprocesador)
        self.assertEqual(self.service._classifier, mock_classifier)
        self.assertEqual(self.service._dataset, mock_dataset)

    @patch("apps.recommendations.services.knn_service.joblib.load")
    def test_load_models_missing_file(self, mock_joblib_load):
        """Verifica error controlado cuando falta archivo del modelo."""
        mock_joblib_load.side_effect = FileNotFoundError("No such file")

        with self.assertRaises(RuntimeError):
            self.service._load_models()

    def test_validate_user_data_all_present(self):
        """Verifica validación exitosa cuando todos los campos están presentes."""
        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        missing = self.service.validate_user_data(user_data)
        self.assertEqual(missing, [])

    def test_validate_user_data_missing_fields(self):
        """Verifica detección de campos faltantes."""
        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": None,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": None,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        missing = self.service.validate_user_data(user_data)
        self.assertIn("porcentaje_grasa", missing)
        self.assertIn("tiempo_sesion_minutos", missing)

    def test_validate_user_data_empty_string(self):
        """Verifica que strings vacíos se detectan como faltantes."""
        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        missing = self.service.validate_user_data(user_data)
        self.assertIn("sexo", missing)

    def test_build_user_dataframe(self):
        """Verifica construcción correcta del DataFrame."""
        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        df = self.service.build_user_dataframe(user_data)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), self.service.REQUIRED_FIELDS)
        self.assertEqual(df.iloc[0]["edad"], 25)
        self.assertEqual(df.iloc[0]["sexo"], "Masculino")

    @patch.object(KNNRecommendationService, "_load_models")
    def test_get_recommendation_missing_data(self, mock_load):
        """Verifica error controlado cuando faltan datos del usuario."""
        mock_load.return_value = None
        self.service._models_loaded = True
        self.service._preprocesador = MagicMock()
        self.service._classifier = MagicMock()
        self.service._dataset = MagicMock()
        self.service._dataset.__len__ = MagicMock(return_value=100)

        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": None,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": None,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        result = self.service.get_recommendation(user_data)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "missing_data")
        self.assertIn("porcentaje de grasa", result["message"])
        self.assertIn("tiempo disponible por sesión", result["message"])

    @patch.object(KNNRecommendationService, "_load_models")
    def test_get_recommendation_preprocessing_error(self, mock_load):
        """Verifica manejo de error en preprocesamiento."""
        mock_load.return_value = None
        self.service._models_loaded = True
        self.service._preprocesador = MagicMock()
        self.service._preprocesador.transform.side_effect = ValueError("Error transform")
        self.service._classifier = MagicMock()
        self.service._dataset = MagicMock()
        self.service._dataset.__len__ = MagicMock(return_value=100)

        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        result = self.service.get_recommendation(user_data)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "preprocessing_error")

    @patch.object(KNNRecommendationService, "_load_models")
    def test_get_recommendation_classifier_error(self, mock_load):
        """Verifica manejo de error en consulta al clasificador."""
        mock_load.return_value = None
        self.service._models_loaded = True
        self.service._preprocesador = MagicMock()
        self.service._preprocesador.transform.return_value = [[0.1] * 15]
        self.service._classifier = MagicMock()
        self.service._classifier.predict.side_effect = RuntimeError("Classifier error")
        self.service._dataset = MagicMock()
        self.service._dataset.__len__ = MagicMock(return_value=100)

        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        result = self.service.get_recommendation(user_data)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "knn_error")

    @patch.object(KNNRecommendationService, "_load_models")
    def test_get_recommendation_insufficient_dataset(self, mock_load):
        """Verifica error cuando dataset tiene menos de 15 registros."""
        mock_load.return_value = None
        self.service._models_loaded = True
        self.service._preprocesador = MagicMock()
        self.service._preprocesador.transform.return_value = [[0.1] * 15]
        self.service._classifier = MagicMock()
        self.service._classifier.predict.return_value = ["Upper/Lower"]
        self.service._classifier.predict_proba.return_value = [[0.2, 0.2, 0.2, 0.2, 0.2]]
        self.service._classifier.classes_ = np.array(["A", "B", "C", "D", "E"])
        self.service._classifier.kneighbors.return_value = ([[0.1] * 15], [[0] * 15])
        self.service._dataset = MagicMock()
        self.service._dataset.__len__ = MagicMock(return_value=10)

        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        result = self.service.get_recommendation(user_data)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "insufficient_data")

    @patch.object(KNNRecommendationService, "_load_models")
    def test_get_recommendation_successful_prediction(self, mock_load):
        """Verifica predicción exitosa con el clasificador supervisado."""
        mock_load.return_value = None
        self.service._models_loaded = True
        self.service._preprocesador = MagicMock()
        self.service._preprocesador.transform.return_value = [[0.1] * 15]
        self.service._classifier = MagicMock()
        self.service._classifier.predict.return_value = np.array(["Torso/Pierna"])
        self.service._classifier.predict_proba.return_value = np.array([[0.1, 0.2, 0.1, 0.4, 0.2]])
        self.service._classifier.classes_ = np.array(["Arnold Split", "Full Body", "Push Pull Legs", "Torso/Pierna", "Upper/Lower"])
        self.service._classifier.kneighbors.return_value = (np.array([[0.1] * 15]), np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]))

        mock_dataset = pd.DataFrame({
            "rutina_zpasio": ["Torso/Pierna"] * 5 + ["Upper/Lower"] * 4 + ["Full Body"] * 3 + ["Push Pull Legs"] * 2 + ["Arnold Split"] * 1,
        })
        self.service._dataset = mock_dataset

        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "Ganancia muscular",
            "experiencia": "Intermedio",
        }
        result = self.service.get_recommendation(user_data)

        self.assertTrue(result["success"])
        self.assertEqual(result["recommended_routine"], "Torso/Pierna")
        self.assertEqual(result["cases"], 5)
        self.assertIn("vote_proportions", result)
        self.assertIn("vote_details", result)
        self.assertEqual(result["neighbors_analyzed"], 15)
        self.assertIsNone(result["avg_adherence"])
        self.assertIsNone(result["weighted_score"])

    def test_extract_user_features_complete(self):
        """Verifica extracción de features de usuario completo."""
        user = User.objects.create_user(
            username="testuser",
            password="testpass123",
            role=User.Role.USUARIO,
        )
        Profile.objects.create(
            user=user,
            birth_date=date(2000, 1, 1),
            gender="Masculino",
            goal="GANANCIA_MUSCULAR",
            experience_level="INTERMEDIO",
            available_days_per_week=4,
            session_time_available_minutes=60,
            previous_training_days_per_week=2,
        )
        PhysicalData.objects.create(
            user=user,
            weight_kg=Decimal("70.0"),
            height_cm=Decimal("175.0"),
            body_fat_percentage=Decimal("15.0"),
        )

        features = self.service.extract_user_features(user)

        self.assertEqual(features["sexo"], "Masculino")
        self.assertEqual(features["objetivo"], "GANANCIA_MUSCULAR")
        self.assertEqual(features["experiencia"], "INTERMEDIO")
        self.assertEqual(features["dias_disponibles"], 4)
        self.assertEqual(features["tiempo_sesion_minutos"], 60)
        self.assertEqual(features["frecuencia_previa"], 2)
        self.assertIsNotNone(features["imc"])
        self.assertIsNotNone(features["porcentaje_grasa"])
        self.assertIsNotNone(features["edad"])

    def test_extract_user_features_missing_profile(self):
        """Verifica extracción con perfil faltante."""
        user = User.objects.create_user(
            username="testuser2",
            password="testpass123",
            role=User.Role.USUARIO,
        )
        features = self.service.extract_user_features(user)
        self.assertEqual(features, {})

    def test_extract_user_features_missing_physical_data(self):
        """Verifica extracción con datos físicos faltantes."""
        user = User.objects.create_user(
            username="testuser3",
            password="testpass123",
            role=User.Role.USUARIO,
        )
        Profile.objects.create(
            user=user,
            birth_date=date(2000, 1, 1),
            gender="Masculino",
        )
        features = self.service.extract_user_features(user)
        self.assertIsNone(features.get("imc"))
        self.assertIsNone(features.get("porcentaje_grasa"))

    def test_normalize_categories_sexo_masculino(self):
        """Verifica que MASCULINO se transforma a Masculino."""
        user_data = {"sexo": "MASCULINO"}
        normalized = self.service.normalize_categories(user_data)
        self.assertEqual(normalized["sexo"], "Masculino")

    def test_normalize_categories_sexo_femenino(self):
        """Verifica que FEMENINO se transforma a Femenino."""
        user_data = {"sexo": "FEMENINO"}
        normalized = self.service.normalize_categories(user_data)
        self.assertEqual(normalized["sexo"], "Femenino")

    def test_normalize_categories_objetivo_ganancia_muscular(self):
        """Verifica que GANANCIA_MUSCULAR se transforma correctamente."""
        user_data = {"objetivo": "GANANCIA_MUSCULAR"}
        normalized = self.service.normalize_categories(user_data)
        self.assertEqual(normalized["objetivo"], "Ganancia muscular")

    def test_normalize_categories_objetivo_acondicionamiento_general(self):
        """Verifica que ACONDICIONAMIENTO_GENERAL se transforma correctamente."""
        user_data = {"objetivo": "ACONDICIONAMIENTO_GENERAL"}
        normalized = self.service.normalize_categories(user_data)
        self.assertEqual(normalized["objetivo"], "Acondicionamiento general")

    def test_normalize_categories_experiencia_all(self):
        """Verifica que PRINCIPIANTE/INTERMEDIO/AVANZADO se transforman correctamente."""
        for django_val, expected in [
            ("PRINCIPIANTE", "Principiante"),
            ("INTERMEDIO", "Intermedio"),
            ("AVANZADO", "Avanzado"),
        ]:
            with self.subTest(django_val=django_val):
                user_data = {"experiencia": django_val}
                normalized = self.service.normalize_categories(user_data)
                self.assertEqual(normalized["experiencia"], expected)

    def test_normalize_categories_all_fields_together(self):
        """Verifica normalización de todos los campos categóricos juntos."""
        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "FEMENINO",
            "objetivo": "FUERZA",
            "experiencia": "AVANZADO",
        }
        normalized = self.service.normalize_categories(user_data)
        self.assertEqual(normalized["sexo"], "Femenino")
        self.assertEqual(normalized["objetivo"], "Fuerza")
        self.assertEqual(normalized["experiencia"], "Avanzado")
        self.assertEqual(normalized["edad"], 25)

    def test_build_user_dataframe_contains_normalized_categories(self):
        """Verifica que el DataFrame enviado al preprocesador contiene las categorías esperadas."""
        user_data = {
            "edad": 25,
            "imc": 22.5,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "MASCULINO",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "PRINCIPIANTE",
        }
        df = self.service.build_user_dataframe(user_data)
        self.assertEqual(df.iloc[0]["sexo"], "Masculino")
        self.assertEqual(df.iloc[0]["objetivo"], "Ganancia muscular")
        self.assertEqual(df.iloc[0]["experiencia"], "Principiante")


class RoutineMappingTests(TestCase):
    """Pruebas para el mapeo explícito entre nombres KNN y rutinas en BD."""

    def setUp(self):
        self.routine_upper_lower = Routine.objects.create(
            name="Upper / Lower",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
            is_active=True,
        )
        self.routine_torso_pierna = Routine.objects.create(
            name="Torso / Pierna",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
            is_active=True,
        )
        self.routine_ppl = Routine.objects.create(
            name="Push Pull Legs",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=6,
            estimated_duration_minutes=75,
            is_active=True,
        )
        self.routine_full_body = Routine.objects.create(
            name="Full Body",
            goal="MANTENIMIENTO",
            level="PRINCIPIANTE",
            days_per_week=3,
            estimated_duration_minutes=45,
            is_active=True,
        )
        self.routine_arnold = Routine.objects.create(
            name="Arnold Split",
            goal="GANANCIA_MUSCULAR",
            level="AVANZADO",
            days_per_week=6,
            estimated_duration_minutes=90,
            is_active=True,
        )
        self.routine_inactive = Routine.objects.create(
            name="Upper / Lower",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
            is_active=False,
        )

    def test_mapping_contains_expected_keys(self):
        """Verifica que el mapeo contiene las claves esperadas del modelo KNN."""
        expected_keys = {
            "Upper/Lower",
            "Torso/Pierna",
            "Push Pull Legs",
            "Full Body",
            "Arnold Split",
        }
        self.assertEqual(set(KNN_TO_ROUTINE_NAME_MAP.keys()), expected_keys)

    def test_mapping_upper_lower(self):
        """Verifica mapeo Upper/Lower -> Upper / Lower."""
        self.assertEqual(KNN_TO_ROUTINE_NAME_MAP["Upper/Lower"], "Upper / Lower")

    def test_mapping_torso_pierna(self):
        """Verifica mapeo Torso/Pierna -> Torso / Pierna."""
        self.assertEqual(KNN_TO_ROUTINE_NAME_MAP["Torso/Pierna"], "Torso / Pierna")

    def test_mapping_push_pull_legs_exact(self):
        """Verifica que Push Pull Legs mapea a sí mismo (coincidencia exacta)."""
        self.assertEqual(KNN_TO_ROUTINE_NAME_MAP["Push Pull Legs"], "Push Pull Legs")

    def test_mapping_full_body_exact(self):
        """Verifica que Full Body mapea a sí mismo (coincidencia exacta)."""
        self.assertEqual(KNN_TO_ROUTINE_NAME_MAP["Full Body"], "Full Body")

    def test_mapping_arnold_split_exact(self):
        """Verifica que Arnold Split mapea a sí mismo (coincidencia exacta)."""
        self.assertEqual(KNN_TO_ROUTINE_NAME_MAP["Arnold Split"], "Arnold Split")

    def test_get_routine_by_knn_name_upper_lower(self):
        """Verifica búsqueda de rutina por nombre KNN Upper/Lower."""
        routine = get_routine_by_knn_name("Upper/Lower")
        self.assertIsNotNone(routine)
        self.assertEqual(routine.name, "Upper / Lower")
        self.assertEqual(routine.pk, self.routine_upper_lower.pk)

    def test_get_routine_by_knn_name_torso_pierna(self):
        """Verifica búsqueda de rutina por nombre KNN Torso/Pierna."""
        routine = get_routine_by_knn_name("Torso/Pierna")
        self.assertIsNotNone(routine)
        self.assertEqual(routine.name, "Torso / Pierna")
        self.assertEqual(routine.pk, self.routine_torso_pierna.pk)

    def test_get_routine_by_knn_name_push_pull_legs(self):
        """Verifica búsqueda de rutina por nombre KNN Push Pull Legs."""
        routine = get_routine_by_knn_name("Push Pull Legs")
        self.assertIsNotNone(routine)
        self.assertEqual(routine.name, "Push Pull Legs")
        self.assertEqual(routine.pk, self.routine_ppl.pk)

    def test_get_routine_by_knn_name_full_body(self):
        """Verifica búsqueda de rutina por nombre KNN Full Body."""
        routine = get_routine_by_knn_name("Full Body")
        self.assertIsNotNone(routine)
        self.assertEqual(routine.name, "Full Body")
        self.assertEqual(routine.pk, self.routine_full_body.pk)

    def test_get_routine_by_knn_name_arnold_split(self):
        """Verifica búsqueda de rutina por nombre KNN Arnold Split."""
        routine = get_routine_by_knn_name("Arnold Split")
        self.assertIsNotNone(routine)
        self.assertEqual(routine.name, "Arnold Split")
        self.assertEqual(routine.pk, self.routine_arnold.pk)

    def test_get_routine_by_knn_name_unknown_returns_none(self):
        """Verifica que nombre KNN desconocido devuelve None."""
        routine = get_routine_by_knn_name("Rutina Desconocida")
        self.assertIsNone(routine)

    def test_get_routine_by_knn_name_inactive_returns_none(self):
        """Verifica que rutina inactiva no se devuelve."""
        Routine.objects.filter(name="Upper / Lower", is_active=False).delete()
        inactive = Routine.objects.create(
            name="Upper / Lower",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
            is_active=False,
        )
        routine = get_routine_by_knn_name("Upper/Lower")
        self.assertIsNotNone(routine)
        self.assertTrue(routine.is_active)
        self.assertEqual(routine.pk, self.routine_upper_lower.pk)

    def test_get_routine_by_knn_name_no_db_match_returns_none(self):
        """Verifica que si no existe en BD devuelve None."""
        Routine.objects.all().delete()
        routine = get_routine_by_knn_name("Upper/Lower")
        self.assertIsNone(routine)

    def test_get_knn_names_with_matching_routine(self):
        """Verifica lista de nombres KNN con correspondencia en BD."""
        matching = get_knn_names_with_matching_routine()
        self.assertIn("Upper/Lower", matching)
        self.assertIn("Torso/Pierna", matching)
        self.assertIn("Push Pull Legs", matching)
        self.assertIn("Full Body", matching)
        self.assertIn("Arnold Split", matching)
        self.assertEqual(len(matching), 5)

    def test_get_knn_names_with_matching_routine_empty_when_no_routines(self):
        """Verifica lista vacía cuando no hay rutinas en BD."""
        Routine.objects.all().delete()
        matching = get_knn_names_with_matching_routine()
        self.assertEqual(matching, [])

    def test_get_all_mappings_returns_copy(self):
        """Verifica que get_all_mappings devuelve una copia del mapeo."""
        mappings = get_all_mappings()
        self.assertEqual(mappings, KNN_TO_ROUTINE_NAME_MAP)
        self.assertIsNot(mappings, KNN_TO_ROUTINE_NAME_MAP)


class RecommendationViewsTests(TestCase):
    """Pruebas de integración para las vistas de recomendaciones."""

    def setUp(self):
        self.client = Client()

        self.trainer = User.objects.create_user(
            username="trainer",
            password="trainerpass123",
            role=User.Role.ENTRENADOR,
        )

        self.user = User.objects.create_user(
            username="testuser",
            password="userpass123",
            role=User.Role.USUARIO,
        )

        self.profile = Profile.objects.create(
            user=self.user,
            birth_date=date(2000, 1, 1),
            gender="Masculino",
            goal="GANANCIA_MUSCULAR",
            experience_level="INTERMEDIO",
            available_days_per_week=4,
            session_time_available_minutes=60,
            previous_training_days_per_week=2,
        )

        PhysicalData.objects.create(
            user=self.user,
            weight_kg=Decimal("70.0"),
            height_cm=Decimal("175.0"),
            body_fat_percentage=Decimal("15.0"),
        )

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_entrenador_can_generate_recommendation(self, mock_service):
        """1. Entrenador puede generar recomendación."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4, "Torso/Pierna": 0.3, "Full Body": 0.1, "Push Pull Legs": 0.1, "Arnold Split": 0.1},
            "vote_details": [
                {"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15},
                {"rutina_zpasio": "Torso/Pierna", "proporcion_votos": 0.3, "vecinos_cercanos": 5, "distancia_promedio": 0.2},
                {"rutina_zpasio": "Full Body", "proporcion_votos": 0.1, "vecinos_cercanos": 2, "distancia_promedio": 0.3},
                {"rutina_zpasio": "Push Pull Legs", "proporcion_votos": 0.1, "vecinos_cercanos": 1, "distancia_promedio": 0.35},
                {"rutina_zpasio": "Arnold Split", "proporcion_votos": 0.1, "vecinos_cercanos": 1, "distancia_promedio": 0.4},
            ],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recommendations/recommendation_detail.html")

    def test_usuario_normal_recibe_403(self):
        """2. Usuario normal recibe 403."""
        self.client.force_login(self.user)

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 403)

    def test_anonimo_redirige_a_login(self):
        """3. Anónimo redirige a login."""
        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_usuario_completo_obtiene_recomendacion(self, mock_service):
        """4. Usuario completo obtiene recomendación."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4, "Torso/Pierna": 0.3, "Full Body": 0.1, "Push Pull Legs": 0.1, "Arnold Split": 0.1},
            "vote_details": [
                {"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15},
                {"rutina_zpasio": "Torso/Pierna", "proporcion_votos": 0.3, "vecinos_cercanos": 5, "distancia_promedio": 0.2},
                {"rutina_zpasio": "Full Body", "proporcion_votos": 0.1, "vecinos_cercanos": 2, "distancia_promedio": 0.3},
                {"rutina_zpasio": "Push Pull Legs", "proporcion_votos": 0.1, "vecinos_cercanos": 1, "distancia_promedio": 0.35},
                {"rutina_zpasio": "Arnold Split", "proporcion_votos": 0.1, "vecinos_cercanos": 1, "distancia_promedio": 0.4},
            ],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upper/Lower")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_usuario_con_dato_faltante_recibe_mensaje_claro(self, mock_service):
        """5. Usuario con dato requerido faltante recibe mensaje claro."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": None,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": None,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": False,
            "error": "missing_data",
            "message": "No se puede generar la recomendación porque faltan:\n- porcentaje de grasa\n- tiempo disponible por sesión",
            "missing_fields": ["porcentaje_grasa", "tiempo_sesion_minutos"],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "porcentaje de grasa")
        self.assertContains(response, "tiempo disponible por sesión")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_no_se_inventan_valores_faltantes(self, mock_service):
        """6. No se inventan valores faltantes."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": None,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": False,
            "error": "missing_data",
            "message": "No se puede generar la recomendación porque faltan:\n- porcentaje de grasa",
            "missing_fields": ["porcentaje_grasa"],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "porcentaje de grasa")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_se_consultan_15_vecinos(self, mock_service):
        """7. Se consultan 15 vecinos (K=15)."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        mock_instance.get_recommendation.assert_called_once()

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_recomendacion_no_crea_userroutineassignment(self, mock_service):
        """8. La recomendación NO crea UserRoutineAssignment."""
        from apps.routines.models import UserRoutineAssignment

        self.client.force_login(self.trainer)
        initial_count = UserRoutineAssignment.objects.count()

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(UserRoutineAssignment.objects.count(), initial_count)

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_rutina_existente_puede_enlazarse(self, mock_service):
        """9. Rutina existente puede enlazarse con el resultado."""
        Routine.objects.create(
            name="Upper/Lower",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
        )

        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(Routine.objects.filter(name="Upper/Lower").exists())

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_rutina_inexistente_produce_mensaje_controlado(self, mock_service):
        """10. Rutina inexistente produce mensaje controlado."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Rutina Inexistente XYZ",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Rutina Inexistente XYZ": 0.4},
            "vote_details": [{"rutina_zpasio": "Rutina Inexistente XYZ", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Rutina Inexistente XYZ")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_archivo_modelo_faltante_produce_error_controlado(self, mock_service):
        """11. Archivo del modelo faltante produce error controlado."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.side_effect = FileNotFoundError("Archivo no encontrado")

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Archivo del modelo o dataset no encontrado")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_dataset_faltante_produce_error_controlado(self, mock_service):
        """12. Dataset faltante produce error controlado."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.side_effect = FileNotFoundError("Dataset no encontrado")

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Archivo del modelo o dataset no encontrado")

    def test_usuarios_existentes_siguen_intactos(self):
        """13. Usuarios existentes siguen intactos."""
        initial_count = User.objects.count()
        self.assertGreaterEqual(initial_count, 2)

    def test_rutinas_existentes_siguen_intactas(self):
        """14. Rutinas existentes siguen intactas."""
        initial_count = Routine.objects.count()
        self.assertGreaterEqual(initial_count, 0)

    def test_evaluaciones_existentes_siguen_intactas(self):
        """15. Evaluaciones existentes siguen intactas."""
        from apps.assessments.models import PhysicalEvaluation
        initial_count = PhysicalEvaluation.objects.count()
        self.assertGreaterEqual(initial_count, 0)

    def test_asignaciones_existentes_siguen_intactas(self):
        """16. Asignaciones existentes siguen intactas."""
        from apps.routines.models import UserRoutineAssignment
        initial_count = UserRoutineAssignment.objects.count()
        self.assertGreaterEqual(initial_count, 0)

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_recomendacion_con_rutina_enlazada_muestra_boton_ver_rutina(self, mock_service):
        """17. Recomendación con rutina enlazada muestra botón 'Ver rutina'."""
        routine = Routine.objects.create(
            name="Upper / Lower",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
            is_active=True,
        )

        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ver rutina")
        self.assertContains(response, f'href="/rutinas/{routine.pk}/"')

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_recomendacion_con_rutina_enlazada_muestra_boton_asignar_esta_rutina(self, mock_service):
        """18. Recomendación con rutina enlazada muestra botón 'Asignar esta rutina'."""
        routine = Routine.objects.create(
            name="Upper / Lower",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
            is_active=True,
        )

        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Asignar esta rutina")
        expected_url = f"/rutinas/asignaciones/nueva/testuser/?routine={routine.pk}"
        self.assertContains(response, expected_url)

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_recomendacion_sin_rutina_enlazada_solo_muestra_asignar_rutina_generico(self, mock_service):
        """19. Recomendación sin rutina enlazada muestra solo 'Asignar rutina' genérico."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Rutina Inexistente",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Rutina Inexistente": 0.4},
            "vote_details": [{"rutina_zpasio": "Rutina Inexistente", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Asignar rutina")
        self.assertNotContains(response, "Ver rutina")
        self.assertNotContains(response, "Asignar esta rutina")
        self.assertContains(response, '/rutinas/asignaciones/nueva/testuser/')
        self.assertNotContains(response, '?routine=')

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_recomendacion_con_rutina_enlazada_torso_pierna(self, mock_service):
        """20. Recomendación con Torso/Pierna enlaza correctamente con Torso / Pierna."""
        routine = Routine.objects.create(
            name="Torso / Pierna",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
            is_active=True,
        )

        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Torso/Pierna",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Torso/Pierna": 0.4},
            "vote_details": [{"rutina_zpasio": "Torso/Pierna", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ver rutina")
        self.assertContains(response, "Asignar esta rutina")
        self.assertContains(response, f'href="/rutinas/{routine.pk}/"')
        self.assertContains(response, f'?routine={routine.pk}')

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_recomendacion_push_pull_legs_enlace_exacto(self, mock_service):
        """21. Push Pull Legs usa coincidencia exacta sin espacios."""
        routine = Routine.objects.create(
            name="Push Pull Legs",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=6,
            estimated_duration_minutes=75,
            is_active=True,
        )

        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 6,
            "tiempo_sesion_minutos": 75,
            "frecuencia_previa": 3,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Push Pull Legs",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Push Pull Legs": 0.4},
            "vote_details": [{"rutina_zpasio": "Push Pull Legs", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ver rutina")
        self.assertContains(response, "Asignar esta rutina")
        self.assertContains(response, f'href="/rutinas/{routine.pk}/"')

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_recomendacion_full_body_enlace_exacto(self, mock_service):
        """22. Full Body usa coincidencia exacta."""
        routine = Routine.objects.create(
            name="Full Body",
            goal="MANTENIMIENTO",
            level="PRINCIPIANTE",
            days_per_week=3,
            estimated_duration_minutes=45,
            is_active=True,
        )

        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 3,
            "tiempo_sesion_minutos": 45,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "PRINCIPIANTE",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Full Body",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Full Body": 0.4},
            "vote_details": [{"rutina_zpasio": "Full Body", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ver rutina")
        self.assertContains(response, "Asignar esta rutina")
        self.assertContains(response, f'href="/rutinas/{routine.pk}/"')

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_recomendacion_arnold_split_enlace_exacto(self, mock_service):
        """23. Arnold Split usa coincidencia exacta."""
        routine = Routine.objects.create(
            name="Arnold Split",
            goal="GANANCIA_MUSCULAR",
            level="AVANZADO",
            days_per_week=6,
            estimated_duration_minutes=90,
            is_active=True,
        )

        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 6,
            "tiempo_sesion_minutos": 90,
            "frecuencia_previa": 5,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "AVANZADO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Arnold Split",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Arnold Split": 0.4},
            "vote_details": [{"rutina_zpasio": "Arnold Split", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ver rutina")
        self.assertContains(response, "Asignar esta rutina")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_asignar_esta_rutina_no_crea_asignacion_automatica(self, mock_service):
        """24. 'Asignar esta rutina' redirige al formulario, no crea asignación."""
        from apps.routines.models import UserRoutineAssignment

        routine = Routine.objects.create(
            name="Upper / Lower",
            goal="MANTENIMIENTO",
            level="INTERMEDIO",
            days_per_week=4,
            estimated_duration_minutes=60,
            is_active=True,
        )

        self.client.force_login(self.trainer)
        initial_count = UserRoutineAssignment.objects.count()

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(UserRoutineAssignment.objects.count(), initial_count)

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_sinteticos_warning_mostrado_en_template(self, mock_service):
        """25. El aviso de datos sintéticos se muestra en la plantilla."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "sintéticos")
        self.assertContains(response, "prototipo experimental")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_clasificador_supervisado_usa_k15_weights_uniform(self, mock_service):
        """26. El clasificador integrado usa K=15 y weights='uniform'."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_instance.get_recommendation.call_count, 1)

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_no_usar_adherencia_progreso_para_predecir(self, mock_service):
        """27. El resultado no incluye adherencia ni progreso sintético como criterio de predicción."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 7,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.4},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.4, "vecinos_cercanos": 6, "distancia_promedio": 0.15}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Adherencia promedio")
        self.assertNotContains(response, "Puntaje ponderado")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_vote_percentages_display_correctly(self, mock_service):
        """Verifica que los porcentajes de votos se muestran multiplicados por 100 con 2 decimales."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 10,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {
                "Upper/Lower": 0.6667,
                "Torso/Pierna": 0.2000,
                "Full Body": 0.0667,
                "Push Pull Legs": 0.0400,
                "Arnold Split": 0.0267,
            },
            "vote_details": [
                {"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.6667, "vecinos_cercanos": 10, "distancia_promedio": 0.15},
                {"rutina_zpasio": "Torso/Pierna", "proporcion_votos": 0.2000, "vecinos_cercanos": 3, "distancia_promedio": 0.20},
                {"rutina_zpasio": "Full Body", "proporcion_votos": 0.0667, "vecinos_cercanos": 1, "distancia_promedio": 0.25},
                {"rutina_zpasio": "Push Pull Legs", "proporcion_votos": 0.0400, "vecinos_cercanos": 1, "distancia_promedio": 0.30},
                {"rutina_zpasio": "Arnold Split", "proporcion_votos": 0.0267, "vecinos_cercanos": 0, "distancia_promedio": None},
            ],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "0,6667 (66,67%)")
        self.assertContains(response, "0,2000 (20,00%)")
        self.assertContains(response, "0,0667 (6,67%)")
        self.assertContains(response, "0,0400 (4,00%)")
        self.assertContains(response, "0,0267 (2,67%)")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_sin_vecinos_displayed_for_zero_neighbors(self, mock_service):
        """Verifica que 'Sin vecinos' se muestra cuando vecinos_cercanos es 0."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 10,
            "avg_distance": 0.15,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.6667, "Arnold Split": 0.0},
            "vote_details": [
                {"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.6667, "vecinos_cercanos": 10, "distancia_promedio": 0.15},
                {"rutina_zpasio": "Arnold Split", "proporcion_votos": 0.0, "vecinos_cercanos": 0, "distancia_promedio": None},
            ],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sin vecinos")

    @patch("apps.recommendations.views.KNNRecommendationService")
    def test_avg_distance_shows_sin_vecinos_when_cases_zero(self, mock_service):
        """Verifica que la distancia promedio principal muestra 'Sin vecinos' cuando cases es 0."""
        self.client.force_login(self.trainer)

        mock_instance = mock_service.return_value
        mock_instance.extract_user_features.return_value = {
            "edad": 24,
            "imc": 22.86,
            "porcentaje_grasa": 15.0,
            "dias_disponibles": 4,
            "tiempo_sesion_minutos": 60,
            "frecuencia_previa": 2,
            "sexo": "Masculino",
            "objetivo": "GANANCIA_MUSCULAR",
            "experiencia": "INTERMEDIO",
        }
        mock_instance.get_recommendation.return_value = {
            "success": True,
            "recommended_routine": "Upper/Lower",
            "cases": 0,
            "avg_distance": None,
            "avg_adherence": None,
            "weighted_score": None,
            "neighbors_analyzed": 15,
            "vote_proportions": {"Upper/Lower": 0.0},
            "vote_details": [{"rutina_zpasio": "Upper/Lower", "proporcion_votos": 0.0, "vecinos_cercanos": 0, "distancia_promedio": None}],
            "alternatives": [],
        }

        url = reverse("recommendations:generate", kwargs={"username": "testuser"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sin vecinos")


class EvaluarKNNSupervisadoCommandTests(TestCase):
    """Pruebas para el comando evaluar_knn_supervisado."""

    def setUp(self):
        from django.core.management import call_command
        from io import StringIO
        self.call_command = call_command
        self.out = StringIO()

    @patch("apps.recommendations.management.commands.evaluar_knn_supervisado.pd.read_csv")
    def test_command_runs_without_error(self, mock_read_csv):
        """Verifica que el comando se ejecuta sin errores básicos."""
        np.random.seed(42)
        n_samples = 100
        mock_df = pd.DataFrame({
            "id_sintetico": [f"SYN-{i:04d}" for i in range(n_samples)],
            "edad": np.random.randint(18, 65, n_samples),
            "sexo": np.random.choice(["Masculino", "Femenino"], n_samples),
            "imc": np.random.uniform(16, 35, n_samples),
            "porcentaje_grasa": np.random.uniform(7, 45, n_samples),
            "objetivo": np.random.choice(
                ["Ganancia muscular", "Fuerza", "Resistencia", "Acondicionamiento general"],
                n_samples
            ),
            "nivel_experiencia": np.random.randint(1, 4, n_samples),
            "experiencia": np.random.choice(["Principiante", "Intermedio", "Avanzado"], n_samples),
            "dias_disponibles": np.random.randint(2, 7, n_samples),
            "tiempo_sesion_minutos": np.random.choice([45, 60, 75, 90], n_samples),
            "frecuencia_previa": np.random.randint(0, 6, n_samples),
            "rutina_zpasio": np.random.choice(
                ["Full Body", "Upper/Lower", "Torso/Pierna", "Push Pull Legs", "Arnold Split"],
                n_samples
            ),
            "sesiones_planificadas": np.random.randint(12, 72, n_samples),
            "sesiones_realizadas": np.random.randint(8, 70, n_samples),
            "adherencia_porcentaje": np.random.uniform(50, 100, n_samples),
            "progreso_sintetico": np.random.uniform(0, 6, n_samples),
            "resultado_sintetico": np.random.choice(["Bajo", "Moderado", "Alto"], n_samples),
            "fuente": ["sintetico"] * n_samples,
        })
        for i, routine in enumerate(["Full Body", "Upper/Lower", "Torso/Pierna", "Push Pull Legs", "Arnold Split"]):
            mock_df.loc[i, "rutina_zpasio"] = routine

        mock_read_csv.return_value = mock_df

        try:
            self.call_command(
                "evaluar_knn_supervisado",
                "--k-values", "5", "10",
                "--weights", "uniform",
                "--output-csv", "/tmp/test_results.csv",
                "--summary-csv", "/tmp/test_summary.csv",
                "--per-fold-csv", "/tmp/test_per_fold.csv",
                "--confusion-csv", "/tmp/test_confusion.csv",
                stdout=self.out,
            )
        except SystemExit:
            pass

        output = self.out.getvalue()
        self.assertIn("EVALUACIÓN SUPERVISADA KNN CLASSIFIER", output)
        self.assertIn("Partición 1", output)
        self.assertIn("Partición 5", output)

    @patch("apps.recommendations.management.commands.evaluar_knn_supervisado.pd.read_csv")
    def test_command_validates_k_values(self, mock_read_csv):
        """Verifica validación de valores de K."""
        np.random.seed(42)
        n_samples = 20
        mock_df = pd.DataFrame({
            "id_sintetico": [f"SYN-{i:04d}" for i in range(n_samples)],
            "edad": np.random.randint(18, 65, n_samples),
            "sexo": np.random.choice(["Masculino", "Femenino"], n_samples),
            "imc": np.random.uniform(16, 35, n_samples),
            "porcentaje_grasa": np.random.uniform(7, 45, n_samples),
            "objetivo": np.random.choice(
                ["Ganancia muscular", "Fuerza", "Resistencia", "Acondicionamiento general"],
                n_samples
            ),
            "nivel_experiencia": np.random.randint(1, 4, n_samples),
            "experiencia": np.random.choice(["Principiante", "Intermedio", "Avanzado"], n_samples),
            "dias_disponibles": np.random.randint(2, 7, n_samples),
            "tiempo_sesion_minutos": np.random.choice([45, 60, 75, 90], n_samples),
            "frecuencia_previa": np.random.randint(0, 6, n_samples),
            "rutina_zpasio": np.random.choice(
                ["Full Body", "Upper/Lower"],
                n_samples
            ),
            "sesiones_planificadas": np.random.randint(12, 72, n_samples),
            "sesiones_realizadas": np.random.randint(8, 70, n_samples),
            "adherencia_porcentaje": np.random.uniform(50, 100, n_samples),
            "progreso_sintetico": np.random.uniform(0, 6, n_samples),
            "resultado_sintetico": np.random.choice(["Bajo", "Moderado", "Alto"], n_samples),
            "fuente": ["sintetico"] * n_samples,
        })
        mock_read_csv.return_value = mock_df

        from django.core.management.base import CommandError
        with self.assertRaises(CommandError):
            self.call_command(
                "evaluar_knn_supervisado",
                "--k-values", "25",
                stdout=self.out,
            )

    @patch("apps.recommendations.management.commands.evaluar_knn_supervisado.pd.read_csv")
    @patch("apps.recommendations.management.commands.evaluar_knn_supervisado.joblib.dump")
    def test_command_saves_candidate_artifacts(self, mock_joblib_dump, mock_read_csv):
        """Verifica que se guardan los artefactos candidatos cuando se solicita."""
        np.random.seed(42)
        n_samples = 100
        mock_df = pd.DataFrame({
            "id_sintetico": [f"SYN-{i:04d}" for i in range(n_samples)],
            "edad": np.random.randint(18, 65, n_samples),
            "sexo": np.random.choice(["Masculino", "Femenino"], n_samples),
            "imc": np.random.uniform(16, 35, n_samples),
            "porcentaje_grasa": np.random.uniform(7, 45, n_samples),
            "objetivo": np.random.choice(
                ["Ganancia muscular", "Fuerza", "Resistencia", "Acondicionamiento general"],
                n_samples
            ),
            "nivel_experiencia": np.random.randint(1, 4, n_samples),
            "experiencia": np.random.choice(["Principiante", "Intermedio", "Avanzado"], n_samples),
            "dias_disponibles": np.random.randint(2, 7, n_samples),
            "tiempo_sesion_minutos": np.random.choice([45, 60, 75, 90], n_samples),
            "frecuencia_previa": np.random.randint(0, 6, n_samples),
            "rutina_zpasio": np.random.choice(
                ["Full Body", "Upper/Lower", "Torso/Pierna", "Push Pull Legs", "Arnold Split"],
                n_samples
            ),
            "sesiones_planificadas": np.random.randint(12, 72, n_samples),
            "sesiones_realizadas": np.random.randint(8, 70, n_samples),
            "adherencia_porcentaje": np.random.uniform(50, 100, n_samples),
            "progreso_sintetico": np.random.uniform(0, 6, n_samples),
            "resultado_sintetico": np.random.choice(["Bajo", "Moderado", "Alto"], n_samples),
            "fuente": ["sintetico"] * n_samples,
        })
        for i, routine in enumerate(["Full Body", "Upper/Lower", "Torso/Pierna", "Push Pull Legs", "Arnold Split"]):
            mock_df.loc[i, "rutina_zpasio"] = routine

        mock_read_csv.return_value = mock_df

        try:
            self.call_command(
                "evaluar_knn_supervisado",
                "--k-values", "5", "10",
                "--weights", "uniform",
                "--save-artifacts",
                "--output-csv", "/tmp/test_results.csv",
                "--summary-csv", "/tmp/test_summary.csv",
                "--per-fold-csv", "/tmp/test_per_fold.csv",
                "--confusion-csv", "/tmp/test_confusion.csv",
                stdout=self.out,
            )
        except SystemExit:
            pass

        self.assertEqual(mock_joblib_dump.call_count, 2)
        output = self.out.getvalue()
        self.assertIn("ARTEFACTOS CANDIDATOS", output)
        self.assertIn("knn_classifier_candidate.joblib", output)
        self.assertIn("preprocesador_classifier_candidate.joblib", output)

    @patch("apps.recommendations.management.commands.evaluar_knn_supervisado.pd.read_csv")
    def test_command_outputs_comparison_table(self, mock_read_csv):
        """Verifica que se muestra la tabla comparativa de configuraciones."""
        np.random.seed(42)
        n_samples = 100
        mock_df = pd.DataFrame({
            "id_sintetico": [f"SYN-{i:04d}" for i in range(n_samples)],
            "edad": np.random.randint(18, 65, n_samples),
            "sexo": np.random.choice(["Masculino", "Femenino"], n_samples),
            "imc": np.random.uniform(16, 35, n_samples),
            "porcentaje_grasa": np.random.uniform(7, 45, n_samples),
            "objetivo": np.random.choice(
                ["Ganancia muscular", "Fuerza", "Resistencia", "Acondicionamiento general"],
                n_samples
            ),
            "nivel_experiencia": np.random.randint(1, 4, n_samples),
            "experiencia": np.random.choice(["Principiante", "Intermedio", "Avanzado"], n_samples),
            "dias_disponibles": np.random.randint(2, 7, n_samples),
            "tiempo_sesion_minutos": np.random.choice([45, 60, 75, 90], n_samples),
            "frecuencia_previa": np.random.randint(0, 6, n_samples),
            "rutina_zpasio": np.random.choice(
                ["Full Body", "Upper/Lower", "Torso/Pierna", "Push Pull Legs", "Arnold Split"],
                n_samples
            ),
            "sesiones_planificadas": np.random.randint(12, 72, n_samples),
            "sesiones_realizadas": np.random.randint(8, 70, n_samples),
            "adherencia_porcentaje": np.random.uniform(50, 100, n_samples),
            "progreso_sintetico": np.random.uniform(0, 6, n_samples),
            "resultado_sintetico": np.random.choice(["Bajo", "Moderado", "Alto"], n_samples),
            "fuente": ["sintetico"] * n_samples,
        })
        for i, routine in enumerate(["Full Body", "Upper/Lower", "Torso/Pierna", "Push Pull Legs", "Arnold Split"]):
            mock_df.loc[i, "rutina_zpasio"] = routine

        mock_read_csv.return_value = mock_df

        try:
            self.call_command(
                "evaluar_knn_supervisado",
                "--k-values", "5", "10",
                "--weights", "uniform", "distance",
                "--output-csv", "/tmp/test_results.csv",
                "--summary-csv", "/tmp/test_summary.csv",
                "--per-fold-csv", "/tmp/test_per_fold.csv",
                "--confusion-csv", "/tmp/test_confusion.csv",
                stdout=self.out,
            )
        except SystemExit:
            pass

        output = self.out.getvalue()
        self.assertIn("RESUMEN GLOBAL", output)
        self.assertIn("K = 5, weights = uniform", output)
        self.assertIn("K = 5, weights = distance", output)
        self.assertIn("K = 10, weights = uniform", output)
        self.assertIn("K = 10, weights = distance", output)
        self.assertIn("Configuración provisional elegida", output)