"""Comando para evaluar hiperparámetro K del recomendador KNN.

Compara K=5, K=10, K=15, K=20 analizando estabilidad y métricas
de las recomendaciones generadas usando los artefactos locales existentes.
"""

import csv
import logging
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from apps.profiles.models import Profile, PhysicalData
from apps.recommendations.services import KNNRecommendationService

logger = logging.getLogger(__name__)

User = get_user_model()


class Command(BaseCommand):
    """Evalúa diferentes valores de K para el recomendador KNN."""

    help = (
        "Evalúa K=5, 10, 15, 20 comparando estabilidad y métricas "
        "de recomendación usando artefactos locales."
    )

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

    CATEGORY_MAPPING = {
        "sexo": {
            "MASCULINO": "Masculino",
            "FEMENINO": "Femenino",
        },
        "objetivo": {
            "GANANCIA_MUSCULAR": "Ganancia muscular",
            "FUERZA": "Fuerza",
            "RESISTENCIA": "Resistencia",
            "ACONDICIONAMIENTO_GENERAL": "Acondicionamiento general",
        },
        "experiencia": {
            "PRINCIPIANTE": "Principiante",
            "INTERMEDIO": "Intermedio",
            "AVANZADO": "Avanzado",
        },
    }

    MIN_CASES_FOR_RECOMMENDATION = 3

    def add_arguments(self, parser):
        parser.add_argument(
            "--k-values",
            nargs="+",
            type=int,
            default=[5, 10, 15, 20],
            help="Valores de K a evaluar (default: 5 10 15 20)",
        )
        parser.add_argument(
            "--output-csv",
            type=str,
            default="knn_evaluation_results.csv",
            help="Ruta del archivo CSV de salida (default: knn_evaluation_results.csv)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Mostrar detalle por perfil y K",
        )

    def handle(self, *args, **options):
        k_values = options["k_values"]
        output_csv = options["output_csv"]
        verbose = options["verbose"]

        self.stdout.write(self.style.NOTICE("=" * 70))
        self.stdout.write(self.style.NOTICE("EVALUACIÓN DE HIPERPARÁMETRO K - RECOMENDADOR KNN ZPASIO"))
        self.stdout.write(self.style.NOTICE("=" * 70))

        self._validate_k_values(k_values)

        self.stdout.write(f"\nValores de K a evaluar: {k_values}")
        self.stdout.write(f"Archivo de salida: {output_csv}")

        eligible_profiles = self._get_eligible_profiles()
        total_users = User.objects.filter(role=User.Role.USUARIO).count()
        excluded = total_users - len(eligible_profiles)

        self.stdout.write(
            f"\nPerfiles elegibles (9 campos completos): {len(eligible_profiles)}"
        )
        self.stdout.write(f"Perfiles excluidos (datos incompletos): {excluded}")

        if not eligible_profiles:
            raise CommandError(
                "No hay perfiles con los 9 campos requeridos completos. "
                "No se puede realizar la evaluación."
            )

        self.stdout.write(
            self.style.WARNING(
                "\n⚠ IMPORTANTE: El dataset actual contiene registros SINTÉTICOS. "
                "Esta prueba evalúa el COMPORTAMIENTO y ESTABILIDAD del prototipo, "
                "NO la eficacia real de las rutinas."
            )
        )

        results = self._evaluate_all_profiles(eligible_profiles, k_values, verbose)

        self._print_summary(results, k_values)
        self._export_csv(results, k_values, output_csv)

        self.stdout.write(self.style.SUCCESS(f"\n✓ Evaluación completada. Resultados en: {output_csv}"))

    def _validate_k_values(self, k_values):
        if not k_values:
            raise CommandError("Debe especificar al menos un valor de K")

        dataset_size = self._get_dataset_size()
        for k in k_values:
            if k <= 0:
                raise CommandError(f"K debe ser positivo: {k}")
            if k > dataset_size:
                raise CommandError(
                    f"K={k} supera el tamaño del dataset ({dataset_size} registros)"
                )

    def _get_dataset_size(self):
        paths = self._get_model_paths()
        import pandas as pd

        df = pd.read_csv(paths["dataset"])
        return len(df)

    def _get_model_paths(self):
        base_dir = Path(settings.BASE_DIR)
        ml_dir = base_dir / "apps" / "recommendations" / "ml"
        return {
            "preprocesador": ml_dir / "preprocesador_zpasio.joblib",
            "knn": ml_dir / "knn_zpasio.joblib",
            "dataset": ml_dir / "dataset_modelo_zpasio.csv",
        }

    def _get_eligible_profiles(self):
        users = User.objects.filter(role=User.Role.USUARIO).select_related(
            "profile", "physical_data"
        )
        eligible = []

        for user in users:
            profile = getattr(user, "profile", None)
            physical = getattr(user, "physical_data", None)

            if not profile or not physical:
                continue

            features = self._extract_user_features(user)
            missing = self._validate_features(features)

            if not missing:
                eligible.append(
                    {
                        "user_id": user.id,
                        "username": user.username,
                        "features": features,
                    }
                )

        return eligible

    def _extract_user_features(self, user):
        profile = user.profile
        physical = user.physical_data

        features = {}

        features["edad"] = profile.age
        features["sexo"] = profile.gender
        features["objetivo"] = profile.goal
        features["experiencia"] = profile.experience_level
        features["dias_disponibles"] = profile.available_days_per_week
        features["tiempo_sesion_minutos"] = profile.session_time_available_minutes
        features["frecuencia_previa"] = profile.previous_training_days_per_week

        features["imc"] = float(physical.bmi) if physical.bmi else None
        features["porcentaje_grasa"] = (
            float(physical.body_fat_percentage)
            if physical.body_fat_percentage
            else None
        )

        return features

    def _validate_features(self, features):
        missing = []
        for field in self.REQUIRED_FIELDS:
            value = features.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                missing.append(field)
        return missing

    def _normalize_categories(self, user_data):
        normalized = user_data.copy()
        for field, mapping in self.CATEGORY_MAPPING.items():
            value = normalized.get(field)
            if value in mapping:
                normalized[field] = mapping[value]
        return normalized

    def _evaluate_all_profiles(self, eligible_profiles, k_values, verbose):
        import joblib
        import pandas as pd
        import numpy as np

        paths = self._get_model_paths()

        preprocesador = joblib.load(paths["preprocesador"])
        knn = joblib.load(paths["knn"])
        dataset = pd.read_csv(paths["dataset"])

        dataset_size = len(dataset)
        self.stdout.write(f"\nDataset size: {dataset_size} registros")
        self.stdout.write(f"KNN default n_neighbors: {knn.n_neighbors}")

        all_results = []

        for idx, profile_data in enumerate(eligible_profiles, 1):
            anon_id = f"Perfil {idx:03d}"
            user_features = profile_data["features"]

            self.stdout.write(
                f"\n{'─' * 50}"
            )
            self.stdout.write(f"Evaluando {anon_id} ({profile_data['username']})...")

            normalized = self._normalize_categories(user_features)
            import pandas as pd
            user_df = pd.DataFrame([normalized], columns=self.REQUIRED_FIELDS)
            user_processed = preprocesador.transform(user_df)

            for k in k_values:
                result = self._evaluate_single_k(
                    user_processed,
                    dataset,
                    knn,
                    k,
                    anon_id,
                    profile_data["username"],
                )
                all_results.append(result)

                if verbose:
                    self._print_result(result)

        return all_results

    def _evaluate_single_k(
        self, user_processed, dataset, knn, k, anon_id, real_username
    ):
        import numpy as np

        try:
            distances, indices = knn.kneighbors(user_processed, n_neighbors=k)
        except Exception as e:
            return {
                "anon_id": anon_id,
                "real_username": real_username,
                "k": k,
                "success": False,
                "error": f"Error en kneighbors: {e}",
                "recommended_routine": None,
                "cases": 0,
                "avg_distance": None,
                "avg_adherence": None,
                "weighted_score": None,
                "neighbors_queried": k,
                "valid_candidates": 0,
            }

        distances = distances[0]
        indices = indices[0]

        if len(dataset) < k:
            return {
                "anon_id": anon_id,
                "real_username": real_username,
                "k": k,
                "success": False,
                "error": f"Dataset ({len(dataset)}) < K ({k})",
                "recommended_routine": None,
                "cases": 0,
                "avg_distance": None,
                "avg_adherence": None,
                "weighted_score": None,
                "neighbors_queried": k,
                "valid_candidates": 0,
            }

        neighbors = dataset.iloc[indices].copy()
        neighbors["distancia_knn"] = distances

        neighbors["peso_similitud"] = 1 / (neighbors["distancia_knn"] + 1e-6)
        neighbors["progreso_ponderado"] = (
            neighbors["progreso_sintetico"] * neighbors["peso_similitud"]
        )

        grouped = (
            neighbors.groupby("rutina_zpasio")
            .agg(
                casos=("rutina_zpasio", "count"),
                distancia_promedio=("distancia_knn", "mean"),
                adherencia_promedio=("adherencia_porcentaje", "mean"),
                suma_progreso=("progreso_ponderado", "sum"),
                suma_pesos=("peso_similitud", "sum"),
            )
            .reset_index()
        )

        grouped["puntaje_resultado"] = (
            grouped["suma_progreso"] / grouped["suma_pesos"]
        ).round(4)

        valid_routines = grouped[
            grouped["casos"] >= self.MIN_CASES_FOR_RECOMMENDATION
        ].copy()

        if valid_routines.empty:
            return {
                "anon_id": anon_id,
                "real_username": real_username,
                "k": k,
                "success": False,
                "error": "insufficient_evidence",
                "recommended_routine": None,
                "cases": 0,
                "avg_distance": None,
                "avg_adherence": None,
                "weighted_score": None,
                "neighbors_queried": k,
                "valid_candidates": len(grouped),
            }

        best = valid_routines.sort_values("puntaje_resultado", ascending=False).iloc[0]

        return {
            "anon_id": anon_id,
            "real_username": real_username,
            "k": k,
            "success": True,
            "error": None,
            "recommended_routine": best["rutina_zpasio"],
            "cases": int(best["casos"]),
            "avg_distance": round(best["distancia_promedio"], 4),
            "avg_adherence": round(best["adherencia_promedio"], 1),
            "weighted_score": round(best["puntaje_resultado"], 4),
            "neighbors_queried": k,
            "valid_candidates": len(valid_routines),
        }

    def _print_result(self, result):
        if result["success"]:
            self.stdout.write(
                f"  K={result['k']:2d}: {result['recommended_routine']:20s} | "
                f"casos={result['cases']:2d} | "
                f"dist_media={result['avg_distance']:.4f} | "
                f"adherencia={result['avg_adherence']:.1f}% | "
                f"score={result['weighted_score']:.4f} | "
                f"candidatas={result['valid_candidates']}"
            )
        else:
            self.stdout.write(
                f"  K={result['k']:2d}: SIN RECOMENDACIÓN ({result['error']}) | "
                f"candidatas={result['valid_candidates']}"
            )

    def _print_summary(self, results, k_values):
        self.stdout.write(f"\n{'=' * 70}")
        self.stdout.write(self.style.NOTICE("RESUMEN POR VALOR DE K"))
        self.stdout.write(f"{'=' * 70}")

        profiles_evaluated = len(set(r["anon_id"] for r in results))

        for k in k_values:
            k_results = [r for r in results if r["k"] == k]
            successful = [r for r in k_results if r["success"]]

            self.stdout.write(f"\n--- K = {k} ---")
            self.stdout.write(f"  Perfiles evaluados: {len(k_results)}")
            self.stdout.write(
                f"  Con recomendación válida: {len(successful)} "
                f"({len(successful)/len(k_results)*100:.1f}%)"
            )
            self.stdout.write(f"  Sin evidencia suficiente: {len(k_results) - len(successful)}")

            if successful:
                routine_counts = {}
                for r in successful:
                    routine_counts[r["recommended_routine"]] = (
                        routine_counts.get(r["recommended_routine"], 0) + 1
                    )
                self.stdout.write("  Rutinas recomendadas:")
                for routine, count in sorted(
                    routine_counts.items(), key=lambda x: -x[1]
                ):
                    self.stdout.write(
                        f"    {routine}: {count} perfiles "
                        f"({count/len(successful)*100:.1f}%)"
                    )

        self._print_stability_analysis(results, k_values, profiles_evaluated)

    def _print_stability_analysis(self, results, k_values, n_profiles):
        self.stdout.write(f"\n{'=' * 70}")
        self.stdout.write(self.style.NOTICE("ANÁLISIS DE ESTABILIDAD"))
        self.stdout.write(f"{'=' * 70}")

        for i in range(n_profiles):
            anon_id = f"Perfil {i+1:03d}"
            profile_results = [r for r in results if r["anon_id"] == anon_id]

            successful_recs = [r for r in profile_results if r["success"]]
            if not successful_recs:
                self.stdout.write(f"\n{anon_id}: Sin recomendaciones válidas en ningún K")
                continue

            routines = [r["recommended_routine"] for r in successful_recs]
            unique_routines = set(routines)

            self.stdout.write(f"\n{anon_id}:")
            for r in successful_recs:
                self.stdout.write(
                    f"  K={r['k']:2d}: {r['recommended_routine']:20s} "
                    f"(casos={r['cases']}, score={r['weighted_score']:.4f})"
                )

            if len(unique_routines) == 1:
                self.stdout.write(
                    self.style.SUCCESS(f"  → ESTABLE: misma rutina en todos los K")
                )
            else:
                self.stdout.write(
                    self.style.WARNING(
                        f"  → INESTABLE: {len(unique_routines)} rutinas distintas: "
                        f"{', '.join(sorted(unique_routines))}"
                    )
                )

    def _export_csv(self, results, k_values, output_csv):
        fieldnames = [
            "anon_id",
            "k",
            "recommended_routine",
            "cases",
            "avg_distance",
            "avg_adherence",
            "weighted_score",
            "neighbors_queried",
            "valid_candidates",
            "success",
            "error",
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                row = {
                    "anon_id": r["anon_id"],
                    "k": r["k"],
                    "recommended_routine": r["recommended_routine"] or "",
                    "cases": r["cases"] if r["success"] else "",
                    "avg_distance": r["avg_distance"] if r["success"] else "",
                    "avg_adherence": r["avg_adherence"] if r["success"] else "",
                    "weighted_score": r["weighted_score"] if r["success"] else "",
                    "neighbors_queried": r["neighbors_queried"],
                    "valid_candidates": r["valid_candidates"],
                    "success": r["success"],
                    "error": r["error"] or "",
                }
                writer.writerow(row)

        self.stdout.write(f"\nCSV exportado: {output_csv}")