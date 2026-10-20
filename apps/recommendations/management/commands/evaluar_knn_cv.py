"""Evaluación retrospectiva del hiperparámetro K con validación cruzada estratificada.

Compara K=5, 10, 15, 20 usando el dataset experimental sintético.
Implementa validación cruzada de 5 particiones sin fuga de información.
"""

import csv
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Evalúa K=5, 10, 15, 20 con validación cruzada estratificada 5-fold."""

    help = (
        "Evaluación retrospectiva KNN con validación cruzada estratificada. "
        "Compara K=5, 10, 15, 20 sin fuga de información."
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

    MIN_CASES_FOR_RECOMMENDATION = 3
    N_SPLITS = 5
    RANDOM_STATE = 42

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
            default="knn_cv_evaluation_results.csv",
            help="Ruta del archivo CSV de resultados detallados (default: knn_cv_evaluation_results.csv)",
        )
        parser.add_argument(
            "--summary-csv",
            type=str,
            default="knn_cv_evaluation_summary.csv",
            help="Ruta del archivo CSV de resumen por K (default: knn_cv_evaluation_summary.csv)",
        )
        parser.add_argument(
            "--stability-csv",
            type=str,
            default="knn_cv_stability_analysis.csv",
            help="Ruta del archivo CSV de análisis de estabilidad (default: knn_cv_stability_analysis.csv)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Mostrar detalle por partición y K",
        )

    def handle(self, *args, **options):
        k_values = options["k_values"]
        output_csv = options["output_csv"]
        summary_csv = options["summary_csv"]
        stability_csv = options["stability_csv"]
        verbose = options["verbose"]

        self._validate_k_values(k_values)

        self.stdout.write(self.style.NOTICE("=" * 70))
        self.stdout.write(self.style.NOTICE("EVALUACIÓN RETROSPECTIVA KNN - VALIDACIÓN CRUZADA 5-FOLD"))
        self.stdout.write(self.style.NOTICE("=" * 70))

        self.stdout.write(f"\nValores de K a evaluar: {k_values}")
        self.stdout.write(f"Archivo resultados detallados: {output_csv}")
        self.stdout.write(f"Archivo resumen por K: {summary_csv}")
        self.stdout.write(f"Archivo análisis estabilidad: {stability_csv}")

        self.stdout.write(
            self.style.WARNING(
                "\n⚠ IMPORTANTE: El dataset contiene registros SINTÉTICOS. "
                "Esta evaluación mide el COMPORTAMIENTO del prototipo frente a "
                "etiquetas sintéticas. NO demuestra eficacia de rutinas ni "
                "valida rendimiento en clientes reales de Zpasio."
            )
        )

        dataset_path = self._get_dataset_path()
        df = pd.read_csv(dataset_path)

        self.stdout.write(f"\nDataset cargado: {len(df)} registros")
        self.stdout.write(f"Rutinas únicas: {df['rutina_zpasio'].nunique()}")
        self.stdout.write(f"Distribución de rutinas:\n{df['rutina_zpasio'].value_counts().to_string()}")

        X = df[self.REQUIRED_FIELDS].copy()
        y = df["rutina_zpasio"].copy()

        skf = StratifiedKFold(n_splits=self.N_SPLITS, shuffle=True, random_state=self.RANDOM_STATE)

        all_results = []
        fold_summaries = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            self.stdout.write(f"\n{'=' * 70}")
            self.stdout.write(self.style.NOTICE(f"PARTICIÓN {fold_idx}/{self.N_SPLITS}"))
            self.stdout.write(f"{'=' * 70}")

            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)

            self.stdout.write(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

            preprocessor = self._build_preprocessor()
            preprocessor.fit(X_train)

            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            knn = NearestNeighbors(
                n_neighbors=max(k_values),
                metric="euclidean",
                algorithm="brute",
            )
            knn.fit(X_train_processed)

            fold_results = self._evaluate_fold(
                X_test_processed,
                X_test,
                y_test,
                X_train,
                y_train,
                knn,
                preprocessor,
                k_values,
                fold_idx,
                verbose,
            )

            all_results.extend(fold_results)

            fold_summary = self._compute_fold_summary(fold_results, k_values, fold_idx)
            fold_summaries.append(fold_summary)

            self._print_fold_summary(fold_summary, k_values)

        global_summary = self._compute_global_summary(all_results, k_values)
        self._print_global_summary(global_summary, k_values)

        stability_analysis = self._compute_stability_analysis(all_results, k_values)
        self._print_stability_analysis(stability_analysis, k_values)

        self._export_detailed_csv(all_results, output_csv)
        self._export_summary_csv(global_summary, fold_summaries, summary_csv)
        self._export_stability_csv(stability_analysis, stability_csv)

        self.stdout.write(self.style.SUCCESS(f"\n✓ Evaluación completada."))
        self.stdout.write(f"  Resultados detallados: {output_csv}")
        self.stdout.write(f"  Resumen por K: {summary_csv}")
        self.stdout.write(f"  Análisis estabilidad: {stability_csv}")

    def _get_dataset_path(self):
        base_dir = Path(settings.BASE_DIR)
        return base_dir / "apps" / "recommendations" / "ml" / "dataset_modelo_zpasio.csv"

    def _validate_k_values(self, k_values):
        if not k_values:
            raise CommandError("Debe especificar al menos un valor de K")

        df = pd.read_csv(self._get_dataset_path())
        dataset_size = len(df)

        for k in k_values:
            if k <= 0:
                raise CommandError(f"K debe ser positivo: {k}")
            if k > dataset_size:
                raise CommandError(
                    f"K={k} supera el tamaño del dataset ({dataset_size} registros)"
                )
            if k < self.MIN_CASES_FOR_RECOMMENDATION:
                self.stdout.write(
                    self.style.WARNING(
                        f"Advertencia: K={k} < mínimo de casos ({self.MIN_CASES_FOR_RECOMMENDATION}). "
                        "Puede no generar recomendaciones válidas."
                    )
                )

    def _build_preprocessor(self):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.REQUIRED_NUMERIC_FIELDS),
                ("cat", categorical_transformer, self.REQUIRED_CATEGORICAL_FIELDS),
            ],
            remainder="drop",
        )

        return preprocessor

    def _evaluate_fold(
        self,
        X_test_processed,
        X_test_raw,
        y_test,
        X_train_raw,
        y_train,
        knn,
        preprocessor,
        k_values,
        fold_idx,
        verbose,
    ):
        full_dataset = pd.read_csv(self._get_dataset_path())
        train_indices = full_dataset.index.isin(X_train_raw.index)
        train_dataset = full_dataset[train_indices].copy()
        train_dataset = train_dataset.reset_index(drop=True)

        results = []

        for test_idx_local, (test_row_processed, test_row_raw, true_routine) in enumerate(
            zip(X_test_processed, X_test_raw.itertuples(index=False), y_test)
        ):
            test_row_df = pd.DataFrame([test_row_raw], columns=self.REQUIRED_FIELDS)
            test_row_processed = preprocessor.transform(test_row_df)

            max_k = max(k_values)
            distances, indices = knn.kneighbors(test_row_processed, n_neighbors=max_k)

            distances = distances[0]
            indices = indices[0]

            for k in k_values:
                k_distances = distances[:k]
                k_indices = indices[:k]

                neighbors = train_dataset.iloc[k_indices].copy()
                neighbors["distancia_knn"] = k_distances

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
                    result = {
                        "fold": fold_idx,
                        "test_index": test_idx_local,
                        "k": k,
                        "success": False,
                        "error": "insufficient_evidence",
                        "recommended_routine": None,
                        "true_routine": true_routine,
                        "cases": 0,
                        "avg_distance": None,
                        "avg_adherence": None,
                        "weighted_score": None,
                        "neighbors_queried": k,
                        "valid_candidates": len(grouped),
                        "match": False,
                    }
                else:
                    best = valid_routines.sort_values(
                        "puntaje_resultado", ascending=False
                    ).iloc[0]

                    recommended = best["rutina_zpasio"]
                    match = recommended == true_routine

                    result = {
                        "fold": fold_idx,
                        "test_index": test_idx_local,
                        "k": k,
                        "success": True,
                        "error": None,
                        "recommended_routine": recommended,
                        "true_routine": true_routine,
                        "cases": int(best["casos"]),
                        "avg_distance": round(best["distancia_promedio"], 4),
                        "avg_adherence": round(best["adherencia_promedio"], 1),
                        "weighted_score": round(best["puntaje_resultado"], 4),
                        "neighbors_queried": k,
                        "valid_candidates": len(valid_routines),
                        "match": match,
                    }

                results.append(result)

                if verbose and result["success"]:
                    self.stdout.write(
                        f"  Fold {fold_idx} Test {test_idx_local} K={k}: "
                        f"{result['recommended_routine']:20s} | "
                        f"casos={result['cases']:2d} | "
                        f"match={result['match']} | "
                        f"true={true_routine}"
                    )
                elif verbose and not result["success"]:
                    self.stdout.write(
                        f"  Fold {fold_idx} Test {test_idx_local} K={k}: "
                        f"SIN RECOMENDACIÓN ({result['error']}) | "
                        f"true={true_routine}"
                    )

        return results

    def _compute_fold_summary(self, fold_results, k_values, fold_idx):
        fold_summary = {"fold": fold_idx}

        for k in k_values:
            k_results = [r for r in fold_results if r["k"] == k]
            successful = [r for r in k_results if r["success"]]
            matched = [r for r in successful if r["match"]]

            fold_summary[f"k{k}_evaluated"] = len(k_results)
            fold_summary[f"k{k}_with_recommendation"] = len(successful)
            fold_summary[f"k{k}_coverage_pct"] = (
                len(successful) / len(k_results) * 100 if k_results else 0
            )
            fold_summary[f"k{k}_matches"] = len(matched)
            fold_summary[f"k{k}_accuracy_pct"] = (
                len(matched) / len(successful) * 100 if successful else 0
            )
            fold_summary[f"k{k}_no_evidence"] = len(k_results) - len(successful)

            if successful:
                routine_counts = {}
                for r in successful:
                    routine_counts[r["recommended_routine"]] = (
                        routine_counts.get(r["recommended_routine"], 0) + 1
                    )
                fold_summary[f"k{k}_routine_distribution"] = routine_counts

        return fold_summary

    def _compute_global_summary(self, all_results, k_values):
        global_summary = {}

        for k in k_values:
            k_results = [r for r in all_results if r["k"] == k]
            successful = [r for r in k_results if r["success"]]
            matched = [r for r in successful if r["match"]]

            global_summary[k] = {
                "total_evaluated": len(k_results),
                "with_recommendation": len(successful),
                "coverage_pct": len(successful) / len(k_results) * 100 if k_results else 0,
                "matches": len(matched),
                "accuracy_pct": len(matched) / len(successful) * 100 if successful else 0,
                "no_evidence": len(k_results) - len(successful),
            }

            if successful:
                routine_counts = {}
                for r in successful:
                    routine_counts[r["recommended_routine"]] = (
                        routine_counts.get(r["recommended_routine"], 0) + 1
                    )
                global_summary[k]["routine_distribution"] = routine_counts

        return global_summary

    def _compute_stability_analysis(self, all_results, k_values):
        test_cases = {}
        for r in all_results:
            key = (r["fold"], r["test_index"])
            if key not in test_cases:
                test_cases[key] = {}
            if r["success"]:
                test_cases[key][r["k"]] = r["recommended_routine"]

        stability_results = []

        for (fold_idx, test_idx), k_recs in test_cases.items():
            if len(k_recs) < 2:
                continue

            routines = list(k_recs.values())
            unique_routines = set(routines)

            row = {
                "fold": fold_idx,
                "test_index": test_idx,
                "n_k_with_recommendation": len(k_recs),
                "unique_routines": len(unique_routines),
                "stable": len(unique_routines) == 1,
            }

            for k in k_values:
                row[f"k{k}_routine"] = k_recs.get(k, None)

            stability_results.append(row)

        return stability_results

    def _print_fold_summary(self, fold_summary, k_values):
        self.stdout.write(f"\n--- Resumen Partición {fold_summary['fold']} ---")

        for k in k_values:
            cov = fold_summary.get(f"k{k}_coverage_pct", 0)
            acc = fold_summary.get(f"k{k}_accuracy_pct", 0)
            matches = fold_summary.get(f"k{k}_matches", 0)
            no_ev = fold_summary.get(f"k{k}_no_evidence", 0)
            evaluated = fold_summary.get(f"k{k}_evaluated", 0)

            self.stdout.write(
                f"  K={k:2d}: Evaluados={evaluated:3d} | "
                f"Con rec={fold_summary.get(f'k{k}_with_recommendation', 0):3d} "
                f"({cov:.1f}%) | "
                f"Aciertos={matches:3d} ({acc:.1f}%) | "
                f"Sin evidencia={no_ev:3d}"
            )

            dist = fold_summary.get(f"k{k}_routine_distribution", {})
            if dist:
                dist_str = ", ".join(
                    f"{r}:{c}" for r, c in sorted(dist.items(), key=lambda x: -x[1])
                )
                self.stdout.write(f"    Distribución: {dist_str}")

    def _print_global_summary(self, global_summary, k_values):
        self.stdout.write(f"\n{'=' * 70}")
        self.stdout.write(self.style.NOTICE("RESUMEN GLOBAL"))
        self.stdout.write(f"{'=' * 70}")

        for k in k_values:
            gs = global_summary[k]
            self.stdout.write(f"\n--- K = {k} ---")
            self.stdout.write(
                f"  Total evaluados: {gs['total_evaluated']}"
            )
            self.stdout.write(
                f"  Con recomendación válida: {gs['with_recommendation']} "
                f"({gs['coverage_pct']:.1f}%)"
            )
            self.stdout.write(
                f"  Coincidencias con rutina asignada: {gs['matches']} "
                f"({gs['accuracy_pct']:.1f}% de los que tienen recomendación)"
            )
            self.stdout.write(
                f"  Sin evidencia suficiente: {gs['no_evidence']}"
            )

            if "routine_distribution" in gs:
                self.stdout.write("  Distribución de rutinas recomendadas:")
                for routine, count in sorted(
                    gs["routine_distribution"].items(), key=lambda x: -x[1]
                ):
                    pct = count / gs["with_recommendation"] * 100
                    self.stdout.write(f"    {routine}: {count} ({pct:.1f}%)")

    def _print_stability_analysis(self, stability_results, k_values):
        self.stdout.write(f"\n{'=' * 70}")
        self.stdout.write(self.style.NOTICE("ANÁLISIS DE ESTABILIDAD ENTRE K"))
        self.stdout.write(f"{'=' * 70}")

        total_cases = len(stability_results)
        stable_cases = sum(1 for r in stability_results if r["stable"])

        self.stdout.write(f"\nCasos con recomendación en ≥2 K: {total_cases}")
        self.stdout.write(f"Casos estables (misma rutina en todos los K): {stable_cases}")
        self.stdout.write(f"Casos inestables: {total_cases - stable_cases}")

        if total_cases > 0:
            self.stdout.write(f"Porcentaje estabilidad: {stable_cases/total_cases*100:.1f}%")

        self.stdout.write("\nDetalle por caso (muestra primeras 20):")
        for i, r in enumerate(stability_results[:20]):
            routines = []
            for k in k_values:
                routine = r.get(f"k{k}_routine")
                if routine:
                    routines.append(f"K={k}:{routine}")

            status = self.style.SUCCESS("ESTABLE") if r["stable"] else self.style.WARNING("INESTABLE")
            self.stdout.write(f"  Fold {r['fold']} Test {r['test_index']}: {' | '.join(routines)} → {status}")

        if len(stability_results) > 20:
            self.stdout.write(f"  ... y {len(stability_results) - 20} casos más")

        pair_comparisons = self._compute_pairwise_stability(stability_results, k_values)
        self.stdout.write("\nComparación par a par (casos donde ambos K tienen recomendación):")
        for (k1, k2), stats in pair_comparisons.items():
            self.stdout.write(
                f"  K={k1} vs K={k2}: "
                f"iguales={stats['same']}, "
                f"diferentes={stats['different']}, "
                f"total comparados={stats['total']}, "
                f"% iguales={stats['same_pct']:.1f}%"
            )

    def _compute_pairwise_stability(self, stability_results, k_values):
        pairs = {}
        for i, k1 in enumerate(k_values):
            for k2 in k_values[i+1:]:
                pairs[(k1, k2)] = {"same": 0, "different": 0, "total": 0}

        for r in stability_results:
            for (k1, k2), stats in pairs.items():
                r1 = r.get(f"k{k1}_routine")
                r2 = r.get(f"k{k2}_routine")
                if r1 and r2:
                    stats["total"] += 1
                    if r1 == r2:
                        stats["same"] += 1
                    else:
                        stats["different"] += 1

        for (k1, k2), stats in pairs.items():
            stats["same_pct"] = (
                stats["same"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )

        return pairs

    def _export_detailed_csv(self, all_results, output_csv):
        fieldnames = [
            "fold",
            "test_index",
            "k",
            "success",
            "error",
            "recommended_routine",
            "true_routine",
            "match",
            "cases",
            "avg_distance",
            "avg_adherence",
            "weighted_score",
            "neighbors_queried",
            "valid_candidates",
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in all_results:
                row = {
                    "fold": r["fold"],
                    "test_index": r["test_index"],
                    "k": r["k"],
                    "success": r["success"],
                    "error": r["error"] or "",
                    "recommended_routine": r["recommended_routine"] or "",
                    "true_routine": r["true_routine"] or "",
                    "match": r["match"],
                    "cases": r["cases"] if r["success"] else "",
                    "avg_distance": r["avg_distance"] if r["success"] else "",
                    "avg_adherence": r["avg_adherence"] if r["success"] else "",
                    "weighted_score": r["weighted_score"] if r["success"] else "",
                    "neighbors_queried": r["neighbors_queried"],
                    "valid_candidates": r["valid_candidates"],
                }
                writer.writerow(row)

    def _export_summary_csv(self, global_summary, fold_summaries, summary_csv):
        fieldnames = [
            "scope",
            "fold",
            "k",
            "total_evaluated",
            "with_recommendation",
            "coverage_pct",
            "matches",
            "accuracy_pct",
            "no_evidence",
        ]

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for fold_summary in fold_summaries:
                fold = fold_summary["fold"]
                for k in sorted(global_summary.keys()):
                    writer.writerow({
                        "scope": "fold",
                        "fold": fold,
                        "k": k,
                        "total_evaluated": fold_summary.get(f"k{k}_evaluated", 0),
                        "with_recommendation": fold_summary.get(f"k{k}_with_recommendation", 0),
                        "coverage_pct": round(fold_summary.get(f"k{k}_coverage_pct", 0), 2),
                        "matches": fold_summary.get(f"k{k}_matches", 0),
                        "accuracy_pct": round(fold_summary.get(f"k{k}_accuracy_pct", 0), 2),
                        "no_evidence": fold_summary.get(f"k{k}_no_evidence", 0),
                    })

            for k in sorted(global_summary.keys()):
                gs = global_summary[k]
                writer.writerow({
                    "scope": "global",
                    "fold": "",
                    "k": k,
                    "total_evaluated": gs["total_evaluated"],
                    "with_recommendation": gs["with_recommendation"],
                    "coverage_pct": round(gs["coverage_pct"], 2),
                    "matches": gs["matches"],
                    "accuracy_pct": round(gs["accuracy_pct"], 2),
                    "no_evidence": gs["no_evidence"],
                })

    def _export_stability_csv(self, stability_results, stability_csv):
        fieldnames = [
            "fold",
            "test_index",
            "n_k_with_recommendation",
            "unique_routines",
            "stable",
        ]
        for k in [5, 10, 15, 20]:
            fieldnames.append(f"k{k}_routine")

        with open(stability_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in stability_results:
                row = {
                    "fold": r["fold"],
                    "test_index": r["test_index"],
                    "n_k_with_recommendation": r["n_k_with_recommendation"],
                    "unique_routines": r["unique_routines"],
                    "stable": r["stable"],
                }
                for k in [5, 10, 15, 20]:
                    row[f"k{k}_routine"] = r.get(f"k{k}_routine", "")
                writer.writerow(row)