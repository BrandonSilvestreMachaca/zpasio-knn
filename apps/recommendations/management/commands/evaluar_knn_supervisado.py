"""Evaluación supervisada de KNeighborsClassifier con validación cruzada estratificada.

Compara K=5, 10, 15, 20 con weights='uniform' y 'distance'.
Usa Pipeline con ColumnTransformer para preprocesamiento correcto sin fuga de información.
"""

import csv
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Evalúa KNeighborsClassifier con validación cruzada estratificada 5-fold."""

    help = (
        "Evaluación supervisada KNN Classifier con CV estratificada. "
        "Compara K=5, 10, 15, 20 con weights=uniform/distance."
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

    TARGET_FIELD = "rutina_zpasio"

    N_SPLITS = 5
    RANDOM_STATE = 42
    K_VALUES = [5, 10, 15, 20]
    WEIGHTS_OPTIONS = ["uniform", "distance"]

    def add_arguments(self, parser):
        parser.add_argument(
            "--k-values",
            nargs="+",
            type=int,
            default=self.K_VALUES,
            help="Valores de K a evaluar (default: 5 10 15 20)",
        )
        parser.add_argument(
            "--weights",
            nargs="+",
            type=str,
            default=self.WEIGHTS_OPTIONS,
            choices=self.WEIGHTS_OPTIONS,
            help="Opciones de weights para KNN (default: uniform distance)",
        )
        parser.add_argument(
            "--output-csv",
            type=str,
            default="knn_supervised_evaluation_results.csv",
            help="Ruta del archivo CSV de resultados detallados",
        )
        parser.add_argument(
            "--summary-csv",
            type=str,
            default="knn_supervised_evaluation_summary.csv",
            help="Ruta del archivo CSV de resumen por configuración",
        )
        parser.add_argument(
            "--per-fold-csv",
            type=str,
            default="knn_supervised_per_fold.csv",
            help="Ruta del archivo CSV de resultados por partición",
        )
        parser.add_argument(
            "--confusion-csv",
            type=str,
            default="knn_supervised_confusion_matrices.csv",
            help="Ruta del archivo CSV de matrices de confusión",
        )
        parser.add_argument(
            "--save-artifacts",
            action="store_true",
            help="Guardar clasificador y preprocesador candidatos entrenados con dataset completo",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Mostrar detalle por partición y configuración",
        )

    def handle(self, *args, **options):
        k_values = options["k_values"]
        weights_options = options["weights"]
        output_csv = options["output_csv"]
        summary_csv = options["summary_csv"]
        per_fold_csv = options["per_fold_csv"]
        confusion_csv = options["confusion_csv"]
        save_artifacts = options["save_artifacts"]
        verbose = options["verbose"]

        self._validate_k_values(k_values)

        self.stdout.write(self.style.NOTICE("=" * 80))
        self.stdout.write(
            self.style.NOTICE("EVALUACIÓN SUPERVISADA KNN CLASSIFIER - VALIDACIÓN CRUZADA 5-FOLD")
        )
        self.stdout.write(self.style.NOTICE("=" * 80))

        self.stdout.write(f"\nValores de K a evaluar: {k_values}")
        self.stdout.write(f"Opciones de weights: {weights_options}")
        self.stdout.write(f"Archivo resultados detallados: {output_csv}")
        self.stdout.write(f"Archivo resumen: {summary_csv}")
        self.stdout.write(f"Archivo por partición: {per_fold_csv}")
        self.stdout.write(f"Archivo matrices confusión: {confusion_csv}")
        if save_artifacts:
            self.stdout.write(self.style.NOTICE("Guardar artefactos candidatos: SÍ"))

        self.stdout.write(
            self.style.WARNING(
                "\n⚠ IMPORTANTE: El dataset contiene registros SINTÉTICOS. "
                "Esta evaluación mide la capacidad de PREDECIR etiquetas sintéticas "
                "de rutinas. NO demuestra eficacia de rutinas ni valida rendimiento "
                "en clientes reales de Zpasio."
            )
        )

        dataset_path = self._get_dataset_path()
        df = pd.read_csv(dataset_path)

        self.stdout.write(f"\nDataset cargado: {len(df)} registros")
        self.stdout.write(f"Rutinas únicas (clases): {df[self.TARGET_FIELD].nunique()}")
        self.stdout.write(f"Distribución de clases:\n{df[self.TARGET_FIELD].value_counts().to_string()}")

        class_counts = df[self.TARGET_FIELD].value_counts()
        min_class_count = class_counts.min()
        self.stdout.write(f"\nClase minoritaria: {class_counts.idxmin()} ({min_class_count} muestras)")

        if min_class_count < self.N_SPLITS:
            self.stdout.write(
                self.style.ERROR(
                    f"\n⚠ ADVERTENCIA: La clase minoritaria tiene {min_class_count} muestras, "
                    f"menos que n_splits={self.N_SPLITS}. "
                    "StratifiedKFold fallará o producirá particiones sin esa clase. "
                    "Considerar reducir n_splits o usar LeaveOneOut."
                )
            )

        X = df[self.REQUIRED_FIELDS].copy()
        y = df[self.TARGET_FIELD].copy()

        skf = StratifiedKFold(n_splits=self.N_SPLITS, shuffle=True, random_state=self.RANDOM_STATE)

        all_configs = [(k, w) for k in k_values for w in weights_options]
        all_results = []
        per_fold_results = []
        confusion_matrices = {}

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            self.stdout.write(f"\n{'=' * 80}")
            self.stdout.write(self.style.NOTICE(f"PARTICIÓN {fold_idx}/{self.N_SPLITS}"))
            self.stdout.write(f"{'=' * 80}")

            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)

            self.stdout.write(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
            self.stdout.write(f"Distribución train:\n{y_train.value_counts().to_string()}")
            self.stdout.write(f"Distribución test:\n{y_test.value_counts().to_string()}")

            baseline = y_train.mode()[0]
            baseline_acc = (y_test == baseline).mean()
            self.stdout.write(f"Baseline (clase más frecuente en train): {baseline} | Accuracy: {baseline_acc:.4f}")

            fold_config_results = {}

            for k, weights in all_configs:
                self.stdout.write(f"\n  --- K={k}, weights={weights} ---")

                pipeline = self._build_pipeline(k, weights)
                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
                rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

                cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

                result = {
                    "fold": fold_idx,
                    "k": k,
                    "weights": weights,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "precision_macro": prec_macro,
                    "recall_macro": rec_macro,
                    "f1_macro": f1_macro,
                    "baseline_accuracy": baseline_acc,
                    "baseline_class": baseline,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
                all_results.append(result)
                fold_config_results[(k, weights)] = result

                per_fold_results.append({
                    "fold": fold_idx,
                    "k": k,
                    "weights": weights,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "precision_macro": prec_macro,
                    "recall_macro": rec_macro,
                    "f1_macro": f1_macro,
                    "baseline_accuracy": baseline_acc,
                })

                confusion_matrices[(fold_idx, k, weights)] = cm

                if verbose:
                    self.stdout.write(f"    Accuracy: {acc:.4f} | Balanced: {bal_acc:.4f}")
                    self.stdout.write(f"    Precision macro: {prec_macro:.4f} | Recall macro: {rec_macro:.4f} | F1 macro: {f1_macro:.4f}")
                    self.stdout.write(f"    Baseline: {baseline_acc:.4f}")

            self._print_fold_summary(fold_config_results, all_configs, fold_idx)

        global_summary = self._compute_global_summary(all_results, all_configs)
        self._print_global_summary(global_summary, all_configs)

        best_config = self._select_best_config(global_summary)
        self.stdout.write(self.style.SUCCESS(f"\n✓ Configuración provisional elegida: K={best_config[0]}, weights={best_config[1]}"))

        self._export_detailed_csv(all_results, output_csv)
        self._export_summary_csv(global_summary, all_configs, summary_csv)
        self._export_per_fold_csv(per_fold_results, per_fold_csv)
        self._export_confusion_csv(confusion_matrices, confusion_csv, sorted(y.unique()))

        if save_artifacts:
            self._train_and_save_candidate_artifacts(X, y, best_config)

        self.stdout.write(self.style.SUCCESS(f"\n✓ Evaluación completada."))
        self.stdout.write(f"  Resultados detallados: {output_csv}")
        self.stdout.write(f"  Resumen global: {summary_csv}")
        self.stdout.write(f"  Por partición: {per_fold_csv}")
        self.stdout.write(f"  Matrices confusión: {confusion_csv}")

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

    def _build_preprocessor(self):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.REQUIRED_NUMERIC_FIELDS),
                ("cat", categorical_transformer, self.REQUIRED_CATEGORICAL_FIELDS),
            ],
            remainder="drop",
        )

        return preprocessor

    def _build_pipeline(self, n_neighbors, weights):
        preprocessor = self._build_preprocessor()
        classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric="euclidean",
            algorithm="brute",
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
        return pipeline

    def _print_fold_summary(self, fold_config_results, all_configs, fold_idx):
        self.stdout.write(f"\n--- Resumen Partición {fold_idx} ---")

        for k, weights in all_configs:
            result = fold_config_results[(k, weights)]
            self.stdout.write(
                f"  K={k:2d} weights={weights:8s}: "
                f"Acc={result['accuracy']:.4f} | BalAcc={result['balanced_accuracy']:.4f} | "
                f"PrecMacro={result['precision_macro']:.4f} | RecMacro={result['recall_macro']:.4f} | "
                f"F1Macro={result['f1_macro']:.4f} | Baseline={result['baseline_accuracy']:.4f}"
            )

    def _compute_global_summary(self, all_results, all_configs):
        global_summary = {}

        for k, weights in all_configs:
            config_results = [r for r in all_results if r["k"] == k and r["weights"] == weights]

            global_summary[(k, weights)] = {
                "k": k,
                "weights": weights,
                "mean_accuracy": np.mean([r["accuracy"] for r in config_results]),
                "std_accuracy": np.std([r["accuracy"] for r in config_results]),
                "mean_balanced_accuracy": np.mean([r["balanced_accuracy"] for r in config_results]),
                "std_balanced_accuracy": np.std([r["balanced_accuracy"] for r in config_results]),
                "mean_precision_macro": np.mean([r["precision_macro"] for r in config_results]),
                "std_precision_macro": np.std([r["precision_macro"] for r in config_results]),
                "mean_recall_macro": np.mean([r["recall_macro"] for r in config_results]),
                "std_recall_macro": np.std([r["recall_macro"] for r in config_results]),
                "mean_f1_macro": np.mean([r["f1_macro"] for r in config_results]),
                "std_f1_macro": np.std([r["f1_macro"] for r in config_results]),
                "mean_baseline_accuracy": np.mean([r["baseline_accuracy"] for r in config_results]),
                "fold_results": config_results,
            }

        return global_summary

    def _print_global_summary(self, global_summary, all_configs):
        self.stdout.write(f"\n{'=' * 80}")
        self.stdout.write(self.style.NOTICE("RESUMEN GLOBAL (media ± std sobre 5 particiones)"))
        self.stdout.write(f"{'=' * 80}")

        for k, weights in all_configs:
            gs = global_summary[(k, weights)]
            self.stdout.write(f"\n--- K = {k}, weights = {weights} ---")
            self.stdout.write(
                f"  Accuracy:           {gs['mean_accuracy']:.4f} ± {gs['std_accuracy']:.4f}"
            )
            self.stdout.write(
                f"  Balanced Accuracy:  {gs['mean_balanced_accuracy']:.4f} ± {gs['std_balanced_accuracy']:.4f}"
            )
            self.stdout.write(
                f"  Precision macro:    {gs['mean_precision_macro']:.4f} ± {gs['std_precision_macro']:.4f}"
            )
            self.stdout.write(
                f"  Recall macro:       {gs['mean_recall_macro']:.4f} ± {gs['std_recall_macro']:.4f}"
            )
            self.stdout.write(
                f"  F1 macro:           {gs['mean_f1_macro']:.4f} ± {gs['std_f1_macro']:.4f}"
            )
            self.stdout.write(
                f"  Baseline (train mode): {gs['mean_baseline_accuracy']:.4f}"
            )

            self.stdout.write("  Detalle por fold:")
            for fr in gs["fold_results"]:
                self.stdout.write(
                    f"    Fold {fr['fold']}: Acc={fr['accuracy']:.4f} | "
                    f"BalAcc={fr['balanced_accuracy']:.4f} | "
                    f"F1Macro={fr['f1_macro']:.4f} | "
                    f"Baseline={fr['baseline_accuracy']:.4f}"
                )

    def _select_best_config(self, global_summary):
        best_config = None
        best_f1 = -1

        for (k, weights), gs in global_summary.items():
            if gs["mean_f1_macro"] > best_f1:
                best_f1 = gs["mean_f1_macro"]
                best_config = (k, weights)

        return best_config

    def _train_and_save_candidate_artifacts(self, X, y, best_config):
        k, weights = best_config
        self.stdout.write(f"\n{'=' * 80}")
        self.stdout.write(self.style.NOTICE("ENTRENANDO ARTEFACTOS CANDIDATOS CON DATASET COMPLETO"))
        self.stdout.write(f"{'=' * 80}")

        pipeline = self._build_pipeline(k, weights)
        pipeline.fit(X, y)

        base_dir = Path(settings.BASE_DIR)
        ml_dir = base_dir / "apps" / "recommendations" / "ml"

        classifier_path = ml_dir / "knn_classifier_candidate.joblib"
        preprocessor_path = ml_dir / "preprocesador_classifier_candidate.joblib"

        joblib.dump(pipeline.named_steps["classifier"], classifier_path)
        joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_path)

        self.stdout.write(self.style.SUCCESS(f"\n✓ Artefactos guardados:"))
        self.stdout.write(f"  Clasificador: {classifier_path}")
        self.stdout.write(f"  Preprocesador: {preprocessor_path}")

        self.stdout.write(
            self.style.WARNING(
                "\n⚠ NOTA: Estos son ARTEFACTOS CANDIDATOS para evaluación. "
                "NO sobrescriben los archivos de producción "
                "(knn_zpasio.joblib, preprocesador_zpasio.joblib). "
                "La integración en producción requiere validación adicional "
                "y decisión explícita."
            )
        )

    def _export_detailed_csv(self, all_results, output_csv):
        fieldnames = [
            "fold",
            "k",
            "weights",
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "baseline_accuracy",
            "baseline_class",
            "n_train",
            "n_test",
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in all_results:
                writer.writerow(r)

    def _export_summary_csv(self, global_summary, all_configs, summary_csv):
        fieldnames = [
            "k",
            "weights",
            "mean_accuracy",
            "std_accuracy",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "mean_precision_macro",
            "std_precision_macro",
            "mean_recall_macro",
            "std_recall_macro",
            "mean_f1_macro",
            "std_f1_macro",
            "mean_baseline_accuracy",
        ]

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for k, weights in all_configs:
                gs = global_summary[(k, weights)]
                writer.writerow({
                    "k": k,
                    "weights": weights,
                    "mean_accuracy": round(gs["mean_accuracy"], 6),
                    "std_accuracy": round(gs["std_accuracy"], 6),
                    "mean_balanced_accuracy": round(gs["mean_balanced_accuracy"], 6),
                    "std_balanced_accuracy": round(gs["std_balanced_accuracy"], 6),
                    "mean_precision_macro": round(gs["mean_precision_macro"], 6),
                    "std_precision_macro": round(gs["std_precision_macro"], 6),
                    "mean_recall_macro": round(gs["mean_recall_macro"], 6),
                    "std_recall_macro": round(gs["std_recall_macro"], 6),
                    "mean_f1_macro": round(gs["mean_f1_macro"], 6),
                    "std_f1_macro": round(gs["std_f1_macro"], 6),
                    "mean_baseline_accuracy": round(gs["mean_baseline_accuracy"], 6),
                })

    def _export_per_fold_csv(self, per_fold_results, per_fold_csv):
        fieldnames = [
            "fold",
            "k",
            "weights",
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "baseline_accuracy",
        ]

        with open(per_fold_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in per_fold_results:
                writer.writerow(r)

    def _export_confusion_csv(self, confusion_matrices, confusion_csv, class_labels):
        fieldnames = ["fold", "k", "weights"] + [f"true_{c}_pred_{p}" for c in class_labels for p in class_labels]

        with open(confusion_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for (fold_idx, k, weights), cm in sorted(confusion_matrices.items()):
                row = {"fold": fold_idx, "k": k, "weights": weights}
                for i, true_label in enumerate(class_labels):
                    for j, pred_label in enumerate(class_labels):
                        row[f"true_{true_label}_pred_{pred_label}"] = cm[i, j]
                writer.writerow(row)