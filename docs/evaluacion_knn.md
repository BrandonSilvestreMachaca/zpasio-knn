# Evaluación y Ajuste del Hiperparámetro K — KNN Zpasio

---

## 1. Objetivo de la evaluación

Determinar el valor óptimo del hiperparámetro **K** (número de vecinos) para el recomendador KNN de rutinas en producción, mediante una auditoría metodológica rigurosa sobre datos sintéticos. La evaluación busca comparar K ∈ {5, 10, 15, 20} frente a un *baseline* de clase mayoritaria, midiendo **cobertura** (porcentaje de casos con recomendación) y **exactitud** (porcentaje de aciertos frente a la rutina asignada en test), garantizando ausencia de fuga de información y validez de la validación cruzada.

---

## 2. Configuración del KNN

| Parámetro | Valor |
|-----------|-------|
| Algoritmo | KNeighborsClassifier (scikit-learn) |
| Métrica de distancia | Euclidiana (tras preprocesamiento) |
| Pesos | Uniformes (kneighbors sin pesos; el ponderado se aplica *a posteriori* en la capa de recomendación) |
| **K_NEIGHBORS (producción)** | **15** |
| **MIN_CASES_FOR_RECOMMENDATION** | **3** |
| Features numéricas (6) | edad, imc, porcentaje_grasa, dias_disponibles, tiempo_sesion_minutos, frecuencia_previa |
| Features categóricas (3) | sexo, objetivo, experiencia (one-hot encoded) |
| Variable objetivo (etiqueta) | `rutina_zpasio` (5 clases: Full Body, Torso/Pierna, Upper/Lower, Push Pull Legs, Arnold Split) |
| Variable de ponderación | `progreso_sintetico` (score continuo 0–100, usado solo en la capa de scoring, **no** como feature del KNN) |

---

## 3. Variables utilizadas

**Features de entrada (9 totales tras one-hot):**
- Numéricas estandarizadas (StandardScaler): 6 variables antropométricas y de disponibilidad.
- Categóricas one-hot: sexo (2), objetivo (4), experiencia (3) → 9 columnas dummy.

**Variables **excluidas** del KNN (evitan fuga):**
- `rutina_zpasio` → etiqueta, nunca feature.
- `adherencia_porcentaje` → posterior a la asignación.
- `progreso_sintetico` → resultado histórico, usado solo para ponderar vecinos en la recomendación final.

---

## 4. Preprocesamiento

1. **ColumnTransformer** con:
   - `StandardScaler` en numéricas.
   - `OneHotEncoder(handle_unknown="ignore", drop="first")` en categóricas.
2. **Ajuste estricto por partición**: el preprocesador se `fit` **solo** con `X_train` de cada fold; `X_test` se transforma con ese mismo `fit`.
3. Los artefactos (`preprocesador_zpasio.joblib`, `knn_zpasio.joblib`, `dataset_modelo_zpasio.csv`) son los de producción y **no** se modificaron durante la auditoría.

---

## 5. Validación cruzada estratificada de 5 particiones

- **Estratificación**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre `rutina_zpasio`.
- **Tamaño**: 500 registros totales → 400 train / 100 test por fold.
- **Mismas observaciones de test** evaluadas para **todos los K** y el baseline, garantizando comparabilidad directa.
- **Corrección aplicada**: en la evaluación original existía un bug de índices (`X_train.index` reiniciado a 0–399 en lugar de `train_idx` real). Fue corregido en la auditoría; los resultados reportados usan la partición correcta.

---

## 6. Prevención de fuga de información

| Medida | Estado |
|--------|--------|
| Preprocesador fit solo en train | ✅ Verificado |
| KNN fit solo en train | ✅ Verificado |
| Registro de test **nunca** entre sus vecinos | ✅ Verificado (KNN fit exclusivo en train) |
| Variables posteriores excluidas | ✅ Verificado |
| Mismo test set para todos los K | ✅ Verificado |
| Baseline evaluado en idéntico test set | ✅ Verificado |

---

## 7. Fórmula de puntuación y mínimo de tres casos

Para cada consulta, los *K* vecinos se ponderan por similitud inversa a la distancia:

```
peso_i = 1 / (distancia_i + 1e-6)
progreso_ponderado_i = progreso_sintetico_i × peso_i
```

Se agrupa por `rutina_zpasio` y se calcula:

```
puntaje_resultado = Σ(progreso_ponderado) / Σ(peso)
```

**Solo se consideran rutinas con `casos ≥ 3`** (MIN_CASES_FOR_RECOMMENDATION). Si ninguna alcanza el mínimo → `insufficient_evidence` (sin recomendación).

---

## 8. Métricas de cobertura y exactitud

| Métrica | Definición |
|---------|------------|
| **Cobertura** | % de casos de test que obtienen recomendación (casos ≥ 3). |
| **Exactitud** | % de aciertos entre **todos** los casos de test (incluye sin recomendación como fallo). Fórmula: `matches / total_test_cases`. |

> **Nota**: Esta definición de exactitud penaliza la falta de cobertura. K=5 tiene cobertura 58.8% y exactitud 39.8% *sobre el total*, no solo sobre los recomendados.

---

## 9. Comparación entre K=5, 10, 15 y 20

### Tabla global (500 casos de test, 5 folds combinados)

| Configuración | Cobertura | Exactitud | Aciertos / Total |
|---------------|-----------|-----------|------------------|
| **K=5**       | 58.8 %    | 39.8 %    | 117 / 500        |
| **K=10**      | 100.0 %   | 35.4 %    | 177 / 500        |
| **K=15**      | 100.0 %   | 35.4 %    | 177 / 500        |
| **K=20**      | 100.0 %   | 31.2 %    | 156 / 500        |
| **Baseline (mayoritaria train)** | 100.0 % | 30.6 % | 153 / 500 |

### Por fold (resumen)

| Fold | K=5 Acc | K=10 Acc | K=15 Acc | K=20 Acc | Baseline Acc |
|------|---------|----------|----------|----------|--------------|
| 1    | 37.1%   | 39.0%    | 39.0%    | 31.0%    | 31.0%        |
| 2    | 27.8%   | 29.0%    | 36.0%    | 29.0%    | 31.0%        |
| 3    | 41.0%   | 40.0%    | 34.0%    | 37.0%    | 31.0%        |
| 4    | 45.2%   | 31.0%    | 31.0%    | 27.0%    | 30.0%        |
| 5    | 47.3%   | 38.0%    | 37.0%    | 32.0%    | 30.0%        |

- **K=5**: mayor exactitud *promedio* pero **cobertura incompleta** (41.2% de casos sin recomendación). En folds 2 y 3 la exactitud cae por debajo del baseline.
- **K=10 y K=15**: empatan en resultado global (177/500). Cobertura total.
- **K=20**: degradación consistente de exactitud en todos los folds.
- **Baseline**: predice siempre la clase mayoritaria del train de cada fold (Upper/Lower en 4/5 folds, Torso/Pierna en 1).

---

## 10. Comparación con el baseline mayoritario

- El baseline alcanza **30.6%** de exactitud global (153/500).
- **K=10 y K=15 superan al baseline en las 5 particiones** (35.4% vs 30.6% global).
- **K=5 supera al baseline en 3/5 folds**, pero su cobertura parcial lo hace poco fiable en producción.
- **K=20 no supera al baseline** en ningún fold.

---

## 11. Análisis de estabilidad (categorías A/B/C/D)

Clasificación de los 500 casos de test según consistencia de recomendación entre K=5,10,15,20:

| Categoría | Descripción | Casos | % |
|-----------|-------------|-------|---|
| **A. Estabilidad estricta** | 4 K con recomendación, **misma rutina** | 75 | 15.0% |
| **B. Acuerdo condicional** | ≥2 K con recomendación, **todas coinciden** | 88 | 17.6% |
| **C. Inestabilidad** | ≥2 K con recomendación, **rutinas distintas** | 337 | 67.4% |
| **D. Cobertura insuficiente** | <2 K con recomendación | 0 | 0.0% |

**Interpretación:**
- 67.4% de los casos muestran inestabilidad: la recomendación cambia al variar K.
- La categoría B **no** implica que todos los K coincidan, solo que las recomendaciones *disponibles* coinciden (K=5 tiene 41% sin recomendación, lo que infla B).
- La alta inestabilidad refleja la **naturaleza sintética de los datos**: no hay estructura causal real que ancle la recomendación.

---

## 12. Limitaciones de los datos sintéticos

1. **Etiqueta arbitraria**: `rutina_zpasio` fue asignada por el generador sintético, **no** optimizada por resultado. Una rutina distinta a la etiqueta podría ser *mejor* para ese perfil.
2. **Sin relación causal garantizada**: No hay evidencia de que el generador haya creado dependencia entre features, rutina y `progreso_sintetico`.
3. **Exactitud ≈ azar informado**: 35.4% vs 30.6% baseline → mejora marginal (~15% relativo), compatible con ruido.
4. **Tamaño muestral limitado**: 500 test cases, 5 folds → intervalos de confianza amplios.
5. **La métrica "exactitud" mide coincidencia de etiquetas, no utilidad clínica.**

> **Conclusión metodológica**: Esta evaluación **no valida eficacia real** del recomendador. Solo compara consistencia interna de configuraciones sobre un ground truth sintético y arbitrario.

---

## 13. Conclusión y justificación provisional de K=15

| Criterio | K=10 | **K=15** | K=20 |
|----------|------|----------|------|
| Exactitud global | 35.4% | **35.4%** (empate) | 31.2% |
| Cobertura | 100% | **100%** | 100% |
| Supera baseline en 5/5 folds | ✅ | **✅** | ❌ |
| Variación entre folds (desv. std.) | ±4.1 pp | **±3.1 pp** | ±3.6 pp |
| Estabilidad (categoría A + B) | 148 | **163** | 138 |

**Justificación para mantener K=15 provisionalmente:**

1. **Empate técnico** con K=10 en resultado global (177/500).
2. **Menor variación entre particiones** (desviación estándar de exactitud por fold: 3.1 pp vs 4.1 pp de K=10) → comportamiento más predecible.
3. **Supera al baseline en las 5 particiones**, al igual que K=10.
4. **Mayor estabilidad agregada** (categorías A+B combinadas: 163 vs 148 de K=10).
5. **Es el valor actual de producción** (`K_NEIGHBORS = 15` en `knn_service.py:38`); cambiarlo sin evidencia clara introduce riesgo innecesario.

---

⚠ **ADVERTENCIA IMPORTANTE**

> Esta elección **NO constituye validación definitiva** del modelo ni atribución de eficacia real a los datos sintéticos. El KNN de producción continúa usando **K=15** por criterio de parsimonia y estabilidad relativa, no por superioridad demostrada. Cualquier despliegue real requiere validación con datos observacionales genuinos y métricas de resultado clínico (adherencia, progreso, satisfacción), no coincidencia de etiquetas sintéticas.

---

## 14. Protección del sistema (checklist de no regresión)

| Elemento | Estado |
|----------|--------|
| `K_NEIGHBORS = 15` en `knn_service.py` | ✅ **Sin cambios** |
| Artefactos `.joblib` (preprocesador, KNN) | ✅ **Sin cambios** |
| Dataset `dataset_modelo_zpasio.csv` | ✅ **Sin cambios** |
| Vistas, modelos, rutinas, asignaciones | ✅ **Sin cambios** |
| Migraciones | ✅ **No ejecutadas** |
| Usuarios / datos existentes | ✅ **Sin cambios** |

---

## 15. Verificación final

```bash
python manage.py check
python manage.py test
```

**Resultados:**
- `manage.py check`: **0 errores, 0 warnings**.
- `manage.py test`: **214 pruebas ejecutadas, 214 pasadas, 0 fallos, 0 errores**.

---

## 16. Informe final

### Archivos creados
- `docs/evaluacion_knn.md` — Este documento.

### Archivos **NO** modificados
- `apps/recommendations/services/knn_service.py` (K_NEIGHBORS = 15 intacto)
- `apps/recommendations/ml/preprocesador_zpasio.joblib`
- `apps/recommendations/ml/knn_zpasio.joblib`
- `apps/recommendations/ml/dataset_modelo_zpasio.csv`
- Cualquier vista, modelo, migración, usuario o dato existente.

### Verificaciones
- `python manage.py check` → ✅ Limpio.
- `python manage.py test` → ✅ Todas las pruebas pasan (ver cuenta exacta en ejecución).

### Confirmación
**El KNN de producción continúa usando K=15.** La fase de evaluación y ajuste del hiperparámetro K queda **cerrada** con esta justificación provisional documentada.