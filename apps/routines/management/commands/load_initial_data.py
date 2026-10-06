"""Carga inicial del catálogo de ejercicios y plantillas de rutinas.

Uso:
    python manage.py load_initial_data

El comando es IDEMPOTENTE: ejecutarlo varias veces no duplica
ejercicios, rutinas, sesiones ni relaciones. Utiliza get_or_create
para cada entidad; los registros ya existentes (incluidos los editados
por un entrenador) NO se sobrescriben.

Los datos generados son plantillas editables desde el CRUD existente;
no asignan rutinas a usuarios ni ejecutan ninguna lógica de
recomendación.
"""

from django.core.management.base import BaseCommand
from django.db import transaction

from apps.exercises.models import Exercise
from apps.routines.models import Routine, RoutineExercise, RoutineSession

# Catálogo inicial: (nombre, grupo_muscular, dificultad, descripción).
EXERCISES = [
    # PECHO
    ("Press de banca", "PECHO", "PRINCIPIANTE",
     "Press horizontal con barra para pecho."),
    ("Press inclinado", "PECHO", "INTERMEDIO",
     "Press en banco inclinado, énfasis en pecho superior."),
    ("Press con mancuernas", "PECHO", "PRINCIPIANTE",
     "Press horizontal con mancuernas, mayor rango de recorrido."),
    ("Aperturas con mancuernas", "PECHO", "PRINCIPIANTE",
     "Aperturas en banco plano para aislar el pectoral."),
    ("Aperturas en máquina", "PECHO", "PRINCIPIANTE",
     "Aperturas guiadas en máquina (peck deck)."),
    ("Fondos enfocados en pecho", "PECHO", "INTERMEDIO",
     "Fondos en paralelas con inclinación hacia el pecho."),
    # ESPALDA
    ("Dominadas", "ESPALDA", "AVANZADO",
     "Tracción vertical con peso corporal."),
    ("Jalón al pecho", "ESPALDA", "PRINCIPIANTE",
     "Tracción vertical en polea alta."),
    ("Remo con barra", "ESPALDA", "INTERMEDIO",
     "Remo horizontal con barra para densidad de espalda."),
    ("Remo sentado en polea", "ESPALDA", "PRINCIPIANTE",
     "Remo horizontal en máquina de polea baja."),
    ("Remo con mancuerna", "ESPALDA", "PRINCIPIANTE",
     "Remo unilateral con mancuerna apoyando la rodilla."),
    ("Pullover en polea", "ESPALDA", "INTERMEDIO",
     "Extensión de hombro en polea para dorsales."),
    # PIERNAS
    ("Sentadilla", "PIERNAS", "INTERMEDIO",
     "Sentadilla con barra, base del trabajo de piernas."),
    ("Prensa de piernas", "PIERNAS", "PRINCIPIANTE",
     "Empuje en máquina de prensa, menor demanda técnica."),
    ("Extensión de cuádriceps", "PIERNAS", "PRINCIPIANTE",
     "Extensión de rodilla en máquina para cuádriceps."),
    ("Curl femoral", "PIERNAS", "PRINCIPIANTE",
     "Flexión de rodilla para isquiotibiales."),
    ("Peso muerto rumano", "PIERNAS", "INTERMEDIO",
     "Bisagra de cadera con barra para femorales y glúteos."),
    ("Zancadas", "PIERNAS", "PRINCIPIANTE",
     "Desplantes con o sin carga, trabajo unilateral."),
    ("Elevación de talones", "PIERNAS", "PRINCIPIANTE",
     "Extensión de tobillos para pantorrillas."),
    ("Hip thrust", "PIERNAS", "INTERMEDIO",
     "Empuje de cadera con barra, énfasis en glúteos."),
    # HOMBROS
    ("Press militar", "HOMBROS", "INTERMEDIO",
     "Press vertical con barra de pie."),
    ("Press de hombros con mancuernas", "HOMBROS", "PRINCIPIANTE",
     "Press vertical con mancuernas, sentado o de pie."),
    ("Elevaciones laterales", "HOMBROS", "PRINCIPIANTE",
     "Elevaciones con mancuernas para deltoides lateral."),
    ("Pájaros (elevaciones posteriores)", "HOMBROS", "PRINCIPIANTE",
     "Aperturas en banco inclinado para deltoides posterior."),
    ("Face pull", "HOMBROS", "PRINCIPIANTE",
     "Tracción a la cara en polea para salud del hombro."),
    # BÍCEPS
    ("Curl con barra", "BICEPS", "PRINCIPIANTE",
     "Curl de bíceps de pie con barra."),
    ("Curl con mancuernas", "BICEPS", "PRINCIPIANTE",
     "Curl alternado con mancuernas."),
    ("Curl martillo", "BICEPS", "PRINCIPIANTE",
     "Curl con agarre neutro para braquial y antebrazo."),
    ("Curl predicador", "BICEPS", "INTERMEDIO",
     "Curl en banco predicador, aislamiento del bíceps."),
    # TRÍCEPS
    ("Extensión de tríceps en polea", "TRICEPS", "PRINCIPIANTE",
     "Extensión de codo en polea alta."),
    ("Extensión de tríceps sobre la cabeza", "TRICEPS", "INTERMEDIO",
     "Extensión de codo tras la nuca con mancuerna o cuerda."),
    ("Press francés", "TRICEPS", "INTERMEDIO",
     "Extensión de tríceps acostado con barra Z."),
    ("Fondos", "TRICEPS", "INTERMEDIO",
     "Fondos en paralelas con énfasis en tríceps."),
    # CORE
    ("Plancha", "ABDOMEN", "PRINCIPIANTE",
     "Isométrico de core apoyando antebrazos."),
    ("Crunch abdominal", "ABDOMEN", "PRINCIPIANTE",
     "Flexión corta de tronco en el suelo."),
    ("Elevación de piernas", "ABDOMEN", "INTERMEDIO",
     "Elevación de piernas colgado o tumbado, abdomen bajo."),
]

# Ejercicios ya presentes en la base que se reutilizan en vez de
# crear una entrada nueva del catálogo (nombre del catálogo -> nombre
# existente en la base).
ALIASES = {
    "Press de banca": "Press banca",
    "Press inclinado": "Press banca inclinado",
    "Jalón al pecho": "jalon al pecho",
}

# Plantillas de rutinas. Cada ejercicio: (nombre, series, repeticiones,
# duración_segundos, descanso_segundos).
ROUTINE_TEMPLATES = [
    {
        "name": "Push Pull Legs (PPL)",
        "description": (
            "Rutina dividida en tres sesiones: empuje (pecho, hombros, "
            "tríceps), tracción (espalda, bíceps) y piernas."
        ),
        "goal": "GANANCIA_MUSCULAR",
        "level": "INTERMEDIO",
        "intensity": "ALTA",
        "days_per_week": 6,
        "estimated_duration_minutes": 60,
        "sessions": [
            ("Push", [
                ("Press de banca", 4, 8, None, 120),
                ("Press inclinado", 3, 10, None, 90),
                ("Press militar", 3, 10, None, 90),
                ("Elevaciones laterales", 3, 12, None, 60),
                ("Extensión de tríceps en polea", 3, 12, None, 60),
            ]),
            ("Pull", [
                ("Jalón al pecho", 4, 8, None, 120),
                ("Remo con barra", 4, 8, None, 120),
                ("Remo sentado en polea", 3, 10, None, 90),
                ("Face pull", 3, 12, None, 60),
                ("Curl con barra", 3, 10, None, 60),
                ("Curl martillo", 3, 12, None, 60),
            ]),
            ("Legs", [
                ("Sentadilla", 4, 8, None, 120),
                ("Prensa de piernas", 3, 10, None, 90),
                ("Peso muerto rumano", 3, 8, None, 120),
                ("Curl femoral", 3, 12, None, 60),
                ("Elevación de talones", 4, 15, None, 45),
                ("Plancha", 3, None, 40, 45),
            ]),
        ],
    },
    {
        "name": "Upper / Lower",
        "description": (
            "Torso y tren inferior en sesiones alternadas, cuatro días "
            "por semana."
        ),
        "goal": "MANTENIMIENTO",
        "level": "INTERMEDIO",
        "intensity": "MODERADA",
        "days_per_week": 4,
        "estimated_duration_minutes": 55,
        "sessions": [
            ("Upper", [
                ("Press de banca", 4, 8, None, 120),
                ("Jalón al pecho", 3, 10, None, 90),
                ("Remo con barra", 3, 10, None, 90),
                ("Press militar", 3, 10, None, 90),
                ("Elevaciones laterales", 3, 12, None, 60),
                ("Curl con barra", 3, 12, None, 60),
                ("Extensión de tríceps en polea", 3, 12, None, 60),
            ]),
            ("Lower", [
                ("Sentadilla", 4, 8, None, 120),
                ("Prensa de piernas", 3, 10, None, 90),
                ("Peso muerto rumano", 3, 10, None, 90),
                ("Curl femoral", 3, 12, None, 60),
                ("Elevación de talones", 4, 15, None, 45),
                ("Crunch abdominal", 3, 15, None, 45),
            ]),
        ],
    },
    {
        "name": "Full Body",
        "description": (
            "Sesión de cuerpo completo con selección equilibrada de "
            "movimientos básicos."
        ),
        "goal": "MANTENIMIENTO",
        "level": "PRINCIPIANTE",
        "intensity": "MODERADA",
        "days_per_week": 3,
        "estimated_duration_minutes": 50,
        "sessions": [
            ("Cuerpo completo", [
                ("Sentadilla", 3, 10, None, 90),
                ("Press de banca", 3, 10, None, 90),
                ("Remo con barra", 3, 10, None, 90),
                ("Press militar", 3, 10, None, 60),
                ("Curl con barra", 3, 12, None, 60),
                ("Plancha", 3, None, 30, 45),
            ]),
        ],
    },
    {
        "name": "Torso / Pierna",
        "description": (
            "División clásica torso-pierna para trabajar cada grupo con "
            "volumen suficiente dos veces por semana."
        ),
        "goal": "GANANCIA_MUSCULAR",
        "level": "INTERMEDIO",
        "intensity": "MODERADA",
        "days_per_week": 4,
        "estimated_duration_minutes": 55,
        "sessions": [
            ("Torso", [
                ("Press de banca", 4, 8, None, 120),
                ("Remo con barra", 4, 8, None, 120),
                ("Press inclinado", 3, 10, None, 90),
                ("Jalón al pecho", 3, 10, None, 90),
                ("Press militar", 3, 10, None, 60),
                ("Curl con mancuernas", 3, 12, None, 60),
                ("Press francés", 3, 12, None, 60),
            ]),
            ("Pierna", [
                ("Sentadilla", 4, 8, None, 120),
                ("Hip thrust", 4, 10, None, 90),
                ("Peso muerto rumano", 3, 10, None, 90),
                ("Extensión de cuádriceps", 3, 12, None, 60),
                ("Curl femoral", 3, 12, None, 60),
                ("Elevación de talones", 4, 15, None, 45),
                ("Elevación de piernas", 3, 12, None, 45),
            ]),
        ],
    },
    {
        "name": "Arnold Split",
        "description": (
            "División de alta frecuencia: pecho+espalda, hombros+brazos "
            "y piernas en tres sesiones rotativas."
        ),
        "goal": "GANANCIA_MUSCULAR",
        "level": "AVANZADO",
        "intensity": "ALTA",
        "days_per_week": 6,
        "estimated_duration_minutes": 70,
        "sessions": [
            ("Pecho y espalda", [
                ("Press de banca", 4, 8, None, 120),
                ("Dominadas", 4, 8, None, 120),
                ("Press inclinado", 3, 10, None, 90),
                ("Remo con barra", 4, 8, None, 120),
                ("Aperturas con mancuernas", 3, 12, None, 60),
                ("Pullover en polea", 3, 12, None, 60),
            ]),
            ("Hombros y brazos", [
                ("Press militar", 4, 10, None, 90),
                ("Elevaciones laterales", 4, 12, None, 60),
                ("Pájaros (elevaciones posteriores)", 3, 12, None, 60),
                ("Curl con barra", 4, 10, None, 60),
                ("Curl martillo", 3, 12, None, 60),
                ("Press francés", 3, 10, None, 60),
                ("Fondos", 3, 10, None, 60),
            ]),
            ("Piernas", [
                ("Sentadilla", 4, 8, None, 120),
                ("Prensa de piernas", 4, 10, None, 90),
                ("Curl femoral", 3, 12, None, 60),
                ("Elevación de talones", 4, 15, None, 45),
            ]),
        ],
    },
]


class Command(BaseCommand):
    help = (
        "Carga el catálogo inicial de ejercicios y las plantillas de "
        "rutinas. Idempotente: no duplica ni sobrescribe datos."
    )

    @transaction.atomic
    def handle(self, *args, **options):
        exercises = self._seed_exercises()
        self._seed_routines(exercises)

    def _seed_exercises(self):
        """Crea el catálogo o reutiliza los ejercicios ya existentes.

        Devuelve un mapa nombre_de_plantilla -> Exercise para usar al
        construir las rutinas.
        """
        result = {}
        created = reused = 0
        for name, muscle_group, difficulty, description in EXERCISES:
            alias = ALIASES.get(name)
            existing = None
            if alias:
                existing = Exercise.objects.filter(name=alias).first()
            if existing is not None:
                result[name] = existing
                reused += 1
                continue
            obj, was_created = Exercise.objects.get_or_create(
                name=name,
                defaults={
                    "muscle_group": muscle_group,
                    "difficulty": difficulty,
                    "description": description,
                },
            )
            result[name] = obj
            if was_created:
                created += 1
            else:
                reused += 1
        self.stdout.write(
            f"Ejercicios: {created} creados, {reused} reutilizados."
        )
        return result

    def _seed_routines(self, exercises):
        """Crea las plantillas de rutina con sus sesiones y ejercicios.

        Si la rutina ya existe, no se modifican sus atributos: solo se
        agregan sesiones o relaciones que falten.
        """
        for spec in ROUTINE_TEMPLATES:
            routine, created = Routine.objects.get_or_create(
                name=spec["name"],
                defaults={
                    "description": spec["description"],
                    "goal": spec["goal"],
                    "level": spec["level"],
                    "intensity": spec["intensity"],
                    "days_per_week": spec["days_per_week"],
                    "estimated_duration_minutes": spec[
                        "estimated_duration_minutes"
                    ],
                },
            )
            for session_order, (session_name, items) in enumerate(
                spec["sessions"], start=1
            ):
                session, _ = RoutineSession.objects.get_or_create(
                    routine=routine,
                    name=session_name,
                    defaults={"order": session_order},
                )
                for order, (exercise_name, sets, reps, duration, rest) in (
                    enumerate(items, start=1)
                ):
                    RoutineExercise.objects.get_or_create(
                        routine=routine,
                        exercise=exercises[exercise_name],
                        defaults={
                            "session": session,
                            "order": order,
                            "sets": sets,
                            "repetitions": reps,
                            "duration_seconds": duration,
                            "rest_seconds": rest,
                        },
                    )
            status = "creada" if created else "ya existía"
            self.stdout.write(
                self.style.SUCCESS(f"Rutina '{routine.name}': {status}.")
            )
