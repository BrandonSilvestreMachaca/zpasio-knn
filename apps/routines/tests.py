"""Pruebas del módulo de gestión de rutinas de Zpasio."""

from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from apps.accounts.models import CustomUser
from apps.exercises.models import Exercise

from .models import Routine, RoutineExercise, RoutineSession, UserRoutineAssignment


def make_user(username, role, password="testpass123"):
    return CustomUser.objects.create_user(
        username=username, password=password, role=role
    )


def make_exercise(name, muscle="PECHO"):
    return Exercise.objects.create(
        name=name, muscle_group=muscle, difficulty=Exercise.Difficulty.PRINCIPIANTE
    )


def formset_payload(*rows):
    """Construye el payload del formset de ejercicios para el POST."""
    data = {
        "routine_exercises-TOTAL_FORMS": str(len(rows)),
        "routine_exercises-INITIAL_FORMS": "0",
        "routine_exercises-MIN_NUM_FORMS": "0",
        "routine_exercises-MAX_NUM_FORMS": "1000",
    }
    for i, (exercise_id, order, sets, reps, duration, rest) in enumerate(rows):
        data[f"routine_exercises-{i}-exercise"] = str(exercise_id)
        data[f"routine_exercises-{i}-order"] = str(order)
        data[f"routine_exercises-{i}-sets"] = str(sets)
        if reps is not None:
            data[f"routine_exercises-{i}-repetitions"] = str(reps)
        if duration is not None:
            data[f"routine_exercises-{i}-duration_seconds"] = str(duration)
        data[f"routine_exercises-{i}-rest_seconds"] = str(rest)
    return data


class RoutineCrudTests(TestCase):
    """Creación, edición y comportamiento de la relación Rutina-Ejercicio."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.ex1 = make_exercise("Press de banca")
        self.ex2 = make_exercise("Sentadilla", muscle="PIERNAS")
        self.ex3 = make_exercise("Plancha", muscle="ABDOMEN")
        self.client.force_login(self.trainer)

    def _routine_data(self, **overrides):
        data = {
            "name": "Ganancia muscular - Principiante",
            "description": "Rutina base",
            "goal": "GANANCIA_MUSCULAR",
            "level": "PRINCIPIANTE",
            "intensity": "MODERADA",
            "days_per_week": "3",
            "estimated_duration_minutes": "50",
        }
        data.update(overrides)
        return data

    def test_trainer_can_create_routine(self):
        response = self.client.post(
            reverse("routines:create"),
            {
                **self._routine_data(),
                **formset_payload((self.ex1.pk, 1, 3, 10, None, 90)),
            },
        )
        self.assertEqual(Routine.objects.count(), 1)
        routine = Routine.objects.first()
        self.assertRedirects(
            response, reverse("routines:detail", kwargs={"pk": routine.pk})
        )

    def test_trainer_can_add_existing_exercises(self):
        self.client.post(
            reverse("routines:create"),
            {
                **self._routine_data(),
                **formset_payload((self.ex2.pk, 1, 3, 12, None, 60)),
            },
        )
        routine = Routine.objects.get()
        self.assertIn(self.ex2, routine.exercises.all())

    def test_routine_can_contain_multiple_exercises(self):
        self.client.post(
            reverse("routines:create"),
            {
                **self._routine_data(),
                **formset_payload(
                    (self.ex1.pk, 1, 3, 10, None, 90),
                    (self.ex2.pk, 2, 3, 12, None, 90),
                    (self.ex3.pk, 3, 3, None, 30, 45),
                ),
            },
        )
        routine = Routine.objects.get()
        self.assertEqual(routine.exercises.count(), 3)

    def test_exercise_can_be_used_in_multiple_routines(self):
        for name in ("Rutina A", "Rutina B"):
            self.client.post(
                reverse("routines:create"),
                {
                    **self._routine_data(name=name),
                    **formset_payload((self.ex1.pk, 1, 3, 10, None, 60)),
                },
            )
        self.assertEqual(Routine.objects.count(), 2)
        self.assertEqual(self.ex1.routines.count(), 2)

    def test_sets_reps_duration_rest_are_saved(self):
        self.client.post(
            reverse("routines:create"),
            {
                **self._routine_data(),
                **formset_payload(
                    (self.ex1.pk, 1, 4, 8, None, 90),
                    (self.ex3.pk, 2, 3, None, 30, 45),
                ),
            },
        )
        routine = Routine.objects.get()
        reps_item = routine.routine_exercises.get(exercise=self.ex1)
        self.assertEqual((reps_item.sets, reps_item.repetitions,
                          reps_item.rest_seconds), (4, 8, 90))
        self.assertIsNone(reps_item.duration_seconds)
        duration_item = routine.routine_exercises.get(exercise=self.ex3)
        self.assertEqual((duration_item.sets, duration_item.duration_seconds,
                          duration_item.rest_seconds), (3, 30, 45))
        self.assertIsNone(duration_item.repetitions)

    def test_trainer_can_edit_routine(self):
        routine = Routine.objects.create(
            name="Original", goal="MANTENIMIENTO", level="INTERMEDIO"
        )
        response = self.client.post(
            reverse("routines:update", kwargs={"pk": routine.pk}),
            {
                **self._routine_data(name="Editada"),
                **formset_payload((self.ex1.pk, 1, 3, 10, None, 60)),
            },
        )
        routine.refresh_from_db()
        self.assertEqual(routine.name, "Editada")
        self.assertEqual(routine.goal, "GANANCIA_MUSCULAR")
        self.assertEqual(routine.exercises.count(), 1)
        self.assertRedirects(
            response, reverse("routines:detail", kwargs={"pk": routine.pk})
        )

    def test_trainer_can_add_and_remove_exercises(self):
        routine = Routine.objects.create(
            name="R", goal="RESISTENCIA", level="AVANZADO"
        )
        item = RoutineExercise.objects.create(
            routine=routine, exercise=self.ex1, order=1, sets=3,
            repetitions=10, rest_seconds=60,
        )
        payload = {
            "routine_exercises-TOTAL_FORMS": "2",
            "routine_exercises-INITIAL_FORMS": "1",
            "routine_exercises-MIN_NUM_FORMS": "0",
            "routine_exercises-MAX_NUM_FORMS": "1000",
            # Quitar ex1
            "routine_exercises-0-id": str(item.pk),
            "routine_exercises-0-exercise": str(self.ex1.pk),
            "routine_exercises-0-order": "1",
            "routine_exercises-0-sets": "3",
            "routine_exercises-0-repetitions": "10",
            "routine_exercises-0-rest_seconds": "60",
            "routine_exercises-0-DELETE": "on",
            # Agregar ex2
            "routine_exercises-1-exercise": str(self.ex2.pk),
            "routine_exercises-1-order": "1",
            "routine_exercises-1-sets": "4",
            "routine_exercises-1-repetitions": "15",
            "routine_exercises-1-rest_seconds": "45",
        }
        self.client.post(
            reverse("routines:update", kwargs={"pk": routine.pk}),
            {**self._routine_data(name="R"), **payload},
        )
        self.assertEqual(routine.exercises.count(), 1)
        self.assertIn(self.ex2, routine.exercises.all())
        self.assertNotIn(self.ex1, routine.exercises.all())

    def test_exercises_not_duplicated_when_added_to_routine(self):
        total_before = Exercise.objects.count()
        self.client.post(
            reverse("routines:create"),
            {
                **self._routine_data(),
                **formset_payload((self.ex1.pk, 1, 3, 10, None, 60)),
            },
        )
        self.assertEqual(Exercise.objects.count(), total_before)
        # No se puede agregar dos veces el mismo ejercicio a una rutina
        response = self.client.post(
            reverse("routines:create"),
            {
                **self._routine_data(name="Duplicada"),
                **formset_payload(
                    (self.ex1.pk, 1, 3, 10, None, 60),
                    (self.ex1.pk, 2, 3, 8, None, 60),
                ),
            },
        )
        routine = Routine.objects.filter(name="Duplicada")
        self.assertFalse(routine.exists())
        self.assertIn(
            b"duplicado", response.content.lower()
        )


class RoutineFormValidationTests(TestCase):
    """Errores de validación visibles y comportamiento tras la corrección
    del fallo silencioso en la creación de rutinas."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.ex1 = make_exercise("Press de banca")
        self.client.force_login(self.trainer)

    def _base_payload(self, **overrides):
        data = {
            "name": "Rutina válida",
            "description": "",
            "goal": "GANANCIA_MUSCULAR",
            "level": "PRINCIPIANTE",
            "intensity": "MODERADA",
            "days_per_week": "3",
            "estimated_duration_minutes": "50",
            **formset_payload((self.ex1.pk, 1, 3, 10, None, 90)),
        }
        data.update(overrides)
        return data

    def test_routine_without_name_shows_field_error(self):
        response = self.client.post(
            reverse("routines:create"), self._base_payload(name="")
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Routine.objects.count(), 0)
        self.assertContains(response, "Este campo es obligatorio", count=1,
                            html=False)

    def test_valid_routine_is_created_with_exercise_and_redirects(self):
        response = self.client.post(
            reverse("routines:create"), self._base_payload()
        )
        routine = Routine.objects.get()
        self.assertRedirects(
            response, reverse("routines:detail", kwargs={"pk": routine.pk})
        )
        self.assertEqual(list(routine.exercises.all()), [self.ex1])

    def test_invalid_form_does_not_create_any_records(self):
        # reps + duración a la vez: error de validación del modelo
        payload = self._base_payload()
        payload["routine_exercises-0-repetitions"] = "10"
        payload["routine_exercises-0-duration_seconds"] = "30"
        response = self.client.post(reverse("routines:create"), payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Routine.objects.count(), 0)
        self.assertEqual(RoutineExercise.objects.count(), 0)

    def test_reps_and_duration_error_is_visible_in_page(self):
        """Regresión del bug silencioso: el error no-campo del formset debe
        verse en el HTML renderizado."""
        payload = self._base_payload()
        payload["routine_exercises-0-repetitions"] = "10"
        payload["routine_exercises-0-duration_seconds"] = "30"
        response = self.client.post(reverse("routines:create"), payload)
        self.assertContains(response, "errorlist")
        self.assertContains(response, "repeticiones")

    def test_valid_post_creates_no_duplicates(self):
        self.client.post(reverse("routines:create"), self._base_payload())
        self.assertEqual(Routine.objects.count(), 1)
        self.assertEqual(RoutineExercise.objects.count(), 1)
        self.assertEqual(
            Exercise.objects.filter(name="Press de banca").count(), 1
        )


class RoutinePermissionTests(TestCase):
    """Los usuarios normales no pueden modificar rutinas; las URLs están
    protegidas por rol en el backend."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.user = make_user("user", "USUARIO")
        self.ex1 = make_exercise("Remo")
        self.routine = Routine.objects.create(
            name="Rutina X", goal="GANANCIA_MUSCULAR", level="INTERMEDIO"
        )
        self.client.force_login(self.user)

    def test_user_cannot_create_routine(self):
        response = self.client.get(reverse("routines:create"))
        self.assertEqual(response.status_code, 403)
        response = self.client.post(
            reverse("routines:create"),
            {
                **{
                    "name": "Hack",
                    "goal": "MANTENIMIENTO",
                    "level": "PRINCIPIANTE",
                },
                **formset_payload((self.ex1.pk, 1, 3, 10, None, 60)),
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(Routine.objects.count(), 1)

    def test_user_cannot_edit_routine(self):
        response = self.client.post(
            reverse("routines:update", kwargs={"pk": self.routine.pk}),
            {"name": "Hack", "goal": "MANTENIMIENTO", "level": "AVANZADO"},
        )
        self.assertEqual(response.status_code, 403)
        self.routine.refresh_from_db()
        self.assertEqual(self.routine.name, "Rutina X")

    def test_user_cannot_delete_routine(self):
        response = self.client.post(
            reverse("routines:delete", kwargs={"pk": self.routine.pk}),
            {"confirm": "1"},
        )
        self.assertEqual(response.status_code, 403)
        self.routine.refresh_from_db()
        self.assertTrue(self.routine.is_active)

    def test_urls_protected_by_role(self):
        protected = [
            reverse("routines:create"),
            reverse("routines:update", kwargs={"pk": self.routine.pk}),
            reverse("routines:delete", kwargs={"pk": self.routine.pk}),
        ]
        for url in protected:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 403)
        # Anónimo es redirigido al login
        self.client.logout()
        self.assertEqual(self.client.get(protected[0]).status_code, 302)

    def test_user_can_see_active_routines_only(self):
        Routine.objects.create(
            name="Oculta", goal="MANTENIMIENTO", level="PRINCIPIANTE",
            is_active=False,
        )
        response = self.client.get(reverse("routines:list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Rutina X")
        self.assertNotContains(response, "Oculta")


class RoutineTrainingAttributesTests(TestCase):
    """Campos descriptivos del entrenamiento: intensidad, días por semana
    y duración estimada por sesión."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.ex1 = make_exercise("Press de banca")
        self.client.force_login(self.trainer)

    def _payload(self, **overrides):
        data = {
            "name": "Rutina base",
            "description": "",
            "goal": "GANANCIA_MUSCULAR",
            "level": "PRINCIPIANTE",
            "intensity": "ALTA",
            "days_per_week": "4",
            "estimated_duration_minutes": "60",
            **formset_payload((self.ex1.pk, 1, 3, 10, None, 90)),
        }
        data.update(overrides)
        return data

    def test_routine_is_created_with_training_attributes(self):
        self.client.post(reverse("routines:create"), self._payload())
        routine = Routine.objects.get()
        self.assertEqual(routine.intensity, "ALTA")
        self.assertEqual(routine.days_per_week, 4)
        self.assertEqual(routine.estimated_duration_minutes, 60)

    def test_days_per_week_boundary_values_are_valid(self):
        for days in ("1", "7"):
            with self.subTest(days=days):
                response = self.client.post(
                    reverse("routines:create"),
                    self._payload(days_per_week=days,
                                  name=f"R-{days}"),
                )
                self.assertEqual(response.status_code, 302)

    def test_days_per_week_out_of_range_is_rejected(self):
        for days in ("0", "8"):
            with self.subTest(days=days):
                response = self.client.post(
                    reverse("routines:create"),
                    self._payload(days_per_week=days),
                )
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "entre 1 y 7")
        self.assertEqual(Routine.objects.count(), 0)

    def test_duration_must_be_positive(self):
        response = self.client.post(
            reverse("routines:create"),
            self._payload(estimated_duration_minutes="0"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Routine.objects.count(), 0)
        self.assertContains(response, "mayor que cero")

    def test_invalid_intensity_is_rejected(self):
        response = self.client.post(
            reverse("routines:create"),
            self._payload(intensity="EXTREMA"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Routine.objects.count(), 0)

    def test_intensity_is_required(self):
        payload = self._payload()
        payload.pop("intensity")
        response = self.client.post(reverse("routines:create"), payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Routine.objects.count(), 0)

    def test_detail_shows_training_attributes(self):
        self.client.post(reverse("routines:create"), self._payload())
        routine = Routine.objects.get()
        response = self.client.get(
            reverse("routines:detail", kwargs={"pk": routine.pk})
        )
        self.assertContains(response, "Alta")
        self.assertContains(response, "4 días/semana")
        self.assertContains(response, "60 minutos")


class RoutineExerciseDetailRenderingTests(TestCase):
    """La interfaz distingue ejercicios por repeticiones y por tiempo
    sin mostrar valores absurdos (p. ej. 3 x 0 repeticiones)."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.ex_reps = make_exercise("Sentadilla", muscle="PIERNAS")
        self.ex_time = make_exercise("Plancha", muscle="ABDOMEN")
        self.routine = Routine.objects.create(
            name="Mixta", goal="MANTENIMIENTO", level="INTERMEDIO",
            intensity="BAJA", days_per_week=2,
            estimated_duration_minutes=40,
        )
        RoutineExercise.objects.create(
            routine=self.routine, exercise=self.ex_reps, order=1,
            sets=4, repetitions=10, rest_seconds=90,
        )
        RoutineExercise.objects.create(
            routine=self.routine, exercise=self.ex_time, order=2,
            sets=3, duration_seconds=30, rest_seconds=45,
        )
        self.client.force_login(self.trainer)

    def test_detail_shows_repetitions_for_rep_based_exercise(self):
        response = self.client.get(
            reverse("routines:detail", kwargs={"pk": self.routine.pk})
        )
        self.assertContains(response, "4 series × 10 repeticiones")
        self.assertContains(response, "3 series × 30 segundos")

    def test_time_based_exercise_never_shows_zero_repetitions(self):
        response = self.client.get(
            reverse("routines:detail", kwargs={"pk": self.routine.pk})
        )
        self.assertNotContains(response, "× 0 repeticiones")

    def test_sets_must_be_positive(self):
        item = RoutineExercise(
            routine=self.routine, exercise=self.ex_reps, order=1,
            sets=0, repetitions=10, rest_seconds=60,
        )
        with self.assertRaises(Exception):
            item.full_clean()

    def test_negative_rest_is_invalid(self):
        item = RoutineExercise(
            routine=self.routine, exercise=self.ex_reps, order=1,
            sets=3, repetitions=10, rest_seconds=-5,
        )
        with self.assertRaises(Exception):
            item.full_clean()


class SeedDataTests(TestCase):
    """Carga inicial de catálogo y plantillas (load_initial_data)."""

    EXPECTED_TEMPLATES = (
        "Push Pull Legs (PPL)",
        "Upper / Lower",
        "Full Body",
        "Torso / Pierna",
        "Arnold Split",
    )

    def test_seed_creates_initial_exercise_catalog(self):
        call_command("load_initial_data")
        self.assertGreaterEqual(Exercise.objects.count(), 36)
        self.assertTrue(
            Exercise.objects.filter(name="Sentadilla").exists()
        )
        self.assertTrue(
            Exercise.objects.filter(name="Plancha").exists()
        )

    def test_seed_creates_all_routine_templates(self):
        call_command("load_initial_data")
        for name in self.EXPECTED_TEMPLATES:
            with self.subTest(name=name):
                self.assertTrue(Routine.objects.filter(name=name).exists())

    def test_seed_is_idempotent_no_duplicates(self):
        call_command("load_initial_data")
        counts_before = (
            Exercise.objects.count(),
            Routine.objects.count(),
            RoutineSession.objects.count(),
            RoutineExercise.objects.count(),
        )
        call_command("load_initial_data")
        call_command("load_initial_data")
        counts_after = (
            Exercise.objects.count(),
            Routine.objects.count(),
            RoutineSession.objects.count(),
            RoutineExercise.objects.count(),
        )
        self.assertEqual(counts_before, counts_after)

    def test_every_template_has_sessions(self):
        call_command("load_initial_data")
        expected_sessions = {
            "Push Pull Legs (PPL)": 3,
            "Upper / Lower": 2,
            "Full Body": 1,
            "Torso / Pierna": 2,
            "Arnold Split": 3,
        }
        for name, count in expected_sessions.items():
            with self.subTest(name=name):
                routine = Routine.objects.get(name=name)
                self.assertEqual(routine.sessions.count(), count)

    def test_sessions_contain_exercises(self):
        call_command("load_initial_data")
        for session in RoutineSession.objects.all():
            with self.subTest(session=str(session)):
                self.assertGreater(session.routine_exercises.count(), 0)

    def test_seed_uses_existing_catalog_no_copies(self):
        call_command("load_initial_data")
        total_before = Exercise.objects.count()
        push = Routine.objects.get(name="Push Pull Legs (PPL)")
        for item in push.routine_exercises.all():
            self.assertIn(item.exercise, Exercise.objects.all())
        call_command("load_initial_data")
        self.assertEqual(Exercise.objects.count(), total_before)

    def test_sets_reps_duration_rest_persisted(self):
        call_command("load_initial_data")
        ppl = Routine.objects.get(name="Push Pull Legs (PPL)")
        press = ppl.routine_exercises.get(exercise__name="Press de banca")
        self.assertEqual((press.sets, press.repetitions,
                          press.rest_seconds), (4, 8, 120))
        self.assertIsNone(press.duration_seconds)
        plancha = ppl.routine_exercises.get(exercise__name="Plancha")
        self.assertEqual(
            (plancha.sets, plancha.duration_seconds, plancha.rest_seconds),
            (3, 40, 45),
        )
        self.assertIsNone(plancha.repetitions)

    def test_existing_data_is_preserved(self):
        custom_exercise = Exercise.objects.create(
            name="Mi ejercicio personalizado", muscle_group="OTRO",
            difficulty="PRINCIPIANTE",
        )
        custom_routine = Routine.objects.create(
            name="Rutina propia", goal="PERDIDA_PESO", level="PRINCIPIANTE",
            intensity="BAJA", days_per_week=2, estimated_duration_minutes=30,
        )
        call_command("load_initial_data")
        custom_exercise.refresh_from_db()
        custom_routine.refresh_from_db()
        self.assertEqual(custom_exercise.name, "Mi ejercicio personalizado")
        self.assertEqual(custom_routine.name, "Rutina propia")
        self.assertTrue(custom_routine.is_active)

    def test_seed_does_not_overwrite_edited_templates(self):
        call_command("load_initial_data")
        ppl = Routine.objects.get(name="Push Pull Legs (PPL)")
        ppl.estimated_duration_minutes = 75
        ppl.save()
        call_command("load_initial_data")
        ppl.refresh_from_db()
        self.assertEqual(ppl.estimated_duration_minutes, 75)

    def test_detail_groups_exercises_by_session(self):
        call_command("load_initial_data")
        trainer = make_user("trainer", "ENTRENADOR")
        self.client.force_login(trainer)
        ppl = Routine.objects.get(name="Push Pull Legs (PPL)")
        response = self.client.get(
            reverse("routines:detail", kwargs={"pk": ppl.pk})
        )
        self.assertContains(response, "Día 1 — Push")
        self.assertContains(response, "Día 2 — Pull")
        self.assertContains(response, "Día 3 — Legs")
        self.assertContains(response, "4 series × 8 repeticiones")
        self.assertContains(response, "3 series × 40 segundos")

    def test_flat_routine_without_sessions_still_renders(self):
        trainer = make_user("trainer", "ENTRENADOR")
        self.client.force_login(trainer)
        ex = Exercise.objects.create(
            name="Remo", muscle_group="ESPALDA", difficulty="PRINCIPIANTE"
        )
        routine = Routine.objects.create(
            name="Plana", goal="MANTENIMIENTO", level="PRINCIPIANTE",
        )
        RoutineExercise.objects.create(
            routine=routine, exercise=ex, order=1, sets=3,
            repetitions=10, rest_seconds=60,
        )
        response = self.client.get(
            reverse("routines:detail", kwargs={"pk": routine.pk})
        )
        self.assertContains(response, "Ejercicios")
        self.assertContains(response, "3 series × 10 repeticiones")


class RoutineSessionModelTests(TestCase):
    """Integridad del modelo de sesiones."""

    def setUp(self):
        self.routine = Routine.objects.create(
            name="R", goal="MANTENIMIENTO", level="INTERMEDIO",
        )
        self.exercise = Exercise.objects.create(
            name="Remo", muscle_group="ESPALDA", difficulty="PRINCIPIANTE"
        )

    def test_session_name_unique_per_routine(self):
        RoutineSession.objects.create(routine=self.routine, name="Push")
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            RoutineSession.objects.create(routine=self.routine, name="Push")

    def test_session_must_belong_to_item_routine(self):
        other = Routine.objects.create(
            name="Otra", goal="MANTENIMIENTO", level="PRINCIPIANTE",
        )
        foreign_session = RoutineSession.objects.create(
            routine=other, name="S1"
        )
        item = RoutineExercise(
            routine=self.routine,
            exercise=self.exercise,
            session=foreign_session,
            sets=3,
            repetitions=10,
        )
        with self.assertRaises(Exception):
            item.full_clean()


# =============================================================================
# Pruebas de UserRoutineAssignment (asignaciones de rutina a usuarios)
# =============================================================================


def assignment_payload(**overrides):
    """Payload base para crear una asignación."""
    from datetime import date

    data = {
        "routine": "",  # Se llenará en setUp
        "start_date": date.today().isoformat(),
        "end_date": "",
        "status": "ASIGNADA",
        "initial_evaluation": "",
        "final_evaluation": "",
        "planned_sessions": "20",
        "completed_sessions": "18",
        "observations": "Test observations",
    }
    data.update(overrides)
    return data


class UserRoutineAssignmentModelTests(TestCase):
    """Validaciones del modelo UserRoutineAssignment."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("client", "USUARIO")
        self.other_user = make_user("other", "USUARIO")
        self.routine = Routine.objects.create(
            name="Test Routine", goal="GANANCIA_MUSCULAR", level="PRINCIPIANTE"
        )

    def test_create_assignment_basic(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        self.assertEqual(assignment.user, self.client_user)
        self.assertEqual(assignment.routine, self.routine)
        self.assertEqual(assignment.trainer, self.trainer)
        self.assertEqual(assignment.status, "ASIGNADA")

    def test_str_representation(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        self.assertIn("Test Routine", str(assignment))
        self.assertIn("client", str(assignment))

    def test_adherence_percentage_calculated(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            planned_sessions=20,
            completed_sessions=18,
        )
        self.assertEqual(assignment.adherence_percentage, 90.0)

    def test_adherence_percentage_none_when_zero_planned(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            planned_sessions=0,
            completed_sessions=5,
        )
        self.assertIsNone(assignment.adherence_percentage)

    def test_adherence_percentage_none_when_no_planned(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        self.assertIsNone(assignment.adherence_percentage)

    def test_is_active_property(self):
        for status in ("ASIGNADA", "EN_PROGRESO"):
            assignment = UserRoutineAssignment.objects.create(
                user=self.client_user,
                routine=self.routine,
                trainer=self.trainer,
                start_date="2026-01-01",
                status=status,
            )
            self.assertTrue(assignment.is_active)

        for status in ("FINALIZADA", "CANCELADA"):
            assignment = UserRoutineAssignment.objects.create(
                user=self.client_user,
                routine=self.routine,
                trainer=self.trainer,
                start_date="2026-01-01",
                status=status,
            )
            self.assertFalse(assignment.is_active)


class UserRoutineAssignmentValidationTests(TestCase):
    """Validaciones de integridad del modelo."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("client", "USUARIO")
        self.other_user = make_user("other", "USUARIO")
        self.other_trainer = make_user("trainer2", "ENTRENADOR")
        self.routine = Routine.objects.create(
            name="Test Routine", goal="GANANCIA_MUSCULAR", level="PRINCIPIANTE"
        )
        # Crear evaluaciones para tests
        from apps.assessments.models import PhysicalEvaluation
        self.initial_eval = PhysicalEvaluation.objects.create(
            user=self.client_user,
            trainer=self.trainer,
            evaluation_type="INICIAL",
        )
        self.final_eval = PhysicalEvaluation.objects.create(
            user=self.client_user,
            trainer=self.trainer,
            evaluation_type="FINAL",
        )
        self.other_initial_eval = PhysicalEvaluation.objects.create(
            user=self.other_user,
            trainer=self.trainer,
            evaluation_type="INICIAL",
        )
        self.seguimiento_eval = PhysicalEvaluation.objects.create(
            user=self.client_user,
            trainer=self.trainer,
            evaluation_type="SEGUIMIENTO",
        )

    def test_end_date_before_start_date_rejected(self):
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-02-01",
            end_date="2026-01-01",
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_trainer_must_be_entrenador(self):
        plain_user = make_user("plain", "USUARIO")
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=plain_user,
            start_date="2026-01-01",
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_user_must_be_usuario(self):
        assignment = UserRoutineAssignment(
            user=self.trainer,  # ENTRENADOR como usuario
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_initial_evaluation_must_belong_to_user(self):
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            initial_evaluation=self.other_initial_eval,
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_final_evaluation_must_belong_to_user(self):
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            final_evaluation=self.other_initial_eval,
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_initial_evaluation_must_be_inicial_type(self):
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            initial_evaluation=self.seguimiento_eval,
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_final_evaluation_must_be_final_type(self):
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            final_evaluation=self.initial_eval,
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_completed_sessions_not_exceed_planned(self):
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            planned_sessions=10,
            completed_sessions=15,
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_negative_planned_sessions_rejected(self):
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            planned_sessions=-1,
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_negative_completed_sessions_rejected(self):
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            completed_sessions=-1,
        )
        with self.assertRaises(Exception):
            assignment.full_clean()

    def test_completed_sessions_can_equal_planned(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            planned_sessions=10,
            completed_sessions=10,
        )
        self.assertEqual(assignment.completed_sessions, 10)

    def test_completed_sessions_can_be_zero_with_planned(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            planned_sessions=10,
            completed_sessions=0,
        )
        self.assertEqual(assignment.adherence_percentage, 0.0)


class UserRoutineAssignmentHistoryTests(TestCase):
    """Historial de asignaciones: un usuario puede tener múltiples."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("client", "USUARIO")
        self.routine1 = Routine.objects.create(
            name="Full Body", goal="GANANCIA_MUSCULAR", level="PRINCIPIANTE"
        )
        self.routine2 = Routine.objects.create(
            name="Upper / Lower", goal="GANANCIA_MUSCULAR", level="INTERMEDIO"
        )
        self.routine3 = Routine.objects.create(
            name="Push Pull Legs", goal="GANANCIA_MUSCULAR", level="AVANZADO"
        )

    def test_user_can_have_multiple_assignments(self):
        UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine1,
            trainer=self.trainer,
            start_date="2026-01-01",
            end_date="2026-02-28",
            status="FINALIZADA",
        )
        UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine2,
            trainer=self.trainer,
            start_date="2026-03-01",
            end_date="2026-04-30",
            status="FINALIZADA",
        )
        UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine3,
            trainer=self.trainer,
            start_date="2026-05-01",
            status="EN_PROGRESO",
        )
        self.assertEqual(self.client_user.routine_assignments.count(), 3)

    def test_new_assignment_does_not_delete_previous(self):
        old = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine1,
            trainer=self.trainer,
            start_date="2026-01-01",
            status="FINALIZADA",
        )
        UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine2,
            trainer=self.trainer,
            start_date="2026-03-01",
            status="EN_PROGRESO",
        )
        old.refresh_from_db()
        self.assertEqual(old.status, "FINALIZADA")
        self.assertEqual(UserRoutineAssignment.objects.count(), 2)


class UserRoutineAssignmentViewTests(TestCase):
    """Pruebas de vistas de asignaciones."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("client", "USUARIO")
        self.other_user = make_user("other", "USUARIO")
        self.routine = Routine.objects.create(
            name="Test Routine", goal="GANANCIA_MUSCULAR", level="PRINCIPIANTE"
        )

    def test_trainer_can_create_assignment(self):
        self.client.force_login(self.trainer)
        url = reverse("routines:assignment_create", kwargs={"username": "client"})
        response = self.client.post(
            url,
            {
                "routine": self.routine.pk,
                "start_date": "2026-01-01",
                "status": "ASIGNADA",
                "planned_sessions": "20",
                "completed_sessions": "18",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(UserRoutineAssignment.objects.count(), 1)
        assignment = UserRoutineAssignment.objects.first()
        self.assertEqual(assignment.user, self.client_user)
        self.assertEqual(assignment.trainer, self.trainer)

    def test_user_cannot_create_assignment(self):
        self.client.force_login(self.client_user)
        url = reverse("routines:assignment_create", kwargs={"username": "client"})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)
        self.client.post(
            url,
            {"routine": self.routine.pk, "start_date": "2026-01-01"},
        )
        self.assertEqual(UserRoutineAssignment.objects.count(), 0)

    def test_anonymous_cannot_create_assignment(self):
        url = reverse("routines:assignment_create", kwargs={"username": "client"})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_user_can_see_own_assignments(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        self.client.force_login(self.client_user)
        response = self.client.get(reverse("routines:assignment_list"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["assignments"]), 1)
        self.assertEqual(response.context["assignments"][0], assignment)

    def test_user_cannot_see_other_assignments(self):
        UserRoutineAssignment.objects.create(
            user=self.other_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        self.client.force_login(self.client_user)
        response = self.client.get(reverse("routines:assignment_list"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["assignments"]), 0)

    def test_trainer_can_see_all_assignments(self):
        UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        UserRoutineAssignment.objects.create(
            user=self.other_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-02-01",
        )
        self.client.force_login(self.trainer)
        response = self.client.get(reverse("routines:assignment_list"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["assignments"]), 2)

    def test_user_can_see_own_assignment_detail(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        self.client.force_login(self.client_user)
        url = reverse("routines:assignment_detail", kwargs={"pk": assignment.pk})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["assignment"], assignment)

    def test_user_cannot_see_other_assignment_detail(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.other_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        self.client.force_login(self.client_user)
        url = reverse("routines:assignment_detail", kwargs={"pk": assignment.pk})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)

    def test_trainer_can_see_any_assignment_detail(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.other_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        self.client.force_login(self.trainer)
        url = reverse("routines:assignment_detail", kwargs={"pk": assignment.pk})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_anonymous_redirected_to_login(self):
        assignment = UserRoutineAssignment.objects.create(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        url = reverse("routines:assignment_detail", kwargs={"pk": assignment.pk})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_assignment_uses_existing_routine(self):
        self.client.force_login(self.trainer)
        url = reverse("routines:assignment_create", kwargs={"username": "client"})
        response = self.client.post(
            url,
            {
                "routine": self.routine.pk,
                "start_date": "2026-01-01",
                "status": "ASIGNADA",
                "planned_sessions": "0",
                "completed_sessions": "0",
            },
        )
        self.assertEqual(response.status_code, 302)
        assignment = UserRoutineAssignment.objects.first()
        self.assertEqual(assignment.routine, self.routine)
        # No se duplica la rutina
        self.assertEqual(Routine.objects.count(), 1)


# =============================================================================
# Pruebas de regresión para transacciones atómicas y validaciones
# =============================================================================


class RoutineAtomicTransactionTests(TestCase):
    """Verifica que las operaciones de rutina + ejercicios sean atómicas."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.ex1 = make_exercise("Press de banca")
        self.ex2 = make_exercise("Sentadilla", muscle="PIERNAS")
        self.client.force_login(self.trainer)

    def _routine_data(self, **overrides):
        data = {
            "name": "Rutina atómica",
            "description": "Test",
            "goal": "GANANCIA_MUSCULAR",
            "level": "PRINCIPIANTE",
            "intensity": "MODERADA",
            "days_per_week": "3",
            "estimated_duration_minutes": "50",
        }
        data.update(overrides)
        return data

    def test_create_routine_rollback_on_formset_validation_error(self):
        """Si el formset falla, no se crea la rutina (rollback atómico)."""
        # Payload con ejercicio duplicado en el formset -> error de validación
        payload = {
            **self._routine_data(),
            **formset_payload(
                (self.ex1.pk, 1, 3, 10, None, 60),
                (self.ex1.pk, 2, 3, 8, None, 60),  # Duplicado
            ),
        }
        response = self.client.post(reverse("routines:create"), payload)
        self.assertEqual(response.status_code, 200)
        # No debe crear rutina ni ejercicios de rutina
        self.assertEqual(Routine.objects.count(), 0)
        self.assertEqual(RoutineExercise.objects.count(), 0)

    def test_update_routine_rollback_on_formset_validation_error(self):
        """Si el formset falla en actualización, no se modifica la rutina."""
        routine = Routine.objects.create(
            name="Original", goal="MANTENIMIENTO", level="INTERMEDIO"
        )
        RoutineExercise.objects.create(
            routine=routine, exercise=self.ex1, order=1, sets=3, repetitions=10, rest_seconds=60
        )
        original_name = routine.name

        # Payload con error de validación (repeticiones + duración)
        payload = {
            **self._routine_data(name="Editada"),
            **formset_payload((self.ex1.pk, 1, 3, 10, 30, 60)),  # reps + duración
        }
        response = self.client.post(
            reverse("routines:update", kwargs={"pk": routine.pk}), payload
        )
        self.assertEqual(response.status_code, 200)
        routine.refresh_from_db()
        # La rutina no debe haber cambiado
        self.assertEqual(routine.name, original_name)


class AssignmentAtomicTransactionTests(TestCase):
    """Verifica transacciones atómicas en asignaciones."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("client", "USUARIO")
        self.other_trainer = make_user("trainer2", "ENTRENADOR")
        self.routine = Routine.objects.create(
            name="Test Routine", goal="GANANCIA_MUSCULAR", level="PRINCIPIANTE"
        )
        self.client.force_login(self.trainer)

    def _assignment_payload(self, **overrides):
        from datetime import date
        data = {
            "routine": self.routine.pk,
            "start_date": date.today().isoformat(),
            "end_date": "",
            "status": "ASIGNADA",
            "initial_evaluation": "",
            "final_evaluation": "",
            "planned_sessions": "20",
            "completed_sessions": "18",
            "observations": "Test",
        }
        data.update(overrides)
        return data

    def test_create_assignment_validates_trainer_role(self):
        """El formulario rechaza asignaciones con trainer no ENTRENADOR."""
        # El formulario asigna trainer=request.user (que es ENTRENADOR),
        # pero probamos la validación a nivel de modelo directamente
        plain_user = make_user("plain", "USUARIO")
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=plain_user,  # USUARIO, no ENTRENADOR
            start_date="2026-01-01",
        )
        with self.assertRaises(ValidationError):
            assignment.full_clean()

    def test_create_assignment_validates_user_role(self):
        """El formulario rechaza asignaciones con usuario no USUARIO."""
        assignment = UserRoutineAssignment(
            user=self.trainer,  # ENTRENADOR como usuario
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
        )
        with self.assertRaises(ValidationError):
            assignment.full_clean()

    def test_create_assignment_validates_evaluation_types(self):
        """Valida que evaluaciones inicial/final sean del tipo correcto."""
        from apps.assessments.models import PhysicalEvaluation
        seguimiento = PhysicalEvaluation.objects.create(
            user=self.client_user, trainer=self.trainer, evaluation_type="SEGUIMIENTO"
        )
        assignment = UserRoutineAssignment(
            user=self.client_user,
            routine=self.routine,
            trainer=self.trainer,
            start_date="2026-01-01",
            initial_evaluation=seguimiento,  # Tipo incorrecto
        )
        with self.assertRaises(ValidationError):
            assignment.full_clean()

    def test_form_save_runs_model_clean(self):
        """Verifica que form.save() ejecuta model.clean() (validaciones de rol, fechas, etc.)."""
        from apps.assessments.models import PhysicalEvaluation
        initial_eval = PhysicalEvaluation.objects.create(
            user=self.client_user, trainer=self.trainer, evaluation_type="INICIAL"
        )
        final_eval = PhysicalEvaluation.objects.create(
            user=self.client_user, trainer=self.trainer, evaluation_type="FINAL"
        )

        # Payload válido
        payload = self._assignment_payload(
            initial_evaluation=initial_eval.pk,
            final_evaluation=final_eval.pk,
        )
        response = self.client.post(
            reverse("routines:assignment_create", kwargs={"username": "client"}), payload
        )
        # Debug: si falla, mostrar errores
        if response.status_code != 302:
            print(f"Response status: {response.status_code}")
            if hasattr(response, 'context') and response.context and 'form' in response.context:
                print(f"Form errors: {response.context['form'].errors}")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(UserRoutineAssignment.objects.count(), 1)

        # Payload con evaluación final de tipo incorrecto
        wrong_final = PhysicalEvaluation.objects.create(
            user=self.client_user, trainer=self.trainer, evaluation_type="INICIAL"
        )
        payload = self._assignment_payload(
            initial_evaluation=initial_eval.pk,
            final_evaluation=wrong_final.pk,
        )
        response = self.client.post(
            reverse("routines:assignment_create", kwargs={"username": "client"}), payload
        )
        # Debe mostrar error de validación, no crear asignación
        self.assertEqual(response.status_code, 200)
        self.assertEqual(UserRoutineAssignment.objects.count(), 1)  # Solo la primera


class RoutineExerciseModelValidationTests(TestCase):
    """Validaciones del modelo RoutineExercise."""

    def setUp(self):
        self.routine = Routine.objects.create(
            name="R", goal="MANTENIMIENTO", level="INTERMEDIO"
        )
        self.other_routine = Routine.objects.create(
            name="Otra", goal="MANTENIMIENTO", level="PRINCIPIANTE"
        )
        self.exercise = Exercise.objects.create(
            name="Remo", muscle_group="ESPALDA", difficulty="PRINCIPIANTE"
        )

    def test_session_must_belong_to_same_routine(self):
        """RoutineExercise.clean() valida que la sesión pertenezca a la rutina."""
        foreign_session = RoutineSession.objects.create(
            routine=self.other_routine, name="S1"
        )
        item = RoutineExercise(
            routine=self.routine,
            exercise=self.exercise,
            session=foreign_session,
            sets=3,
            repetitions=10,
        )
        with self.assertRaises(ValidationError):
            item.full_clean()

    def test_repetitions_or_duration_required(self):
        """Debe tener repeticiones O duración, no ninguna ni ambas."""
        # Sin ninguna
        item = RoutineExercise(
            routine=self.routine, exercise=self.exercise, sets=3
        )
        with self.assertRaises(ValidationError):
            item.full_clean()

        # Con ambas
        item = RoutineExercise(
            routine=self.routine,
            exercise=self.exercise,
            sets=3,
            repetitions=10,
            duration_seconds=30,
        )
        with self.assertRaises(ValidationError):
            item.full_clean()

        # Solo repeticiones - válido
        item = RoutineExercise(
            routine=self.routine, exercise=self.exercise, sets=3, repetitions=10
        )
        item.full_clean()  # No debe lanzar

        # Solo duración - válido
        item = RoutineExercise(
            routine=self.routine, exercise=self.exercise, sets=3, duration_seconds=30
        )
        item.full_clean()  # No debe lanzar


class EvaluationAtomicTransactionTests(TestCase):
    """Verifica transacciones atómicas en evaluaciones."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("client", "USUARIO")
        self.client.force_login(self.trainer)

    def test_create_evaluation_rollback_on_formset_error(self):
        """Si el formset de resultados falla, no se crea la evaluación."""
        payload = {
            "evaluation_type": "INICIAL",
            "observations": "",
            "results-TOTAL_FORMS": "2",
            "results-INITIAL_FORMS": "0",
            "results-MIN_NUM_FORMS": "0",
            "results-MAX_NUM_FORMS": "1000",
            "results-0-test_name": "Flexiones",
            "results-0-value": "-5",  # Valor negativo -> error
            "results-0-unit": "repeticiones",
            "results-0-notes": "",
            "results-1-test_name": "Plancha",
            "results-1-value": "60",
            "results-1-unit": "segundos",
            "results-1-notes": "",
        }
        response = self.client.post(
            reverse("assessments:create", kwargs={"username": "client"}), payload
        )
        self.assertEqual(response.status_code, 200)
        # No debe crear evaluación ni resultados
        from apps.assessments.models import PhysicalEvaluation, EvaluationResult
        self.assertEqual(PhysicalEvaluation.objects.count(), 0)
        self.assertEqual(EvaluationResult.objects.count(), 0)

    def test_create_evaluation_needs_at_least_one_result(self):
        """Una evaluación requiere al menos un resultado."""
        payload = {
            "evaluation_type": "INICIAL",
            "observations": "",
            "results-TOTAL_FORMS": "0",
            "results-INITIAL_FORMS": "0",
            "results-MIN_NUM_FORMS": "0",
            "results-MAX_NUM_FORMS": "1000",
        }
        response = self.client.post(
            reverse("assessments:create", kwargs={"username": "client"}), payload
        )
        self.assertEqual(response.status_code, 200)
        from apps.assessments.models import PhysicalEvaluation
        self.assertEqual(PhysicalEvaluation.objects.count(), 0)
