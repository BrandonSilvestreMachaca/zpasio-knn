"""Pruebas del módulo de gestión de ejercicios."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from .forms import ExerciseForm
from .models import Exercise

User = get_user_model()

PASSWORD = "Contrasena123!"


def make_user(username, role):
    return User.objects.create_user(
        username=username,
        email=f"{username}@example.com",
        password=PASSWORD,
        role=role,
    )


def exercise_payload(**overrides):
    data = {
        "name": "Press de banca",
        "description": "Ejercicio básico de empuje horizontal.",
        "muscle_group": Exercise.MuscleGroup.PECHO,
        "difficulty": Exercise.Difficulty.INTERMEDIO,
        "is_active": True,
    }
    data.update(overrides)
    return data


class ExercisePermissionBase(TestCase):
    def setUp(self):
        self.trainer = make_user("entrenador", User.Role.ENTRENADOR)
        self.client_user = make_user("cliente", User.Role.USUARIO)
        self.exercise = Exercise.objects.create(
            name="Sentadilla",
            muscle_group=Exercise.MuscleGroup.PIERNAS,
            difficulty=Exercise.Difficulty.PRINCIPIANTE,
        )


class TrainerCrudTest(ExercisePermissionBase):
    def login(self):
        self.client.login(username="entrenador", password=PASSWORD)

    def test_trainer_can_create_exercise(self):
        self.login()
        response = self.client.post(reverse("exercises:create"), exercise_payload())
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Exercise.objects.filter(name="Press de banca").exists())

    def test_trainer_can_list_exercises(self):
        self.login()
        response = self.client.get(reverse("exercises:list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sentadilla")

    def test_trainer_can_view_detail(self):
        self.login()
        response = self.client.get(reverse("exercises:detail", args=[self.exercise.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sentadilla")

    def test_trainer_can_update_exercise(self):
        self.login()
        response = self.client.post(
            reverse("exercises:update", args=[self.exercise.pk]),
            exercise_payload(name="Sentadilla búlgara"),
        )
        self.assertEqual(response.status_code, 302)
        self.exercise.refresh_from_db()
        self.assertEqual(self.exercise.name, "Sentadilla búlgara")

    def test_trainer_can_delete_exercise(self):
        self.login()
        # GET muestra confirmación
        response = self.client.get(reverse("exercises:delete", args=[self.exercise.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "¿Está seguro")
        # POST confirma y desactiva
        response = self.client.post(
            reverse("exercises:delete", args=[self.exercise.pk]), {"confirm": "1"}
        )
        self.assertEqual(response.status_code, 302)
        self.exercise.refresh_from_db()
        self.assertFalse(self.exercise.is_active)


class UserRestrictionsTest(ExercisePermissionBase):
    def login(self):
        self.client.login(username="cliente", password=PASSWORD)

    def test_user_can_read_list(self):
        self.login()
        response = self.client.get(reverse("exercises:list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sentadilla")

    def test_user_cannot_create(self):
        self.login()
        response = self.client.post(reverse("exercises:create"), exercise_payload())
        self.assertEqual(response.status_code, 403)
        self.assertFalse(Exercise.objects.filter(name="Press de banca").exists())

    def test_user_cannot_update(self):
        self.login()
        response = self.client.post(
            reverse("exercises:update", args=[self.exercise.pk]),
            exercise_payload(name="Hack"),
        )
        self.assertEqual(response.status_code, 403)

    def test_user_cannot_delete(self):
        self.login()
        response = self.client.post(
            reverse("exercises:delete", args=[self.exercise.pk]), {"confirm": "1"}
        )
        self.assertEqual(response.status_code, 403)
        self.exercise.refresh_from_db()
        self.assertTrue(self.exercise.is_active)


class AnonymousAccessTest(TestCase):
    def test_anonymous_redirected_to_login(self):
        for name in ("list", "create"):
            url = reverse(f"exercises:{name}")
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302)
            self.assertIn("login", response.url)


class ExerciseFormValidationTest(TestCase):
    def test_required_fields_validated(self):
        form = ExerciseForm(data={"name": "", "muscle_group": "", "difficulty": ""})
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)
        self.assertIn("muscle_group", form.errors)
        self.assertIn("difficulty", form.errors)

    def test_invalid_choice_rejected(self):
        form = ExerciseForm(data=exercise_payload(muscle_group="INVALIDO"))
        self.assertFalse(form.is_valid())
        self.assertIn("muscle_group", form.errors)

    def test_valid_form_saves(self):
        form = ExerciseForm(data=exercise_payload())
        self.assertTrue(form.is_valid())
        exercise = form.save()
        self.assertEqual(exercise.name, "Press de banca")
        self.assertTrue(Exercise.objects.filter(pk=exercise.pk).exists())
