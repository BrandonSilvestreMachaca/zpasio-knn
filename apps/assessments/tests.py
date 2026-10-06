"""Pruebas del módulo de evaluaciones físicas de Zpasio."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from .models import EvaluationResult, PhysicalEvaluation

User = get_user_model()


def make_user(username, role):
    return User.objects.create_user(
        username=username, password="testpass123", role=role
    )


def results_payload(*rows):
    """Payload del formset de resultados: (prueba, valor, unidad, nota)."""
    data = {
        "results-TOTAL_FORMS": str(len(rows)),
        "results-INITIAL_FORMS": "0",
        "results-MIN_NUM_FORMS": "0",
        "results-MAX_NUM_FORMS": "1000",
    }
    for i, (name, value, unit, notes) in enumerate(rows):
        data[f"results-{i}-test_name"] = name
        data[f"results-{i}-value"] = str(value)
        data[f"results-{i}-unit"] = unit
        data[f"results-{i}-notes"] = notes
    return data


class EvaluationCreationTests(TestCase):
    """El entrenador registra evaluaciones; el usuario no."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("cliente", "USUARIO")
        self.create_url = reverse(
            "assessments:create", kwargs={"username": "cliente"}
        )

    def _payload(self, **overrides):
        data = {
            "evaluation_type": "INICIAL",
            "observations": "",
            **results_payload(
                ("Flexiones de brazos", "20", "repeticiones", ""),
                ("Prueba de plancha", "60", "segundos", ""),
            ),
        }
        data.update(overrides)
        return data

    def test_trainer_can_create_evaluation_with_results(self):
        self.client.force_login(self.trainer)
        response = self.client.post(self.create_url, self._payload())
        evaluation = PhysicalEvaluation.objects.get()
        self.assertRedirects(
            response,
            reverse(
                "profiles:client_detail", kwargs={"username": "cliente"}
            ),
        )
        self.assertEqual(evaluation.user, self.client_user)
        self.assertEqual(evaluation.trainer, self.trainer)
        self.assertEqual(evaluation.evaluation_type, "INICIAL")
        self.assertEqual(evaluation.results.count(), 2)
        r = evaluation.results.get(test_name="Flexiones de brazos")
        self.assertEqual(str(r.value), "20.00")
        self.assertEqual(r.unit, "repeticiones")

    def test_user_cannot_create_evaluation(self):
        self.client.force_login(self.client_user)
        response = self.client.get(self.create_url)
        self.assertEqual(response.status_code, 403)
        self.client.post(self.create_url, self._payload())
        self.assertEqual(PhysicalEvaluation.objects.count(), 0)

    def test_anonymous_redirected_to_login(self):
        response = self.client.get(self.create_url)
        self.assertEqual(response.status_code, 302)

    def test_evaluation_needs_at_least_one_result(self):
        self.client.force_login(self.trainer)
        payload = self._payload()
        payload["results-TOTAL_FORMS"] = "0"
        payload.pop("results-0-test_name", None)
        payload.pop("results-0-value", None)
        payload.pop("results-0-unit", None)
        payload.pop("results-0-notes", None)
        payload.pop("results-1-test_name", None)
        payload.pop("results-1-value", None)
        payload.pop("results-1-unit", None)
        payload.pop("results-1-notes", None)
        response = self.client.post(self.create_url, payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(PhysicalEvaluation.objects.count(), 0)

    def test_negative_result_value_rejected(self):
        self.client.force_login(self.trainer)
        payload = self._payload()
        payload["results-0-value"] = "-5"
        self.client.post(self.create_url, payload)
        self.assertEqual(PhysicalEvaluation.objects.count(), 0)

    def test_cannot_create_for_another_trainer_role(self):
        other_trainer = make_user("trainer2", "ENTRENADOR")
        response = self.client.get(
            reverse(
                "assessments:create", kwargs={"username": other_trainer.username}
            )
        )
        self.assertEqual(response.status_code, 302)  # no autenticado
        self.client.force_login(self.trainer)
        response = self.client.get(
            reverse(
                "assessments:create", kwargs={"username": other_trainer.username}
            )
        )
        self.assertEqual(response.status_code, 404)


class EvaluationHistoryTests(TestCase):
    """Múltiples evaluaciones por usuario sin sobrescribir anteriores."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("cliente", "USUARIO")

    def _create(self, type_="INICIAL"):
        evaluation = PhysicalEvaluation.objects.create(
            user=self.client_user,
            trainer=self.trainer,
            evaluation_type=type_,
        )
        EvaluationResult.objects.create(
            evaluation=evaluation,
            test_name="Flexiones",
            value=10,
            unit="repeticiones",
        )
        return evaluation

    def test_user_can_have_multiple_evaluations(self):
        self._create("INICIAL")
        self._create("SEGUIMIENTO")
        self._create("FINAL")
        self.assertEqual(
            self.client_user.physical_evaluations.count(), 3
        )

    def test_new_evaluation_does_not_delete_previous(self):
        old = self._create("INICIAL")
        self._create("SEGUIMIENTO")
        old.refresh_from_db()
        self.assertEqual(old.results.count(), 1)
        self.assertEqual(
            PhysicalEvaluation.objects.count(), 2
        )

    def test_results_belong_to_their_evaluation(self):
        e1 = self._create("INICIAL")
        e2 = self._create("FINAL")
        self.assertNotEqual(
            list(e1.results.values_list("id", flat=True)),
            list(e2.results.values_list("id", flat=True)),
        )
        self.assertEqual(e1.results.get().evaluation, e1)

    def test_evaluation_typing_choices(self):
        response = self._create("SEGUIMIENTO")
        self.assertEqual(
            response.get_evaluation_type_display(), "Seguimiento"
        )


class EvaluationPermissionTests(TestCase):
    """Control de acceso al detalle y al historial."""

    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.owner = make_user("dueno", "USUARIO")
        self.other = make_user("otro", "USUARIO")
        self.evaluation = PhysicalEvaluation.objects.create(
            user=self.owner,
            trainer=self.trainer,
            evaluation_type="INICIAL",
        )
        EvaluationResult.objects.create(
            evaluation=self.evaluation,
            test_name="Plancha",
            value=60,
            unit="segundos",
        )
        self.detail_url = reverse(
            "assessments:detail", kwargs={"pk": self.evaluation.pk}
        )

    def test_owner_can_see_own_evaluation(self):
        self.client.force_login(self.owner)
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Plancha")
        self.assertContains(response, "60")
        self.assertContains(response, "segundos")

    def test_other_user_cannot_see_foreign_evaluation(self):
        self.client.force_login(self.other)
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, 403)

    def test_trainer_can_see_any_evaluation(self):
        self.client.force_login(self.trainer)
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, 200)

    def test_anonymous_redirected(self):
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, 302)

    def test_my_evaluations_lists_only_own(self):
        PhysicalEvaluation.objects.create(
            user=self.other,
            trainer=self.trainer,
            evaluation_type="INICIAL",
        )
        self.client.force_login(self.owner)
        response = self.client.get(reverse("assessments:list"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["evaluations"]), 1)

    def test_client_detail_shows_evaluations_history(self):
        self.client.force_login(self.trainer)
        response = self.client.get(
            reverse("profiles:client_detail", kwargs={"username": "dueno"})
        )
        self.assertContains(response, "Evaluaciones físicas")
        self.assertContains(response, "Inicial")


class EvaluationModelValidationTests(TestCase):
    def setUp(self):
        self.trainer = make_user("trainer", "ENTRENADOR")
        self.client_user = make_user("cliente", "USUARIO")

    def test_trainer_must_have_trainer_role(self):
        evaluation = PhysicalEvaluation(
            user=self.client_user,
            trainer=self.client_user,
        )
        with self.assertRaises(Exception):
            evaluation.full_clean()

    def test_user2_trainer_must_be_trainer(self):
        plain = make_user("plain", "USUARIO")
        evaluation = PhysicalEvaluation(
            user=self.client_user,
            trainer=plain,
        )
        with self.assertRaises(Exception):
            evaluation.full_clean()

    def test_negative_result_invalid(self):
        evaluation = PhysicalEvaluation.objects.create(
            user=self.client_user, trainer=self.trainer
        )
        result = EvaluationResult(
            evaluation=evaluation,
            test_name="X",
            value=-1,
            unit="repeticiones",
        )
        with self.assertRaises(Exception):
            result.full_clean()
