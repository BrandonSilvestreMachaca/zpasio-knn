"""Pruebas de perfil, datos físicos e IMC de Zpasio."""

from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from .models import PhysicalData, Profile

User = get_user_model()


class ProfileAccessTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="cliente",
            email="cliente@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        self.detail_url = reverse("profiles:detail")

    def test_anonymous_redirected(self):
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, 302)

    def test_profile_auto_created(self):
        self.client.login(username="cliente", password="Contrasena123!")
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(Profile.objects.filter(user=self.user).exists())


class ProfileEditTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="cliente",
            email="cliente@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        self.client.login(username="cliente", password="Contrasena123!")
        self.edit_url = reverse("profiles:edit")

    def test_edit_profile(self):
        response = self.client.post(
            self.edit_url,
            {
                "phone": "5551234567",
                "birth_date": "1995-05-20",
                "gender": "FEMENINO",
                "goal": "PERDIDA_PESO",
                "experience_level": "INTERMEDIO",
            },
        )
        self.assertRedirects(response, reverse("profiles:detail"))
        profile = Profile.objects.get(user=self.user)
        self.assertEqual(profile.goal, "PERDIDA_PESO")
        self.assertEqual(profile.experience_level, "INTERMEDIO")

    def test_edit_available_days(self):
        self.client.post(
            self.edit_url,
            {"available_days_per_week": "3"},
        )
        profile = Profile.objects.get(user=self.user)
        self.assertEqual(profile.available_days_per_week, 3)

    def test_available_days_out_of_range_rejected(self):
        for days in ("0", "8"):
            with self.subTest(days=days):
                response = self.client.post(
                    self.edit_url,
                    {"available_days_per_week": days},
                )
                self.assertContains(response, "entre 1 y 7")
        profile = Profile.objects.get(user=self.user)
        self.assertIsNone(profile.available_days_per_week)


class ProfileAvailabilityTest(TestCase):
    """Campo de disponibilidad y cálculo de edad (preparación KNN)."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="cliente",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )

    def test_available_days_can_be_empty(self):
        profile = Profile.objects.create(user=self.user)
        self.assertIsNone(profile.available_days_per_week)

    def test_available_days_model_validation(self):
        profile = Profile(user=self.user, available_days_per_week=0)
        with self.assertRaises(Exception):
            profile.full_clean()

    def test_age_calculated_from_birth_date(self):
        from datetime import date

        profile = Profile(
            user=self.user,
            birth_date=date(date.today().year - 30, 1, 1),
        )
        self.assertIn(profile.age, (29, 30))

    def test_age_is_none_without_birth_date(self):
        profile = Profile(user=self.user)
        self.assertIsNone(profile.age)

    def test_detail_shows_availability_summary(self):
        Profile.objects.create(
            user=self.user,
            goal="GANANCIA_MUSCULAR",
            experience_level="PRINCIPIANTE",
            available_days_per_week=4,
        )
        self.client.login(username="cliente", password="Contrasena123!")
        response = self.client.get(reverse("profiles:detail"))
        self.assertContains(response, "Días disponibles por semana")


class PhysicalDataTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="cliente",
            email="cliente@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        self.client.login(username="cliente", password="Contrasena123!")
        self.edit_url = reverse("profiles:physical_data_edit")

    def test_save_physical_data_calculates_bmi(self):
        response = self.client.post(
            self.edit_url,
            {"weight_kg": "70.00", "height_cm": "175.00"},
        )
        self.assertRedirects(response, reverse("profiles:detail"))
        data = PhysicalData.objects.get(user=self.user)
        self.assertEqual(data.bmi, Decimal("22.86"))

    def test_bmi_not_editable_manually(self):
        # Incluso si se envía un bmi, se ignora y se recalcula.
        self.client.post(
            self.edit_url,
            {"weight_kg": "70.00", "height_cm": "175.00", "bmi": "999"},
        )
        data = PhysicalData.objects.get(user=self.user)
        self.assertEqual(data.bmi, Decimal("22.86"))

    def test_negative_weight_rejected(self):
        response = self.client.post(
            self.edit_url,
            {"weight_kg": "-70.00", "height_cm": "175.00"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(PhysicalData.objects.filter(user=self.user).exists())

    def test_zero_height_rejected(self):
        response = self.client.post(
            self.edit_url,
            {"weight_kg": "70.00", "height_cm": "0"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(PhysicalData.objects.filter(user=self.user).exists())


class BMICalculationTest(TestCase):
    def test_calculate_bmi(self):
        self.assertEqual(
            PhysicalData.calculate_bmi(Decimal("70"), Decimal("175")),
            Decimal("22.86"),
        )

    def test_calculate_bmi_exact_rounding(self):
        self.assertEqual(
            PhysicalData.calculate_bmi(Decimal("80"), Decimal("180")),
            Decimal("24.69"),
        )

    def test_calculate_bmi_missing_data(self):
        self.assertIsNone(PhysicalData.calculate_bmi(0, Decimal("175")))
        self.assertIsNone(PhysicalData.calculate_bmi(Decimal("70"), 0))
        self.assertIsNone(PhysicalData.calculate_bmi(None, Decimal("175")))

    def test_bmi_does_not_use_body_fat(self):
        # La grasa corporal no debe intervenir en el cálculo del IMC.
        self.assertEqual(
            PhysicalData.calculate_bmi(Decimal("70"), Decimal("175")),
            Decimal("22.86"),
        )
        user = User.objects.create_user(
            username="fatcheck",
            email="fat@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        data = PhysicalData.objects.create(
            user=user,
            weight_kg=Decimal("70.00"),
            height_cm=Decimal("175.00"),
            body_fat_percentage=Decimal("90.0"),
        )
        self.assertEqual(data.bmi, Decimal("22.86"))


class BMIClassificationTest(TestCase):
    def _label(self, bmi):
        classification = PhysicalData.classify_bmi(Decimal(bmi))
        return classification[0] if classification else None

    def test_boundary_values(self):
        cases = {
            "17.0": "Bajo peso",
            "17.9": "Bajo peso",
            "18.5": "Peso normal",
            "24.9": "Peso normal",
            "25.0": "Sobrepeso",
            "29.9": "Sobrepeso",
            "30.0": "Obesidad",
            "31.2": "Obesidad",
        }
        for value, expected in cases.items():
            with self.subTest(bmi=value):
                self.assertEqual(self._label(value), expected)

    def test_none_returns_none(self):
        self.assertIsNone(PhysicalData.classify_bmi(None))

    def test_property_matches_calculation(self):
        user = User.objects.create_user(
            username="pesonormal",
            email="pn@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        data = PhysicalData.objects.create(
            user=user, weight_kg=Decimal("70.00"), height_cm=Decimal("175.00")
        )
        self.assertEqual(data.bmi, Decimal("22.86"))
        self.assertEqual(data.bmi_classification, "Peso normal")
        self.assertEqual(data.bmi_classification_css_class, "normal")

    def test_classification_in_detail_view(self):
        user = User.objects.create_user(
            username="vista",
            email="vista@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        PhysicalData.objects.create(
            user=user, weight_kg=Decimal("70.00"), height_cm=Decimal("175.00")
        )
        self.client.login(username="vista", password="Contrasena123!")
        response = self.client.get(reverse("profiles:detail"))
        # La localización en español renderiza el decimal con coma.
        self.assertContains(response, "22,86")
        self.assertContains(response, "Peso normal")
        self.assertContains(response, "Escala de IMC")

class FichaCaracterizacionTests(TestCase):
    """Campos nuevos de la ficha: tiempo disponible, frecuencia previa,
    cintura y grasa corporal (dato medido)."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="cliente", password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        self.client.login(username="cliente", password="Contrasena123!")

    def test_extended_ficha_is_saved(self):
        self.client.post(
            reverse("profiles:edit"),
            {
                "goal": "GANANCIA_MUSCULAR",
                "experience_level": "PRINCIPIANTE",
                "available_days_per_week": "4",
                "session_time_available_minutes": "60",
                "previous_training_days_per_week": "2",
            },
        )
        profile = Profile.objects.get(user=self.user)
        self.assertEqual(profile.session_time_available_minutes, 60)
        self.assertEqual(profile.previous_training_days_per_week, 2)

    def test_session_time_must_be_positive(self):
        response = self.client.post(
            reverse("profiles:edit"),
            {"session_time_available_minutes": "0"},
        )
        self.assertContains(response, "mayor que cero")

    def test_previous_frequency_range_0_to_7(self):
        for days in ("-1", "8"):
            with self.subTest(days=days):
                request_data = {"previous_training_days_per_week": days}
                response = self.client.post(
                    reverse("profiles:edit"), request_data
                )
                profile = Profile.objects.get(user=self.user)
                self.assertIsNone(
                    profile.previous_training_days_per_week
                )
        self.client.post(
            reverse("profiles:edit"),
            {"previous_training_days_per_week": "0"},
        )
        profile = Profile.objects.get(user=self.user)
        self.assertEqual(profile.previous_training_days_per_week, 0)

    def test_waist_and_measured_body_fat_saved(self):
        self.client.post(
            reverse("profiles:physical_data_edit"),
            {
                "weight_kg": "72",
                "height_cm": "175",
                "waist_circumference_cm": "85.5",
                "body_fat_percentage": "18.5",
            },
        )
        data = PhysicalData.objects.get(user=self.user)
        self.assertEqual(str(data.waist_circumference_cm), "85.5")
        self.assertEqual(str(data.body_fat_percentage), "18.5")
        self.assertEqual(str(data.bmi), "23.51")

    def test_negative_waist_rejected(self):
        response = self.client.post(
            reverse("profiles:physical_data_edit"),
            {
                "weight_kg": "72",
                "height_cm": "175",
                "waist_circumference_cm": "-10",
            },
        )
        self.assertContains(response, "positivo")

    def test_body_fat_range(self):
        for fat in ("-5", "120"):
            with self.subTest(fat=fat):
                response = self.client.post(
                    reverse("profiles:physical_data_edit"),
                    {
                        "weight_kg": "72",
                        "height_cm": "175",
                        "body_fat_percentage": fat,
                    },
                )
                self.assertContains(response, "entre 0 y 100")
