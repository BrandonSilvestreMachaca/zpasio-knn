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