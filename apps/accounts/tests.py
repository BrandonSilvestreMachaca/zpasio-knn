"""Pruebas de autenticación, roles y permisos de Zpasio."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

User = get_user_model()


class PasswordStorageTest(TestCase):
    """Verifica que las contraseñas se guarden hasheadas y no en texto plano."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="cliente",
            email="cliente@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )

    def test_password_is_hashed_not_plaintext(self):
        self.assertNotEqual(self.user.password, "Contrasena123!")
        self.assertTrue(self.user.password.startswith("pbkdf2_sha256"))

    def test_check_password_returns_true(self):
        self.assertTrue(self.user.check_password("Contrasena123!"))

    def test_check_password_returns_false_for_wrong(self):
        self.assertFalse(self.user.check_password("incorrecta"))


class EmailLoginTest(TestCase):
    """El login debe aceptar tanto usuario como email."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="cliente",
            email="cliente@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        self.login_url = reverse("accounts:login")

    def test_login_with_username(self):
        response = self.client.post(
            self.login_url,
            {"username": "cliente", "password": "Contrasena123!"},
        )
        self.assertEqual(response.status_code, 302)

    def test_login_with_email(self):
        response = self.client.post(
            self.login_url,
            {"username": "cliente@example.com", "password": "Contrasena123!"},
        )
        self.assertEqual(response.status_code, 302)

    def test_login_with_email_case_insensitive(self):
        response = self.client.post(
            self.login_url,
            {"username": "CLIENTE@EXAMPLE.COM", "password": "Contrasena123!"},
        )
        self.assertEqual(response.status_code, 302)


class RegisterTest(TestCase):
    def setUp(self):
        self.url = reverse("accounts:register")

    def _post(self, **overrides):
        data = {
            "username": "nuevo_cliente",
            "first_name": "Ana",
            "last_name": "Pérez",
            "email": "ana@example.com",
            "password1": "Contrasena123!",
            "password2": "Contrasena123!",
        }
        data.update(overrides)
        return self.client.post(self.url, data)

    def test_register_creates_user(self):
        response = self._post()
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username="nuevo_cliente").exists())

    def test_register_sets_role_usuario(self):
        self._post()
        user = User.objects.get(username="nuevo_cliente")
        self.assertEqual(user.role, User.Role.USUARIO)

    def test_register_duplicate_email_rejected(self):
        self._post()
        self.client.logout()
        response = self._post(username="otro_cliente")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="otro_cliente").exists())

    def test_register_normalizes_email(self):
        self._post(email="ANA.PEREZ@Example.COM")
        user = User.objects.get(username="nuevo_cliente")
        self.assertEqual(user.email, "ana.perez@example.com")


class LoginLogoutTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="cliente",
            email="cliente@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        self.login_url = reverse("accounts:login")
        self.logout_url = reverse("accounts:logout")

    def test_login_success_redirects(self):
        response = self.client.post(
            self.login_url,
            {"username": "cliente", "password": "Contrasena123!"},
        )
        self.assertEqual(response.status_code, 302)

    def test_login_wrong_password(self):
        response = self.client.post(
            self.login_url,
            {"username": "cliente", "password": "incorrecta"},
        )
        self.assertEqual(response.status_code, 200)

    def test_logout(self):
        self.client.login(username="cliente", password="Contrasena123!")
        response = self.client.post(self.logout_url)
        self.assertEqual(response.status_code, 302)
        # Tras cerrar sesión, acceder al perfil redirige al login.
        self.assertRedirects(response, self.login_url)


class PermissionTest(TestCase):
    def setUp(self):
        self.entrenador = User.objects.create_user(
            username="entrenador",
            email="entrenador@example.com",
            password="Contrasena123!",
            role=User.Role.ENTRENADOR,
        )
        self.cliente = User.objects.create_user(
            username="cliente",
            email="cliente@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        self.client_list_url = reverse("profiles:client_list")

    def test_anonymous_redirected_to_login(self):
        response = self.client.get(self.client_list_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("accounts:login"), response.url)

    def test_usuario_cannot_access_client_list(self):
        self.client.login(username="cliente", password="Contrasena123!")
        response = self.client.get(self.client_list_url)
        self.assertEqual(response.status_code, 403)

    def test_entrenador_can_access_client_list(self):
        self.client.login(username="entrenador", password="Contrasena123!")
        response = self.client.get(self.client_list_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "cliente")

    def test_usuario_cannot_view_other_client_detail(self):
        otro = User.objects.create_user(
            username="otro",
            email="otro@example.com",
            password="Contrasena123!",
            role=User.Role.USUARIO,
        )
        self.client.login(username="cliente", password="Contrasena123!")
        url = reverse("profiles:client_detail", args=[otro.username])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)