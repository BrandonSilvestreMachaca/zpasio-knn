/* Zpasio — mostrar/ocultar contraseña sin guardarla ni exponerla. */

document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".password-field").forEach(function (wrapper) {
        var input = wrapper.querySelector("input[type='password']");
        var toggle = wrapper.querySelector(".password-toggle");
        if (!input || !toggle) {
            return;
        }

        toggle.addEventListener("click", function () {
            var isHidden = input.type === "password";
            input.type = isHidden ? "text" : "password";
            toggle.textContent = isHidden ? "Ocultar" : "Mostrar";
        });
    });
});