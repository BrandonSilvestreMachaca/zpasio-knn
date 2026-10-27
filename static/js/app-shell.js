/* Zpasio — menú lateral en viewport estrecho. Sin dependencias. */

(function () {
    var body = document.body;
    var toggle = document.querySelector("[data-nav-toggle]");
    var sidebar = document.getElementById("app-sidebar");
    if (!toggle || !sidebar) {
        return;
    }

    var closeTriggers = document.querySelectorAll("[data-nav-close]");
    var media = window.matchMedia("(max-width: 768px)");

    function setOpen(open) {
        body.classList.toggle("is-nav-open", open);
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
        toggle.setAttribute("aria-label", open ? "Cerrar menú" : "Abrir menú");
        sidebar.setAttribute("aria-hidden", media.matches ? (open ? "false" : "true") : "false");

        if (!media.matches) {
            return;
        }

        if (open) {
            var focusTarget = sidebar.querySelector("a, button");
            if (focusTarget) {
                focusTarget.focus();
            }
        } else {
            toggle.focus();
        }
    }

    toggle.addEventListener("click", function () {
        setOpen(!body.classList.contains("is-nav-open"));
    });

    closeTriggers.forEach(function (el) {
        el.addEventListener("click", function () {
            setOpen(false);
        });
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && body.classList.contains("is-nav-open")) {
            event.preventDefault();
            setOpen(false);
        }
    });

    function syncViewport() {
        if (!media.matches) {
            body.classList.remove("is-nav-open");
            toggle.setAttribute("aria-expanded", "false");
            toggle.setAttribute("aria-label", "Abrir menú");
            sidebar.setAttribute("aria-hidden", "false");
        } else if (!body.classList.contains("is-nav-open")) {
            sidebar.setAttribute("aria-hidden", "true");
        }
    }

    if (typeof media.addEventListener === "function") {
        media.addEventListener("change", syncViewport);
    } else if (typeof media.addListener === "function") {
        media.addListener(syncViewport);
    }
    syncViewport();
})();
