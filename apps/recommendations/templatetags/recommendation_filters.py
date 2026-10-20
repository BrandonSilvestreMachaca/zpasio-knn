from django import template

register = template.Library()


@register.filter
def mul(value, arg):
    """Multiplica el valor por el argumento."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value


@register.filter
def porcentaje(value):
    """Convierte una proporción (0-1) a porcentaje con 2 decimales."""
    try:
        return round(float(value) * 100, 2)
    except (ValueError, TypeError):
        return value