from django.contrib import admin

from .models import Routine, RoutineExercise, RoutineSession


class RoutineExerciseInline(admin.TabularInline):
    model = RoutineExercise
    extra = 1


class RoutineSessionInline(admin.TabularInline):
    model = RoutineSession
    extra = 1


@admin.register(Routine)
class RoutineAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "goal",
        "level",
        "intensity",
        "days_per_week",
        "estimated_duration_minutes",
        "is_active",
        "created_at",
    )
    list_filter = ("goal", "level", "intensity", "is_active")
    search_fields = ("name",)
    inlines = (RoutineSessionInline, RoutineExerciseInline)
