from django.contrib import admin

from .models import PhysicalData, Profile


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "goal", "experience_level", "birth_date", "updated_at")
    list_filter = ("goal", "experience_level", "gender")
    search_fields = ("user__username", "user__email", "user__first_name", "user__last_name")


@admin.register(PhysicalData)
class PhysicalDataAdmin(admin.ModelAdmin):
    list_display = ("user", "weight_kg", "height_cm", "bmi", "measured_at")
    search_fields = ("user__username", "user__email")
    readonly_fields = ("bmi",)