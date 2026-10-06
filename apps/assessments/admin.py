from django.contrib import admin

from .models import EvaluationResult, PhysicalEvaluation


class EvaluationResultInline(admin.TabularInline):
    model = EvaluationResult
    extra = 1


@admin.register(PhysicalEvaluation)
class PhysicalEvaluationAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "trainer",
        "evaluation_type",
        "conducted_at",
    )
    list_filter = ("evaluation_type",)
    search_fields = ("user__username", "trainer__username")
    inlines = (EvaluationResultInline,)
