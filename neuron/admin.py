
from django.contrib import admin
from .models import CheckText, TrainText, MlModels


# Register your models here.


class CheckTextAdmin(admin.ModelAdmin):
    list_display = ('id', 'text', 'user_id', 'date', 'done', 'isBulling', 'percent')
    list_display_links = ('id', 'text')
    search_fields = ('id', 'text')
    list_editable = ('done',)


class TrainTextAdmin(admin.ModelAdmin):
    list_display = ('id', 'text', 'isBulling')
    list_display_links = ('id', 'text')
    search_fields = ('id', 'text')


class MlModelsAdmin(admin.ModelAdmin):
    list_display = ('id', 'model')


admin.site.register(MlModels, MlModelsAdmin)
admin.site.register(CheckText, CheckTextAdmin)
admin.site.register(TrainText, TrainTextAdmin)
