from django.contrib import admin
from .models import CheckText


# Register your models here.


class CheckTextAdmin(admin.ModelAdmin):
    list_display = ('id', 'text', 'user_id', 'date', 'done', 'percent')
    list_display_links = ('id', 'text')
    search_fields = ('id', 'text')
    list_editable = ('done',)


admin.site.register(CheckText, CheckTextAdmin)
