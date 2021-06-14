from django.views.generic import TemplateView
from rest_framework import routers
from rest_framework.schemas import get_schema_view

from .api import CheckTextViewSet
from django.urls import path
from neuron import views


from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
   path('api/', views.snippet_list),
   path('', views.index),
   path('train/',views.data_train),
   path('restart/',views.re_lern_model)
   ,   path('swagger-ui/', TemplateView.as_view(
        template_name='html/swagger-ui.html',
        extra_context={'schema_url':'openapi-schema'}
    ), name='swagger-ui'),
    path('redoc/', TemplateView.as_view(
        template_name='html/swagger-ui.html',
        extra_context={'schema_url': 'openapi-schema'}
    ), name='redoc'),
    path('openapi', get_schema_view(
        title="Your Project",
        description="API for all things …",
        version="1.0.0"
    ), name='openapi-schema'),
]
urlpatterns = format_suffix_patterns(urlpatterns)