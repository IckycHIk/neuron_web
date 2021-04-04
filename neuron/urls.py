from rest_framework import routers
from .api import CheckTextViewSet
from django.urls import path
from neuron import views


from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
   path('api/', views.snippet_list),
   path('', views.index),
   path('train/',views.data_train),
   path('restart/',views.re_lern_model)
]
urlpatterns = format_suffix_patterns(urlpatterns)