from .models import CheckText, TrainText,MlModels
from rest_framework import viewsets, permissions
from .serializers import CheckTextSerializer, TrainTextSerializer,MlModelsSerializer
from neuron import views


# CheckText ViewSet
class CheckTextViewSet(viewsets.ModelViewSet):
    queryset = CheckText.objects.all()
    viewsets = views
    permission_classes = [
        permissions.AllowAny
    ]
    serializer_class = CheckTextSerializer


class TrainTextViewSet(viewsets.ModelViewSet):
    queryset = TrainText.objects.all()
    viewsets = views
    permission_classes = [
        permissions.AllowAny
    ]
    serializer_class = TrainTextSerializer


class MlModelsViewSet(viewsets.ModelViewSet):
    queryset = MlModels.objects.all()
    viewsets = views
    permission_classes = [
        permissions.AllowAny
    ]
    serializer_class = MlModelsSerializer
