from .models import CheckText
from rest_framework import viewsets, permissions
from .serializers import CheckTextSerializer


# CheckText ViewSet
class CheckTextViewSet(viewsets.ModelViewSet):
    queryset = CheckText.objects.all()
    permission_classes = [
        permissions.AllowAny
    ]
    serializer_class = CheckTextSerializer


