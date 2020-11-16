from rest_framework import serializers
from .models import CheckText


# CheckText serializer
class CheckTextSerializer(serializers.ModelSerializer):
    class Meta:
        model = CheckText
        fields = '__all__'





