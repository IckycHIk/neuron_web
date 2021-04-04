from rest_framework import serializers
from .models import CheckText, TrainText, MlModels


# CheckText serializer
class CheckTextSerializer(serializers.ModelSerializer):
    class Meta:
        model = CheckText
        fields = '__all__'


class TrainTextSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainText
        fields = '__all__'


class MlModelsSerializer(serializers.ModelSerializer):
    class Meta:
        model = MlModels
        fields = '__all__'
