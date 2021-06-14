from random import random

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from .models import CheckText, TrainText, MlModels
from .serializers import CheckTextSerializer, TrainTextSerializer
from neuron_main import MODELS


# Create your views here.

def index(request):
    element = TrainText.objects.all()
    # TrainText.objects.all().delete()

    context = {
        'title': 'Latest posts',
        'element': element
    }
    return render(request, 'html/index.html', context)


def re_lern_model(request):
    MlModels.objects.all().delete()
    neuron = MODELS
    print("Start Relern Models")
    neuron.learnModelSVC()
    return index(request)

@api_view(['GET', 'POST'])
def data_train(request, format=None):
    if request.method == 'GET':
        snippets = TrainText.objects.all()
        serializer = TrainTextSerializer(snippets, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':

        data = JSONParser().parse(request)
        serializer = TrainTextSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)



@api_view(['GET', 'POST'])
def snippet_list(request, format=None):
    neuron = MODELS

    if request.method == 'GET':
        snippets = CheckText.objects.all()
        serializer = CheckTextSerializer(snippets, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':

        data = JSONParser().parse(request)
        data.update({"percent": neuron.isBulling(data["text"])})
        print(data["percent"])
        print(data)
        if data["percent"]>=0.4:
            data.update({"isBulling": True})
        else:
            data.update({"isBulling": False})



        serializer = CheckTextSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)
