from django.shortcuts import render
from django.views.generic import View
from django.utils.decorators import method_decorator
from django.http import HttpResponse
from api.decorator.response import JsonResponseDecorator
import json
import base64
from api.utilities import detect_drowsiness
import os


# Create your views here.
@method_decorator(JsonResponseDecorator, name='dispatch')
class DataUpdateView(View):
    def post(self, request):
        file = open("testfile.txt", "w")
        # I will receive a JSON of a byte object here
        image = request.POST.get('image_string')
        # print(image)
        file.write(str(image))
        image = base64.b64decode(image)
        x = detect_drowsiness.func(image)
        # print(str(x))
        file.close()
        return x
