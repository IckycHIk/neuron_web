from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator


# Create your models here.


class CheckText(models.Model):
    text = models.CharField(max_length=400)
    user_id = models.CharField(max_length=400)
    date = models.DateTimeField(auto_now_add=True)
    done = models.BooleanField(default=False, null=False)
    isBulling = models.BooleanField(default=False, null=False)
    percent = models.FloatField(
        default=1,
        validators=[
            MaxValueValidator(1),
            MinValueValidator(0)
        ])


class TrainText(models.Model):
    text = models.CharField(max_length=400)
    isBulling = models.BooleanField(default=True, null=False)


class MlModels(models.Model):
    model = models.BinaryField()
