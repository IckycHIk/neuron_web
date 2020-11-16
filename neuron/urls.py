from rest_framework import routers
from .api import CheckTextViewSet

router = routers.DefaultRouter()
router.register('api/neuron', CheckTextViewSet, 'neuron')


urlpatterns = router.urls
