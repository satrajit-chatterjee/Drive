from django.conf.urls import url
from api.views import DataUpdateView

# namespacing app
app_name = 'api'

urlpatterns = [
    url('data/update/', DataUpdateView.as_view(), name='data-update')
]