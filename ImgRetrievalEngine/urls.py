from ImgRetrievalEngine  import views
from django.conf.urls import url

urlpatterns = [
    url(r'^upload/',views.uploadImgs),
    url(r'^search/',views.retrievalImgs),
]