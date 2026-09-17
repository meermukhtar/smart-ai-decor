from django.urls import path
from .views import RoomUploadView, AutoDecorateRoomView

urlpatterns = [
    path(
        "upload/",
        RoomUploadView.as_view(),
        name="room-upload"
    ),
    path(
        "auto-decorate/",
        AutoDecorateRoomView.as_view(),
        name="room-auto-decorate"
    ),
]