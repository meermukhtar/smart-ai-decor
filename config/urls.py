"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

import os
from django.contrib import admin
from django.urls import path, re_path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

UI_DIR = os.path.join(settings.BASE_DIR, "ui")

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/rooms/", include("rooms.urls")),
    # Direct access to SmartSpace AI UI
    path("", lambda req: serve(req, "index.html", document_root=UI_DIR)),
    path("ui/", lambda req: serve(req, "index.html", document_root=UI_DIR)),
    re_path(r"^ui/(?P<path>.*)$", serve, {"document_root": UI_DIR}),
    re_path(r"^(?P<path>(style\.css|app\.js))$", serve, {"document_root": UI_DIR}),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
