from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('ranker.urls')),
    path('', include('ranker.urls')),  # Include ranker URLs at root
    path('', TemplateView.as_view(template_name='ranker/landing.html'), name='home'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
