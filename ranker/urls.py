from django.urls import path
from .views import JobDescriptionAPI, ResumeAPI, RankAPI, index_view

urlpatterns = [
    path('', index_view, name='index'),
    path('jobs/', JobDescriptionAPI.as_view(), name='job-api'),
    path('resumes/', ResumeAPI.as_view(), name='resume-api'),
    path('rank/', RankAPI.as_view(), name='rank-api'),
]