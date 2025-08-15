from django.urls import path
from .views import JobDescriptionAPI, ResumeAPI, RankAPI, index_view, landing_page

urlpatterns = [
    path('', landing_page, name='landing'),
    path('ranker/', index_view, name='index'),
    
    # Change these to match the URL you're calling from JavaScript
    path('ranker/jobs/', JobDescriptionAPI.as_view(), name='job-api'),
    path('ranker/resumes/', ResumeAPI.as_view(), name='resume-api'),
    path('ranker/rank/', RankAPI.as_view(), name='rank-api'),  # This matches your JS call
]