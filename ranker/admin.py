from django.contrib import admin
from .models import JobDescription, Resume, ResumeRanking

admin.site.register(JobDescription)
admin.site.register(Resume)
admin.site.register(ResumeRanking)