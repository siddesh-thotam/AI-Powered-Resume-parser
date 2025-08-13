from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class JobDescription(models.Model):
    user = models.ForeignKey(User , on_delete=models.CASCADE, null=True , blank=True)
    title = models.CharField(max_length=255)
    description = models.TextField()
    required_skills = models.JSONField(default=list)
    preferred_skills = models.JSONField(default=list)
    required_experience = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Resume(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE , null=True , blank=True)
    file = models.FileField(upload_to='resumes/')
    original_text = models.TextField()
    processed_text = models.TextField(blank=True, null=True)
    skills = models.JSONField(default=list)
    experience = models.JSONField(default=list)
    upload_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Resume {self.id} - {self.user.username}"

class ResumeRanking(models.Model):
    job_description = models.ForeignKey(JobDescription, on_delete=models.CASCADE)
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE)
    keyword_score = models.FloatField()
    skills_score = models.FloatField()
    experience_score = models.FloatField()
    overall_score = models.FloatField()
    suggestions = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('job_description', 'resume')
    
    def __str__(self):
        return f"Ranking {self.id} - Score: {self.overall_score}"
    
