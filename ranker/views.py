from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import JobDescription, Resume, ResumeRanking
from .serializers import JobDescriptionSerializer, ResumeSerializer
from .nlp_processor import extract_text, preprocess_text, extract_skills, calculate_scores
from django.shortcuts import render
from django.contrib.auth.models import User

class JobDescriptionAPI(APIView):
    def post(self, request):
        serializer = JobDescriptionSerializer(data=request.data)
        if serializer.is_valid():
            job = serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ResumeAPI(APIView):
    def post(self, request):
        serializer = ResumeSerializer(data=request.data)
        if serializer.is_valid():
            resume = serializer.save(user=request.user)
            
            # Process resume
            raw_text = extract_text(resume.file.path)
            processed_text = preprocess_text(raw_text)
            skills = extract_skills(raw_text)
            
            # Update model
            resume.original_text = raw_text
            resume.processed_text = processed_text
            resume.save()
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RankAPI(APIView):
    def post(self, request):
        try:
            job_desc = request.POST.get('job_description', '')
            resumes = request.FILES.getlist('resumes', [])
            
            if not job_desc or not resumes:
                return Response(
                    {'error': 'Job description and resumes are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Extract skills from job description
            job_skills = extract_skills(job_desc)
            
            results = []
            for resume_file in resumes:
                # Save resume temporarily
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    for chunk in resume_file.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name
                
                try:
                    # Process resume
                    raw_text = extract_text(tmp_path)
                    processed_text = preprocess_text(raw_text)
                    resume_skills = extract_skills(raw_text)
                    experience = extract_experience(raw_text)
                    
                    # Calculate scores
                    scores = calculate_scores(
                        job_text=job_desc,
                        resume_text=processed_text,
                        job_skills=job_skills,
                        resume_skills=resume_skills
                    )
                    
                    # Weighted overall score
                    overall_score = (
                        0.5 * scores['keyword_score'] + 
                        0.3 * scores['skills_score'] + 
                        0.2 * scores['experience_score']
                    ) * 100
                    
                    results.append({
                        'filename': resume_file.name,
                        'score': overall_score,
                        'details': {
                            'keywords': f"{scores['keyword_score']*100:.1f}%",
                            'skills': f"{scores['skills_score']*100:.1f}%",
                            'experience': f"{experience} yrs (score: {scores['experience_score']*100:.1f}%)"
                        }
                    })
                
                finally:
                    os.unlink(tmp_path)  # Clean up temp file
            
            return Response({'results': results})
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
                    
def index_view(request):
    return render(request, 'ranker/index.html')
        