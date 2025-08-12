from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import JobDescription, Resume, ResumeRanking
from .serializers import JobDescriptionSerializer, ResumeSerializer
from .nlp_processor import extract_text, preprocess_text, extract_skills, calculate_scores
from django.shortcuts import render

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
        print("\n=== Received Data ===")  # Debugging
        print("POST data:", request.POST)
        print("FILES:", request.FILES)
        
        try:
            job_desc = request.data.get('job_description', '')
            resumes = request.FILES.getlist('resumes', [])
            
            if not job_desc or not resumes:
                return Response(
                    {'error': 'Job description and at least one resume are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create temporary job description
            job = JobDescription.objects.create(
                title="Analysis Job",
                description=job_desc,
                required_skills=[],
                preferred_skills=[],
                required_experience=0
            )
            
            results = []
            for resume_file in resumes:
                # Save resume file
                resume = Resume(file=resume_file)
                resume.save()
                
                # Process text
                raw_text = extract_text(resume.file.path)
                processed_text = preprocess_text(raw_text)
                skills = extract_skills(raw_text)
                
                # Calculate scores
                scores = calculate_scores(
                    job_desc,
                    processed_text,
                    [],
                    skills
                )
                
                results.append({
                    'filename': resume_file.name,
                    'score': (scores['keyword_score'] * 0.5 + 
                             scores['skills_score'] * 0.3 + 
                             scores['experience_score'] * 0.2) * 100
                })
            
            return Response({'results': results})
            
        except Exception as e:
            print("Error:", str(e))  # Debugging
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
def index_view(request):
    return render(request, 'ranker/index.html')
        