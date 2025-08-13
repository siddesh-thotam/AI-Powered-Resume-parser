from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import JobDescription, Resume, ResumeRanking
from .serializers import JobDescriptionSerializer, ResumeSerializer
from .nlp_processor import extract_text, preprocess_text, extract_skills, calculate_scores, extract_experience
from django.shortcuts import render
from django.contrib.auth.models import User
import tempfile
import os
import logging
import traceback

logger = logging.getLogger(__name__)

class JobDescriptionAPI(APIView):
    def post(self, request):
        try:
            serializer = JobDescriptionSerializer(data=request.data)
            if serializer.is_valid():
                job = serializer.save(user=request.user if request.user.is_authenticated else None)
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"JobDescriptionAPI error: {str(e)}")
            return Response(
                {'error': 'Failed to create job description'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ResumeAPI(APIView):
    def post(self, request):
        try:
            serializer = ResumeSerializer(data=request.data)
            if serializer.is_valid():
                resume = serializer.save(user=request.user if request.user.is_authenticated else None)
                
                try:
                    raw_text = extract_text(resume.file.path)
                    processed_text = preprocess_text(raw_text)
                    skills = extract_skills(raw_text)
                    
                    resume.original_text = raw_text
                    resume.processed_text = processed_text
                    resume.skills = skills
                    resume.save()
                    
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
                except Exception as e:
                    resume.delete()  # Clean up if processing fails
                    raise e
                    
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"ResumeAPI error: {str(e)}")
            return Response(
                {'error': 'Failed to process resume'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RankAPI(APIView):
    def post(self, request):
        try:
            # Debug logging
            logger.info(f"RankAPI request received. POST keys: {request.POST.keys()}, FILES: {request.FILES.keys()}")
            
            job_desc = request.POST.get('job_description', '').strip()
            if not job_desc:
                logger.warning("No job description provided")
                return Response(
                    {'error': 'Job description is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if 'resumes' not in request.FILES:
                logger.warning("No resumes uploaded")
                return Response(
                    {'error': 'Please upload at least one resume file'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            resumes = request.FILES.getlist('resumes')
            if not resumes:
                logger.warning("Empty resumes list")
                return Response(
                    {'error': 'No valid resume files found'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Extract skills from job description
            try:
                job_skills = extract_skills(job_desc)
                logger.info(f"Extracted job skills: {job_skills}")
            except Exception as e:
                logger.error(f"Skill extraction failed: {str(e)}")
                job_skills = []
            
            results = []
            for resume_file in resumes:
                result = {
                    'filename': resume_file.name,
                    'score': 0,
                    'error': None,
                    'details': {}
                }
                
                try:
                    # Validate file type
                    if not resume_file.name.lower().endswith(('.pdf', '.docx', '.txt')):
                        raise ValueError("Only PDF, DOCX, and TXT files are supported")
                    
                    # Save to temp file with proper extension
                    ext = os.path.splitext(resume_file.name)[1]
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        for chunk in resume_file.chunks():
                            tmp.write(chunk)
                        tmp_path = tmp.name
                    
                    try:
                        logger.info(f"Processing file: {resume_file.name}")
                        
                        # Text extraction
                        raw_text = extract_text(tmp_path)
                        if not raw_text:
                            raise ValueError("Could not extract text from file")
                        
                        # NLP processing
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
                        
                        # Calculate overall score
                        overall_score = (
                            0.5 * scores['keyword_score'] + 
                            0.3 * scores['skills_score'] + 
                            0.2 * scores['experience_score']
                        ) * 100
                        
                        # Build result
                        result.update({
                            'score': round(overall_score, 1),
                            'details': {
                                'keywords': f"{scores['keyword_score']*100:.1f}%",
                                'skills': f"{scores['skills_score']*100:.1f}%",
                                'experience': f"{experience} yrs (score: {scores['experience_score']*100:.1f}%)",
                                'matched_skills': list(set(job_skills) & set(resume_skills))
                            }
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {resume_file.name}: {str(e)}")
                        result['error'] = str(e)
                        raise e
                        
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                
                except Exception as e:
                    result['error'] = str(e)
                    logger.error(f"Failed to process {resume_file.name}: {traceback.format_exc()}")
                
                results.append(result)
            
            return Response({
                'success': True,
                'results': results,
                'job_skills': job_skills
            })
            
        except Exception as e:
            logger.error(f"RankAPI failed: {traceback.format_exc()}")
            return Response(
                {
                    'error': 'Internal server error',
                    'details': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

def index_view(request):
    return render(request, 'ranker/index.html')