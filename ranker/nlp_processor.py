import spacy
import pdfminer.high_level
import docx 
import numpy as np
import re
import logging
import requests
import json
import os 
from django.conf import settings
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from spacy.matcher import Matcher
logger = logging.getLogger(__name__)

 
# Load medium/large model for better word vectors
# Replace the spaCy loading section with this:
try:
    # Try to load the medium model
    nlp = spacy.load("en_core_web_md")
except OSError:
    try:
        # Try to load the small model if medium is not available
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Download and load the small model if not installed
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    # Fallback to small model or basic processing
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        # Ultimate fallback - create a minimal nlp object
        nlp = None


class SkillNormalizer:
    def __init__(self):
        self.skill_map = self._create_skill_mapping()
        self.word_forms = self._create_word_forms()
        self.stop_phrases = {
            'qualifications', 'requirements', 'skills', 'experience', 'education',
            'years', 'plus', 'strong', 'excellent', 'good', 'knowledge', 'ability'
        }
        
    def _create_skill_mapping(self):
        """Create mapping for common skill variations"""
        return {
            # Programming languages
            'js': 'javascript', 'javascript es6': 'javascript', 'es6': 'javascript',
            'typescript': 'typescript', 'ts': 'typescript', 'python': 'python',
            'python3': 'python', 'py': 'python', 'java': 'java', 'c++': 'c++',
            'cpp': 'c++', 'c#': 'c#', 'csharp': 'c#',
            
            # Frameworks
            'django': 'django', 'flask': 'flask', 'spring': 'spring framework',
            'spring boot': 'spring framework', 'react': 'react', 'reactjs': 'react',
            'react.js': 'react', 'angular': 'angular', 'angularjs': 'angular',
            'vue': 'vue', 'vuejs': 'vue', 'vue.js': 'vue', 'node': 'node.js',
            'nodejs': 'node.js', 'express': 'express.js',
            
            # Databases
            'postgres': 'postgresql', 'postgresql': 'postgresql', 'postgres db': 'postgresql',
            'mysql': 'mysql', 'mongo': 'mongodb', 'mongodb': 'mongodb', 'sql': 'sql',
            'nosql': 'nosql', 'redis': 'redis',
            
            # Cloud
            'aws': 'amazon web services', 'amazon web services': 'amazon web services',
            'azure': 'microsoft azure', 'microsoft azure': 'microsoft azure',
            'gcp': 'google cloud platform', 'google cloud': 'google cloud platform',
            'google cloud platform': 'google cloud platform',
            
            # DevOps
            'docker': 'docker', 'kubernetes': 'kubernetes', 'k8s': 'kubernetes',
            'terraform': 'terraform', 'ansible': 'ansible', 'jenkins': 'jenkins',
            'git': 'git', 'github': 'git', 'gitlab': 'git', 'ci/cd': 'ci/cd',
            'continuous integration': 'ci/cd', 'continuous deployment': 'ci/cd',
            
            # Concepts
            'oop': 'object oriented programming', 'object oriented programming': 'object oriented programming',
            'object-oriented programming': 'object oriented programming',
            'rest': 'rest api', 'restful': 'rest api', 'rest api': 'rest api',
            'graphql': 'graphql', 'api': 'api development',
            
            # Other
            'machine learning': 'machine learning', 'ml': 'machine learning',
            'deep learning': 'deep learning', 'dl': 'deep learning',
            'ai': 'artificial intelligence', 'artificial intelligence': 'artificial intelligence',
            'data science': 'data science',
        }
            
    def _create_word_forms(self):
        """Create mapping for common word forms (singular/plural, etc.)"""
        return {
            'tests': 'testing',
            'test': 'testing',
            'testing': 'testing',
            'developer': 'development',
            'developers': 'development',
            'development': 'development',
            'engineer': 'engineering',
            'engineers': 'engineering',
            'engineering': 'engineering',
            'analyst': 'analysis',
            'analysts': 'analysis',
            'analysis': 'analysis',
        }
    
    def normalize_skill(self, skill):
        """Enhanced skill normalization"""
        if not skill:
            return None
            
        # Convert to lowercase and clean
        skill = skill.lower().strip()
        skill = re.sub(r'[^a-z0-9+#\.\s]', '', skill)
        skill = re.sub(r'\s+', ' ', skill).strip()
        
        # Remove common stop phrases
        words = skill.split()
        filtered_words = [word for word in words if word not in self.stop_phrases and len(word) > 2]
        skill = ' '.join(filtered_words)
        
        if not skill:
            return None
            
        # Check direct mapping first
        if skill in self.skill_map:
            return self.skill_map[skill]
            
        # Check for partial matches in mapping
        for key, value in self.skill_map.items():
            if key in skill:
                return value
                
        # Handle version numbers and common suffixes
        skill = re.sub(r'\s*\d+(\.\d+)*\s*', ' ', skill)
        skill = re.sub(r'\s+(framework|library|tool|technology|language|development|programming)$', '', skill)
        skill = re.sub(r'^(knowledge of|experience with|proficient in|expertise in)\s+', '', skill)
        
        return skill.strip() or None
    
# Initialize the normalizer at module level
skill_normalizer = SkillNormalizer()

def extract_text(file_path):
    """Improved text extraction with error handling"""
    try:
        if file_path.endswith('.pdf'):
            return pdfminer.high_level.extract_text(file_path)
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def preprocess_text(text):
    """More sophisticated text processing with fallback"""
    if nlp is None:
        # Basic fallback processing without spaCy
        text = text.lower()
        # Simple tokenization and stopword removal
        tokens = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return " ".join(tokens)
    
    doc = nlp(text.lower())
    # Include only meaningful tokens
    tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop 
        and not token.is_punct 
        and not token.is_space
        and len(token.lemma_) > 2
    ]
    return " ".join(tokens)

def extract_skills_with_weights(text):
    """Enhanced skill extraction with better pattern matching"""
    doc = nlp(text.lower())
    skills = defaultdict(int)
    
    # Enhanced skill patterns - more comprehensive
    patterns = [
        # Programming languages
        [{"LOWER": {"IN": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", 
                          "php", "go", "rust", "swift", "kotlin", "scala", "r", "matlab"]}}],
        
        # Frameworks and libraries
        [{"LOWER": {"IN": ["django", "flask", "spring", "react", "angular", "vue", "node", 
                          "express", "laravel", "rails", "tensorflow", "pytorch", "scikit", 
                          "pandas", "numpy", "fastapi"]}}],
        
        # Databases
        [{"LOWER": {"IN": ["mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", 
                          "cassandra", "dynamodb", "cosmosdb", "elasticsearch"]}}],
        
        # Cloud and DevOps
        [{"LOWER": {"IN": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", 
                          "ansible", "jenkins", "gitlab", "github", "ci/cd", "devops"]}}],
        
        # Tools and technologies
        [{"LOWER": {"IN": ["git", "svn", "mercurial", "jira", "confluence", "linux", "unix", 
                          "bash", "shell", "rest", "graphql", "soap", "api"]}}],
        
        # Concepts and methodologies
        [{"LOWER": {"IN": ["oop", "agile", "scrum", "kanban", "tdd", "bdd", "microservices", 
                          "serverless", "machine learning", "deep learning", "ai"]}}],
        
        # Specific skill combinations
        [{"LOWER": "object"}, {"LOWER": "oriented"}, {"LOWER": "programming"}],
        [{"LOWER": "restful"}, {"LOWER": "api"}],
        [{"LOWER": "machine"}, {"LOWER": "learning"}],
        [{"LOWER": "deep"}, {"LOWER": "learning"}],
        [{"LOWER": "artificial"}, {"LOWER": "intelligence"}],
    ]
    
    matcher = Matcher(nlp.vocab)
    for pattern in patterns:
        matcher.add("SKILLS", [pattern])
    
    # First pass - pattern matching
    matches = matcher(doc)
    for match_id, start, end in matches:
        skill_text = doc[start:end].text.lower()
        normalized_skill = skill_normalizer.normalize_skill(skill_text)
        if normalized_skill:
            skills[normalized_skill] += 2  # Higher weight for exact matches
    
    # Second pass - noun phrases that contain technical terms
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        # Check if this noun chunk contains known technical terms
        technical_terms = ['python', 'java', 'sql', 'database', 'cloud', 'api', 'web', 
                          'software', 'development', 'programming', 'backend', 'frontend']
        
        if any(term in chunk_text for term in technical_terms):
            normalized_skill = skill_normalizer.normalize_skill(chunk_text)
            if normalized_skill and normalized_skill not in skills:
                skills[normalized_skill] += 1
    
    # Third pass - entities (organizations, technologies)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECH"]:
            normalized_skill = skill_normalizer.normalize_skill(ent.text)
            if normalized_skill:
                skills[normalized_skill] += 1
    
    # Fourth pass - check for emphasized skills (uppercase, bold patterns)
    for token in doc:
        if token.text.isupper() and len(token.text) > 2:
            normalized_skill = skill_normalizer.normalize_skill(token.text)
            if normalized_skill:
                skills[normalized_skill] += 3
    
    # Normalize weights to 1-5 scale
    if skills:
        max_weight = max(skills.values()) if skills.values() else 1
        return {skill: min(5, max(1, 1 + 4 * (weight/max_weight))) for skill, weight in skills.items()}
    
    return {}


def extract_experience(text):
    """Extract experience in years"""
    doc = nlp(text)
    experience = 0
    
    # Match experience patterns
    patterns = [
        r"(\d+)\s*(years?|yrs?)\s*(of)?\s*(experience)?",
        r"experienced.*?(\d+)\s*(years?|yrs?)",
        r"(\d+)\+?\s*(years?|yrs?)\s*(in|of)"
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                years = int(match.group(1))
                experience = max(experience, years)
            except (ValueError, IndexError):
                continue
                
    return experience

def categorize_skill(skill):
    """Categorize skills for better feedback"""
    tech_skills = {'python', 'java', 'javascript', 'c++', 'html', 'css', 
                  'react', 'angular', 'node', 'docker', 'kubernetes', 'aws'}
    soft_skills = {'communication', 'leadership', 'teamwork', 'problem-solving'}
    
    if skill in tech_skills:
        return 'Technical Skill'
    elif skill in soft_skills:
        return 'Soft Skill'
    elif 'aws' in skill or 'azure' in skill or 'cloud' in skill:
        return 'Cloud Technology'
    elif 'sql' in skill or 'database' in skill:
        return 'Database'
    else:
        return 'Other'

def calculate_scores(job_text, resume_text, job_skills, resume_skills):
    """Enhanced scoring algorithm with weighted matching and strength highlighting"""
    # Initialize empty results
    default_result = {
        'keyword_score': 0,
        'skills_score': 0,
        'experience_score': 0,
        'missing_skills': [],
        'matched_skills': [],
        'weighted_skills': {},
        'gap_analysis': [],
        'strengths': []
    }

    if not job_text or not resume_text:
        return default_result

    try:
        # Extract and normalize skills from both documents
        job_skill_weights = extract_skills_with_weights(job_text) or {}
        resume_skill_weights = extract_skills_with_weights(resume_text) or {}
        
        # Normalize all skills
        normalized_job_skills = {}
        for skill, weight in job_skill_weights.items():
            norm_skill = skill_normalizer.normalize_skill(skill)
            if norm_skill:
                normalized_job_skills[norm_skill] = weight

        normalized_resume_skills = {}
        for skill, weight in resume_skill_weights.items():
            norm_skill = skill_normalizer.normalize_skill(skill)
            if norm_skill:
                normalized_resume_skills[norm_skill] = weight


        # Define is_valid_skill function BEFORE using it
        def is_valid_skill(skill):
            if not skill or len(skill) < 3:
                return False
            # Filter out generic terms
            generic_terms = {'skill', 'ability', 'experience', 'knowledge', 'development', 
                            'programming', 'strong', 'excellent', 'good'}
            words = skill.split()
            return not any(term in words for term in generic_terms)
        
        # Apply filtering to remove invalid skills
        normalized_job_skills = {k: v for k, v in normalized_job_skills.items() if is_valid_skill(k)}
        normalized_resume_skills = {k: v for k, v in normalized_resume_skills.items() if is_valid_skill(k)}

        # Get all unique normalized skill names
        all_job_skills = set(normalized_job_skills.keys())
        all_resume_skills = set(normalized_resume_skills.keys())

        # Keyword matching with TF-IDF
        keyword_score = 0
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            tfidf = vectorizer.fit_transform([job_text, resume_text])
            keyword_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except Exception:
            keyword_score = 0

        # Skills matching - compare normalized skills
        matched_skills = all_job_skills & all_resume_skills
        missing_skills = all_job_skills - all_resume_skills
        
        # Calculate skills score based on weights
        total_weight = sum(normalized_job_skills.values())
        matched_weight = sum(normalized_job_skills.get(skill, 0) for skill in matched_skills)
        skills_score = matched_weight / total_weight if total_weight > 0 else 0

        # Experience matching
        resume_exp = extract_experience(resume_text)
        job_exp = extract_experience(job_text) or 3  # Default to 3 years
        exp_score = min(1, resume_exp / job_exp) if job_exp > 0 else 0

        # Generate gap analysis
        gap_analysis = []
        for skill in missing_skills:
            gap_analysis.append({
                'skill': skill,
                'importance': normalized_job_skills.get(skill, 1),
                'category': categorize_skill(skill)
            })

        # Strength analysis
        strength_analysis = []
        for skill, weight in normalized_resume_skills.items():
            if skill in normalized_job_skills or is_generally_valuable_skill(skill):
                strength_analysis.append({
                    'skill': skill,
                    'strength_level': weight,
                    'category': categorize_skill(skill),
                    'relevance': 'job-specific' if skill in normalized_job_skills else 'general'
                })

        suggestions = []
        for skill in missing_skills:
            suggestions.append({
                'skill': skill,
                'suggestion': generate_skill_suggestion(skill),
                'importance': normalized_job_skills.get(skill, 1),
                'category': categorize_skill(skill)
                })

        # Sort results
        gap_analysis.sort(key=lambda x: x['importance'], reverse=True)
        strength_analysis.sort(key=lambda x: x['strength_level'], reverse=True)

        return {
            'keyword_score': min(1, max(0, keyword_score)),
            'skills_score': min(1, max(0, skills_score)),
            'experience_score': min(1, max(0, exp_score)),
            'missing_skills': list(missing_skills),
            'matched_skills': list(matched_skills),
            'weighted_skills': normalized_job_skills,
            'gap_analysis': suggestions,  # This now includes suggestions
            'strengths': strength_analysis[:10],
        }

    except Exception as e:
        logger.error(f"Error in calculate_scores: {str(e)}")
        return default_result



def is_generally_valuable_skill(skill):
    """Determine if a skill is generally valuable even if not in job description"""
    valuable_skills = {
        # Technical skills
        'python', 'javascript', 'java', 'c++', 'c#', 'sql', 
        'aws', 'azure', 'docker', 'kubernetes', 'react', 'angular',
        # Soft skills
        'communication', 'leadership', 'teamwork', 'problem solving',
        'critical thinking', 'time management', 'adaptability'
    }
    
    return skill in valuable_skills    
def generate_skill_suggestion(skill):
    """Generate more specific and varied skill suggestions"""
    suggestions = {
        'python': [
            'Build a portfolio project using Python with Django or Flask framework',
            'Complete Python certification from Python Institute or Coursera',
            'Contribute to open-source Python projects on GitHub'
        ],
        'django': [
            'Create a full-stack application with Django REST framework and React',
            'Take Django for Beginners or Django for APIs courses',
            'Learn about Django ORM optimization and database management'
        ],
        'flask': [
            'Build microservices with Flask and containerize with Docker',
            'Learn Flask authentication and security best practices',
            'Create RESTful APIs with Flask-RESTful extension'
        ],
        'aws': [
            'Get AWS Cloud Practitioner certification as foundation',
            'Practice with AWS Free Tier for hands-on experience',
            'Build and deploy applications using EC2, S3, and Lambda'
        ],
        'docker': [
            'Containerize existing applications and deploy them',
            'Learn Docker Compose for multi-container applications',
            'Study Kubernetes for container orchestration (CKA certification)'
        ],
        'kubernetes': [
            'Complete Kubernetes the Hard Way tutorial for deep understanding',
            'Get Certified Kubernetes Administrator (CKA) certification',
            'Practice with minikube or kind for local development'
        ],
        'postgresql': [
            'Learn advanced PostgreSQL features like window functions and CTEs',
            'Practice database optimization and indexing strategies',
            'Study PostgreSQL administration and backup procedures'
        ],
        'mysql': [
            'Learn MySQL performance tuning and query optimization',
            'Practice database design and normalization principles',
            'Study MySQL replication and high availability setups'
        ],
        'rest api': [
            'Build RESTful APIs with proper status codes and error handling',
            'Learn OpenAPI/Swagger for API documentation',
            'Practice API security with JWT and OAuth2'
        ],
        'git': [
            'Master Git branching strategies like Git Flow',
            'Learn advanced Git commands for complex scenarios',
            'Practice collaborative workflows with pull requests'
        ],
        'agile': [
            'Get Certified Scrum Master (CSM) or similar certification',
            'Participate in Agile workshops or simulations',
            'Learn about different Agile frameworks (Scrum, Kanban, XP)'
        ],
        'machine learning': [
            'Complete machine learning courses on Coursera or edX',
            'Build ML projects with scikit-learn and TensorFlow',
            'Participate in Kaggle competitions for practical experience'
        ],
        'react': [
            'Build interactive UIs with React hooks and context API',
            'Learn state management with Redux or Zustand',
            'Practice component testing with Jest and React Testing Library'
        ],
        'angular': [
            'Learn Angular framework with TypeScript fundamentals',
            'Build enterprise applications with Angular CLI',
            'Study Angular services, dependency injection, and RxJS'
        ]
    }
    
    # Default suggestions categorized by skill type
    default_suggestions = {
        'technical': [
            'Take online courses on platforms like Coursera, Udemy, or edX',
            'Build practical projects to demonstrate proficiency',
            'Obtain relevant certifications from recognized institutions'
        ],
        'soft': [
            'Participate in team projects and collaborative activities',
            'Take communication and leadership workshops',
            'Practice through role-playing and real-world scenarios'
        ],
        'cloud': [
            'Use cloud provider free tiers for hands-on practice',
            'Get cloud certifications (AWS, Azure, GCP)',
            'Build and deploy applications on cloud platforms'
        ],
        'database': [
            'Practice database design and optimization techniques',
            'Learn SQL advanced features and performance tuning',
            'Study database administration and management'
        ]
    }
    
    # Return specific suggestions if available, otherwise default based on category
    if skill in suggestions:
        return random.choice(suggestions[skill])
    else:
        category = categorize_skill(skill)
        if category == 'Technical Skill':
            return random.choice(default_suggestions['technical'])
        elif category == 'Soft Skill':
            return random.choice(default_suggestions['soft'])
        elif category == 'Cloud Technology':
            return random.choice(default_suggestions['cloud'])
        elif category == 'Database':
            return random.choice(default_suggestions['database'])
        else:
            return random.choice(default_suggestions['technical'])

def summarize_job_description(job_text):
    """Enhanced job description summarization"""
    try:
        # First try to use the Hugging Face API if available
        api_key = os.environ.get('HUGGINGFACE_API_KEY') or getattr(settings, 'HUGGINGFACE_API_KEY', None)
        
        if api_key:
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            payload = {
                "inputs": job_text,
                "parameters": {
                    "max_length": 200,
                    "min_length": 100,
                    "do_sample": False
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    summary = result[0].get('summary_text', '')
                    if summary:
                        return summary
        
        # If API fails or not available, use our enhanced fallback
        return generate_enhanced_summary(job_text)
            
    except Exception as e:
        logger.error(f"JD summarization failed: {str(e)}")
        return generate_enhanced_summary(job_text)
    
def generate_enhanced_summary(job_text):
    """Enhanced summary generation with better extraction"""
    # Extract key components
    summary_parts = []
    
    # Extract experience requirement
    experience_match = re.search(r'(\d+)[\s\-–]+(\d+)?\s*years?', job_text.lower())
    if experience_match:
        years = experience_match.group(1)
        if experience_match.group(2):
            years = f"{years}-{experience_match.group(2)}"
        summary_parts.append(f"{years}+ years of experience")
    
    # Extract seniority level
    level_keywords = ['senior', 'junior', 'mid-level', 'lead', 'principal', 'entry-level']
    for level in level_keywords:
        if level in job_text.lower():
            summary_parts.append(f"{level.title()} level position")
            break
    
    # Extract key technologies with better pattern matching
    tech_keywords = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 
        'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'django', 'flask', 'fastapi',
        'spring', 'react', 'angular', 'vue', 'node', 'express', 'tensorflow', 'pytorch',
        'scikit-learn', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'postgresql',
        'mysql', 'mongodb', 'sql', 'git', 'html', 'css', 'rest', 'api', 'ml', 'ai'
    }
    
    found_skills = []
    for skill in tech_keywords:
        if re.search(rf'\b{re.escape(skill)}\b', job_text.lower()):
            found_skills.append(skill)
    
    if found_skills:
        # Group and categorize skills
        languages = [s for s in found_skills if s in ['python', 'java', 'javascript', 'typescript']]
        frameworks = [s for s in found_skills if s in ['django', 'flask', 'fastapi', 'spring', 'react', 'angular']]
        ml_tools = [s for s in found_skills if s in ['tensorflow', 'pytorch', 'scikit-learn', 'ml', 'ai']]
        cloud = [s for s in found_skills if s in ['aws', 'azure', 'gcp', 'docker', 'kubernetes']]
        databases = [s for s in found_skills if s in ['postgresql', 'mysql', 'mongodb', 'sql']]
        
        if languages:
            summary_parts.append(f"Programming: {', '.join(languages[:3])}")
        if frameworks:
            summary_parts.append(f"Frameworks: {', '.join(frameworks[:3])}")
        if ml_tools:
            summary_parts.append(f"ML/AI: {', '.join(ml_tools[:3])}")
        if cloud:
            summary_parts.append(f"Cloud: {', '.join(cloud[:2])}")
        if databases:
            summary_parts.append(f"Databases: {', '.join(databases[:2])}")
    
    # Extract role type
    role_patterns = [
        r'(python developer|backend developer|software engineer|ml engineer|ai engineer)',
        r'(develop|build|create|design).*?(application|system|software|api|service)'
    ]
    
    for pattern in role_patterns:
        match = re.search(pattern, job_text.lower())
        if match:
            role = match.group(1) if match.lastindex else match.group(0)
            summary_parts.append(f"Role: {role.title()}")
            break
    
    # Extract location if mentioned
    location_match = re.search(r'location[:\s]+([^\n]+)', job_text.lower())
    if location_match:
        summary_parts.append(f"Location: {location_match.group(1).title()}")
    
    return " • ".join(summary_parts) if summary_parts else "Comprehensive job description analysis available"