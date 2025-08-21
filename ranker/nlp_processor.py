import re
import logging
import requests
import json
import os
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from django.conf import settings
import pdfminer.high_level
import docx

logger = logging.getLogger(__name__)

# Enhanced NLTK download with better error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.warning(f"NLTK downloads may have failed: {str(e)}")

class SimpleNLP:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK fails
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
                'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
                "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
                'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
                'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
                "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
                "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
                'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
                'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
            }
    
    def process(self, text):
        try:
            tokens = word_tokenize(text.lower())
        except:
            # Fallback tokenization if NLTK fails
            tokens = text.lower().split()
        
        tokens = [
            token for token in tokens 
            if token not in self.stop_words 
            and token.isalnum() 
            and len(token) > 2
        ]
        return tokens

# Initialize the NLP processor
nlp = SimpleNLP()

class SkillNormalizer:
    def __init__(self):
        self.skill_map = self._create_skill_mapping()
        self.stop_phrases = {
            'qualifications', 'requirements', 'skills', 'experience', 'education',
            'years', 'plus', 'strong', 'excellent', 'good', 'knowledge', 'ability'
        }
        
    def _create_skill_mapping(self):
        return {
            'js': 'javascript', 'javascript es6': 'javascript', 'es6': 'javascript',
            'typescript': 'typescript', 'ts': 'typescript', 'python': 'python',
            'python3': 'python', 'py': 'python', 'java': 'java', 'c++': 'c++',
            'cpp': 'c++', 'c#': 'c#', 'csharp': 'c#',
            'django': 'django', 'flask': 'flask', 'spring': 'spring framework',
            'spring boot': 'spring framework', 'react': 'react', 'reactjs': 'react',
            'react.js': 'react', 'angular': 'angular', 'angularjs': 'angular',
            'vue': 'vue', 'vuejs': 'vue', 'vue.js': 'vue', 'node': 'node.js',
            'nodejs': 'node.js', 'express': 'express.js',
            'postgres': 'postgresql', 'postgresql': 'postgresql', 'postgres db': 'postgresql',
            'mysql': 'mysql', 'mongo': 'mongodb', 'mongodb': 'mongodb', 'sql': 'sql',
            'nosql': 'nosql', 'redis': 'redis',
            'aws': 'amazon web services', 'amazon web services': 'amazon web services',
            'azure': 'microsoft azure', 'microsoft azure': 'microsoft azure',
            'gcp': 'google cloud platform', 'google cloud': 'google cloud platform',
            'google cloud platform': 'google cloud platform',
            'docker': 'docker', 'kubernetes': 'kubernetes', 'k8s': 'kubernetes',
            'terraform': 'terraform', 'ansible': 'ansible', 'jenkins': 'jenkins',
            'git': 'git', 'github': 'git', 'gitlab': 'git', 'ci/cd': 'ci/cd',
            'continuous integration': 'ci/cd', 'continuous deployment': 'ci/cd',
            'oop': 'object oriented programming', 'object oriented programming': 'object oriented programming',
            'object-oriented programming': 'object oriented programming',
            'rest': 'rest api', 'restful': 'rest api', 'rest api': 'rest api',
            'graphql': 'graphql', 'api': 'api development',
            'machine learning': 'machine learning', 'ml': 'machine learning',
            'deep learning': 'deep learning', 'dl': 'deep learning',
            'ai': 'artificial intelligence', 'artificial intelligence': 'artificial intelligence',
            'data science': 'data science',
        }
    
    def normalize_skill(self, skill):
        if not skill:
            return None
            
        skill = skill.lower().strip()
        skill = re.sub(r'[^a-z0-9+#\.\s-]', '', skill)  # Allow hyphens
        skill = re.sub(r'\s+', ' ', skill).strip()
        
        words = skill.split()
        filtered_words = [word for word in words if word not in self.stop_phrases and len(word) > 2]
        skill = ' '.join(filtered_words)
        
        if not skill:
            return None
            
        if skill in self.skill_map:
            return self.skill_map[skill]
            
        for key, value in self.skill_map.items():
            if key in skill:
                return value
                
        skill = re.sub(r'\s*\d+(\.\d+)*\s*', ' ', skill)
        skill = re.sub(r'\s+(framework|library|tool|technology|language|development|programming)$', '', skill)
        skill = re.sub(r'^(knowledge of|experience with|proficient in|expertise in)\s+', '', skill)
        
        return skill.strip() or None

skill_normalizer = SkillNormalizer()

def extract_text(file_path):
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
        logger.error(f"Error extracting text: {str(e)}")
        return ""

def preprocess_text(text):
    text = text.lower()
    tokens = nlp.process(text)
    return " ".join(tokens)

def extract_skills_with_weights(text):
    text_lower = text.lower()
    skills = defaultdict(int)
    
    # Enhanced skill patterns with better coverage
    skill_patterns = {
        'python': r'\bpython\b',
        'java': r'\bjava\b',
        'javascript': r'\bjavascript\b|\bjs\b',
        'typescript': r'\btypescript\b|\bts\b',
        'c++': r'\bc\+\+\b',
        'c#': r'\bc#\b|\bcsharp\b',
        'django': r'\bdjango\b',
        'flask': r'\bflask\b',
        'react': r'\breact\b|\breactjs\b|\breact\.js\b',
        'angular': r'\bangular\b|\bangularjs\b',
        'vue': r'\bvue\b|\bvuejs\b|\bvue\.js\b',
        'node.js': r'\bnode\.js\b|\bnodejs\b',
        'express.js': r'\bexpress\.js\b|\bexpressjs\b',
        'aws': r'\baws\b|\bamazon web services\b',
        'azure': r'\bazure\b|\bmicrosoft azure\b',
        'gcp': r'\bgcp\b|\bgoogle cloud\b|\bgoogle cloud platform\b',
        'docker': r'\bdocker\b',
        'kubernetes': r'\bkubernetes\b|\bk8s\b',
        'postgresql': r'\bpostgresql\b|\bpostgres\b',
        'mysql': r'\bmysql\b',
        'mongodb': r'\bmongodb\b|\bmongo\b',
        'sql': r'\bsql\b',
        'git': r'\bgit\b',
        'machine learning': r'\bmachine learning\b|\bml\b',
        'artificial intelligence': r'\bartificial intelligence\b|\bai\b',
        'data science': r'\bdata science\b',
        'rest api': r'\brest api\b|\brestful api\b|\brest\b',
        'graphql': r'\bgraphql\b',
        'html': r'\bhtml\b',
        'css': r'\bcss\b',
        'bootstrap': r'\bbootstrap\b',
        'tailwind': r'\btailwind\b',
        'jenkins': r'\bjenkins\b',
        'terraform': r'\bterraform\b',
        'ansible': r'\bansible\b',
        'tableau': r'\btableau\b',
        'power bi': r'\bpower bi\b|\bpowerbi\b',
        'excel': r'\bexcel\b',
        'sap': r'\bsap\b',
        'hana': r'\bhana\b',
        'etl': r'\betl\b',
        'data warehousing': r'\bdata warehousing\b',
        'data modeling': r'\bdata modeling\b|\bdata modelling\b'
    }
    
    # Find skills using regex patterns - PRIMARY METHOD
    for skill_name, pattern in skill_patterns.items():
        try:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                count = len(matches)
                skills[skill_name] += min(5, 1 + count * 0.5)
        except Exception as e:
            logger.warning(f"Error processing pattern for {skill_name}: {e}")
    
    # SECONDARY METHOD: Only look for specific technical contexts
    try:
        # Look for skills mentioned in specific contexts (e.g., after "experience with", "knowledge of")
        context_patterns = [
            r'(experience with|knowledge of|proficient in|expertise in|skills in)\s+([a-zA-Z+#\s]+)',
            r'(python|java|javascript|sql|aws|azure|docker|kubernetes|react|angular|vue|node)',
        ]
        
        for pattern in context_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match.lastindex >= 2:
                    potential_skill = match.group(2).strip()
                else:
                    potential_skill = match.group(1).strip()
                
                normalized = skill_normalizer.normalize_skill(potential_skill)
                if normalized and normalized in skill_patterns:
                    skills[normalized] += 1
                    
    except Exception as e:
        logger.warning(f"Error in context-based skill extraction: {e}")
    
    return dict(skills)

def extract_experience(text):
    patterns = [
        r"(\d+)\s*(years?|yrs?)\s*(of)?\s*(experience)?",
        r"experienced.*?(\d+)\s*(years?|yrs?)",
        r"(\d+)\+?\s*(years?|yrs?)\s*(in|of)",
        r"(\d+)[\s\-–]+(\d+)?\s*years?"
    ]
    
    experience = 0
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                years = int(match.group(1))
                experience = max(experience, years)
            except (ValueError, IndexError):
                continue
                
    return experience

def categorize_skill(skill):
    tech_skills = {'python', 'java', 'javascript', 'c++', 'html', 'css', 
                  'react', 'angular', 'node', 'docker', 'kubernetes', 'aws',
                  'typescript', 'c#', 'ruby', 'php', 'go', 'rust', 'swift'}
    soft_skills = {'communication', 'leadership', 'teamwork', 'problem-solving',
                  'problem solving', 'critical thinking', 'time management'}
    
    skill_lower = skill.lower()
    
    if any(tech in skill_lower for tech in tech_skills):
        return 'Technical Skill'
    elif any(soft in skill_lower for soft in soft_skills):
        return 'Soft Skill'
    elif 'aws' in skill_lower or 'azure' in skill_lower or 'cloud' in skill_lower or 'gcp' in skill_lower:
        return 'Cloud Technology'
    elif 'sql' in skill_lower or 'database' in skill_lower or 'mysql' in skill_lower or 'postgres' in skill_lower:
        return 'Database'
    else:
        return 'Other'

def is_generally_valuable_skill(skill):
    """Determine if a skill is generally valuable even if not in job description"""
    valuable_skills = {
        # Technical skills
        'python', 'javascript', 'java', 'c++', 'c#', 'sql', 
        'aws', 'azure', 'docker', 'kubernetes', 'react', 'angular',
        'typescript', 'node.js', 'express.js', 'django', 'flask',
        # Soft skills
        'communication', 'leadership', 'teamwork', 'problem solving',
        'critical thinking', 'time management', 'adaptability'
    }
    
    return skill in valuable_skills

def generate_skill_suggestion(skill):
    """Generate specific skill suggestions"""
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
        # ... (keep your existing suggestions)
    }
    
    # Default suggestions
    default_suggestions = {
        'Technical Skill': [
            'Take online courses on platforms like Coursera, Udemy, or edX',
            'Build practical projects to demonstrate proficiency',
            'Obtain relevant certifications from recognized institutions'
        ],
        'Soft Skill': [
            'Participate in team projects and collaborative activities',
            'Take communication and leadership workshops',
            'Practice through role-playing and real-world scenarios'
        ],
        'Cloud Technology': [
            'Use cloud provider free tiers for hands-on practice',
            'Get cloud certifications (AWS, Azure, GCP)',
            'Build and deploy applications on cloud platforms'
        ],
        'Database': [
            'Practice database design and optimization techniques',
            'Learn SQL advanced features and performance tuning',
            'Study database administration and management'
        ],
        'Other': [
            'Research the skill and its applications in your industry',
            'Find online tutorials and practice exercises',
            'Connect with professionals who have this skill'
        ]
    }
    
    # Return specific suggestions if available, otherwise default based on category
    if skill in suggestions:
        return random.choice(suggestions[skill])
    else:
        category = categorize_skill(skill)
        return random.choice(default_suggestions.get(category, default_suggestions['Other']))

def calculate_scores(job_text, resume_text, job_skills, resume_skills):
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
        # Extract skills
        job_skill_weights = extract_skills_with_weights(job_text) or {}
        resume_skill_weights = extract_skills_with_weights(resume_text) or {}
        
        # Debug: Log extracted skills
        logger.info(f"Job skills: {job_skill_weights}")
        logger.info(f"Resume skills: {resume_skill_weights}")
        
        # Normalize skills
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

        def is_valid_skill(skill):
            if not skill or len(skill) < 3:
                return False
            generic_terms = {'skill', 'ability', 'experience', 'knowledge', 'development', 
                            'programming', 'strong', 'excellent', 'good'}
            words = skill.split()
            return not any(term in words for term in generic_terms)
        
        normalized_job_skills = {k: v for k, v in normalized_job_skills.items() if is_valid_skill(k)}
        normalized_resume_skills = {k: v for k, v in normalized_resume_skills.items() if is_valid_skill(k)}

        all_job_skills = set(normalized_job_skills.keys())
        all_resume_skills = set(normalized_resume_skills.keys())

        # Keyword matching with TF-IDF
        keyword_score = 0
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            tfidf = vectorizer.fit_transform([job_text, resume_text])
            keyword_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}")
            keyword_score = 0

        # Skills matching
        matched_skills = all_job_skills & all_resume_skills
        missing_skills = all_job_skills - all_resume_skills
        
        total_weight = sum(normalized_job_skills.values()) if normalized_job_skills else 1
        matched_weight = sum(normalized_job_skills.get(skill, 0) for skill in matched_skills)
        skills_score = matched_weight / total_weight if total_weight > 0 else 0

        # Experience matching
        resume_exp = extract_experience(resume_text)
        job_exp = extract_experience(job_text) or 3
        exp_score = min(1, resume_exp / job_exp) if job_exp > 0 else 0

        # Generate gap analysis with suggestions
        gap_analysis = []
        for skill in missing_skills:
            suggestion = generate_skill_suggestion(skill)
            gap_analysis.append({
                'skill': skill,
                'suggestion': suggestion,
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

        gap_analysis.sort(key=lambda x: x['importance'], reverse=True)
        strength_analysis.sort(key=lambda x: x['strength_level'], reverse=True)

        return {
            'keyword_score': min(1, max(0, keyword_score)),
            'skills_score': min(1, max(0, skills_score)),
            'experience_score': min(1, max(0, exp_score)),
            'missing_skills': list(missing_skills),
            'matched_skills': list(matched_skills),
            'weighted_skills': normalized_job_skills,
            'gap_analysis': gap_analysis,
            'strengths': strength_analysis[:10],
        }

    except Exception as e:
        logger.error(f"Error in calculate_scores: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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

def simple_skill_extraction(text):
    """Fallback skill extraction without spaCy"""
    skills = {}
    text_lower = text.lower()
    
    # Common skills to look for
    common_skills = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php',
        'django', 'flask', 'fastapi', 'spring', 'react', 'angular', 'vue', 'node',
        'tensorflow', 'pytorch', 'scikit-learn', 'machine learning', 'ai',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes',
        'postgresql', 'mysql', 'mongodb', 'sql', 'git', 'html', 'css'
    }
    
    for skill in common_skills:
        if skill in text_lower:
            # Simple scoring based on occurrence
            count = text_lower.count(skill)
            skills[skill] = min(5, 1 + count * 0.5)  # Basic weight calculation
    
    return skills