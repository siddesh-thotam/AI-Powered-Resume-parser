import spacy
import pdfminer.high_level
import docx  # Added missing import
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from spacy.matcher import Matcher

# Load medium/large model for better word vectors
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import en_core_web_md
    nlp = en_core_web_md.load()
    print("Loaded en_core_web_md directly")
except Exception as e:
    print(f"Error loading model: {e}")
    nlp = spacy.load("en_core_web_sm")

class SkillNormalizer:
    def __init__(self):
        # Initialize with common skill variations
        self.skill_map = self._create_skill_mapping()
        self.word_forms = self._create_word_forms()
        
    def _create_skill_mapping(self):
        """Create mapping for common skill variations"""
        return {
            # Programming languages
            'js': 'javascript',
            'javascript es6': 'javascript',
            'es6': 'javascript',
            'typescript': 'typescript',
            'ts': 'typescript',
            'python': 'python',
            'python3': 'python',
            'py': 'python',
            'java': 'java',
            'c++': 'c++',
            'cpp': 'c++',
            'c#': 'c#',
            'csharp': 'c#',
            
            # Web technologies
            'html': 'html',
            'html5': 'html',
            'css': 'css',
            'css3': 'css',
            'react': 'react',
            'reactjs': 'react',
            'react.js': 'react',
            'angular': 'angular',
            'angularjs': 'angular',
            'vue': 'vue',
            'vuejs': 'vue',
            'vue.js': 'vue',
            
            # Cloud/DevOps
            'aws': 'amazon web services',
            'amazon web services': 'amazon web services',
            'azure': 'microsoft azure',
            'gcp': 'google cloud platform',
            'google cloud': 'google cloud platform',
            'docker': 'docker',
            'kubernetes': 'kubernetes',
            'k8s': 'kubernetes',
            'terraform': 'terraform',
            'ansible': 'ansible',
            
            # Databases
            'postgres': 'postgresql',
            'postgresql': 'postgresql',
            'postgres db': 'postgresql',
            'mysql': 'mysql',
            'mongo': 'mongodb',
            'mongodb': 'mongodb',
            'sql': 'sql',
            'nosql': 'nosql',
            
            # Other common skills
            'rest': 'rest api',
            'restful': 'rest api',
            'rest api': 'rest api',
            'graphql': 'graphql',
            'git': 'git',
            'github': 'git',
            'gitlab': 'git',
            'ci/cd': 'ci/cd',
            'continuous integration': 'ci/cd',
            'continuous deployment': 'ci/cd',
            'machine learning': 'machine learning',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'artificial intelligence': 'artificial intelligence',
            'deep learning': 'deep learning',
            'dl': 'deep learning',
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
        """Normalize a skill name to standard form"""
        if not skill:
            return skill
            
        # Convert to lowercase and remove special characters
        skill = skill.lower().strip()
        skill = re.sub(r'[^a-z0-9+#\. ]', '', skill)
        
        # Check direct mapping first
        if skill in self.skill_map:
            return self.skill_map[skill]
            
        # Check for word forms
        for word, normalized in self.word_forms.items():
            if word in skill:
                skill = skill.replace(word, normalized)
                
        # Handle version numbers (e.g., python 3 -> python)
        skill = re.sub(r'\s*\d+(\.\d+)*\s*', ' ', skill).strip()
        
        # Handle common prefixes/suffixes
        skill = re.sub(r'^(knowledge of|experience with|proficient in|expertise in)\s+', '', skill)
        skill = re.sub(r'\s+(framework|library|tool|technology|language)$', '', skill)
        
        return skill.strip()

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
    """More sophisticated text processing"""
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
    """Enhanced skill extraction with normalization"""
    doc = nlp(text)
    skills = defaultdict(int)
    
    # Skill patterns - expanded list
    patterns = [
        # Programming languages
        [{"LOWER": {"IN": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "go", "rust"]}}],
        [{"LOWER": "django"}, {"LOWER": "framework", "OP": "?"}],
        [{"LOWER": "flask"}, {"LOWER": "framework", "OP": "?"}],
        [{"LOWER": "spring"}, {"LOWER": "framework", "OP": "?"}],
        [{"LOWER": "react"}, {"LOWER": {"IN": ["js", "native"]}, "OP": "?"}],
        [{"LOWER": "angular"}, {"IS_DIGIT": True, "OP": "?"}],
        [{"LOWER": "vue"}, {"LOWER": "js", "OP": "?"}],
        
        # Databases
        [{"LOWER": {"IN": ["postgresql", "mysql", "mongodb", "redis", "sqlite", "oracle"]}}],
        [{"LOWER": "sql"}, {"LOWER": "server", "OP": "?"}],
        
        # Cloud/DevOps
        [{"LOWER": {"IN": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible"]}}],
        [{"LOWER": "amazon"}, {"LOWER": "web"}, {"LOWER": "services"}],
        [{"LOWER": "google"}, {"LOWER": "cloud"}],
        [{"LOWER": "microsoft"}, {"LOWER": "azure"}],
        
        # Other technologies
        [{"LOWER": "rest"}, {"LOWER": "api"}],
        [{"LOWER": "graphql"}],
        [{"LOWER": "git"}],
        [{"LOWER": "jenkins"}],
        [{"LOWER": "ci/cd"}],
    ]
    
    matcher = Matcher(nlp.vocab)
    for pattern in patterns:
        matcher.add("SKILLS", [pattern])
    
    # First pass - count raw occurrences
    matches = matcher(doc)
    for match_id, start, end in matches:
        skill_text = doc[start:end].text.lower()
        normalized_skill = skill_normalizer.normalize_skill(skill_text)
        if normalized_skill:
            skills[normalized_skill] += 1
    
    # Second pass - check for emphasis (only uppercase since we can't detect bold in plain text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "TECH"]:
            normalized_skill = skill_normalizer.normalize_skill(ent.text)
            if normalized_skill:
                # Only check for uppercase (no bold/title detection in plain text)
                is_emphasized = ent.text.isupper()
                skills[normalized_skill] += 3 if is_emphasized else 1
    
    # Third pass - check for "required", "must have", etc.
    for token in doc:
        if token.text.lower() in ['required', 'requirement', 'must', 'essential']:
            # Check both children and the token's head for skills
            for related in list(token.children) + [token.head]:
                if related.ent_type_ in ["ORG", "TECH"]:
                    normalized_skill = skill_normalizer.normalize_skill(related.text)
                    if normalized_skill:
                        skills[normalized_skill] += 5
    
    # Normalize weights to 1-5 scale
    if skills:
        max_weight = max(skills.values())
        return {skill: 1 + 4 * (weight/max_weight) for skill, weight in skills.items()}
    
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
    """Enhanced scoring algorithm with weighted matching"""
    # Extract skills with weights from job description
    weighted_job_skills = extract_skills_with_weights(job_text)
    

     # Strength Highlighting - identify skills that are strong in the resume
    strength_analysis = []
    resume_skill_weights = extract_skills_with_weights(resume_text)


    # If weight extraction failed, fall back to equal weights
    if not weighted_job_skills:
        weighted_job_skills = {skill: 1 for skill in job_skills}
    
    # Normalize all skills
    normalized_job_skills = {skill_normalizer.normalize_skill(skill): weight 
                           for skill, weight in weighted_job_skills.items()}


    if not isinstance(resume_skills, set):
        resume_skills = set(resume_skills) if resume_skills else set()
    
    # Normalize resume skills
    normalized_resume_skills = {skill_normalizer.normalize_skill(skill) 
                              for skill in resume_skills}
    
    # Remove empty strings from normalization
    normalized_job_skills = {skill: weight for skill, weight in normalized_job_skills.items() 
                           if skill}
    normalized_resume_skills = {skill for skill in normalized_resume_skills if skill}
    
    # Keyword matching with TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([job_text, resume_text])
        keyword_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        keyword_score = 0

    if resume_skill_weights:
        # Normalize resume skill weights (1-5 scale)
        max_resume_weight = max(resume_skill_weights.values()) if resume_skill_weights.values() else 1
        normalized_resume_skills = {skill: 1 + 4 * (weight/max_resume_weight) 
                                  for skill, weight in resume_skill_weights.items()}
    
    # Weighted skills matching
    total_weight = sum(normalized_job_skills.values())
    matched_weight = 0
    missing_skills = []
    
    for skill, weight in normalized_job_skills.items():
        if skill in normalized_resume_skills:
            matched_weight += weight
        else:
            missing_skills.append(skill)
    
    skills_score = matched_weight / total_weight if total_weight > 0 else 0
    
    # Experience matching
    resume_exp = extract_experience(resume_text)
    job_exp = extract_experience(job_text) or 3
    exp_score = min(1, resume_exp / job_exp) if job_exp > 0 else 0
    
    # Generate gap analysis
    gap_analysis = []
    for skill in missing_skills:
        importance = normalized_job_skills[skill]
        gap_analysis.append({
            'skill': skill,
            'importance': importance,
            'category': categorize_skill(skill)
        })

        if resume_skill_weights:
        # Normalize resume skill weights (1-5 scale)
        max_resume_weight = max(resume_skill_weights.values()) if resume_skill_weights.values() else 1
        weighted_resume_skills = {skill_normalizer.normalize_skill(skill): 1 + 4 * (weight/max_resume_weight) 
                                for skill, weight in resume_skill_weights.items()}
        
        # Remove empty strings
        weighted_resume_skills = {skill: weight for skill, weight in weighted_resume_skills.items() if skill}
        
        for skill, weight in weighted_resume_skills.items():
            # Consider it a strength if:
            # 1. It's in the job description (relevant strength)
            # OR
            # 2. It's a generally valuable skill (even if not mentioned in JD)
            if skill in normalized_job_skills or is_generally_valuable_skill(skill):
                strength_analysis.append({
                    'skill': skill,
                    'strength_level': weight,
                    'category': categorize_skill(skill),
                    'relevance': 'job-specific' if skill in normalized_job_skills else 'general'
                })

    
    # Sort by importance (highest first)
    gap_analysis.sort(key=lambda x: x['importance'], reverse=True)
    
    strength_analysis.sort(key=lambda x: x['strength_level'], reverse=True)

    return {
        'keyword_score': min(1, max(0, keyword_score)),
        'skills_score': min(1, max(0, skills_score)),
        'experience_score': min(1, max(0, exp_score)),
        'missing_skills': missing_skills,
        'matched_skills': list(set(normalized_job_skills.keys()) & normalized_resume_skills),
        'weighted_skills': normalized_job_skills,
        'gap_analysis': gap_analysis,
        'strengths': strength_analysis[:10],
    }

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