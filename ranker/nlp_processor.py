import spacy
import pdfminer.high_level
import docx
from spacy.matcher import Matcher
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load medium/large model for better 

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import en_core_web_md
    nlp = en_core_web_md.load()
    print("Loaded en_core_web_md directly")
except Exception as e:
    print(f"Error loading model: {e}")
    nlp = spacy.load("en_core_web_sm")

    
nlp = spacy.load("en_core_web_md")  # Changed from sm to md for better word vectors

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

def extract_skills(text):
    """Enhanced skill extraction with patterns"""
    doc = nlp(text)
    skills = set()
    
    # Skill patterns
    patterns = [
        [{"LOWER": {"IN": ["python", "java", "javascript"]}}],
        [{"LOWER": "django"}, {"LOWER": "framework", "OP": "?"}],
        [{"LOWER": "postgresql"}],
        [{"LOWER": "rest"}, {"LOWER": "api"}],
        [{"LOWER": "aws"}],
        [{"LOWER": "docker"}],
    ]
    
    matcher = spacy.matcher.Matcher(nlp.vocab)
    for pattern in patterns:
        matcher.add("SKILLS", [pattern])
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        skills.add(doc[start:end].text.lower())
    
    # Add entities recognized as skills
    for ent in doc.ents:
        if ent.label_ in ["ORG", "TECH"]:
            skills.add(ent.text.lower())
    
    return list(skills)

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

def calculate_scores(job_text, resume_text, job_skills, resume_skills):
    """Enhanced scoring algorithm"""
    # Keyword matching with TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([job_text, resume_text])
        keyword_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        keyword_score = 0
    
    # Skills matching with partial matching
    required_skills = set(job_skills)
    resume_skills_set = set(resume_skills)
    
    # Basic exact match
    exact_match = len(required_skills & resume_skills_set)
    
    # Partial match (checks for substrings)
    partial_match = 0
    for req_skill in required_skills:
        for cand_skill in resume_skills_set:
            if req_skill in cand_skill or cand_skill in req_skill:
                partial_match += 0.5  # Partial credit
    
    skills_score = (exact_match + partial_match) / len(required_skills) if required_skills else 0
    
    # Experience matching (assuming job requires 3 years)
    resume_exp = extract_experience(resume_text)
    job_exp = 3  # Default expected experience
    exp_score = min(1, resume_exp / job_exp)  # Normalized to 0-1
    
    return {
        'keyword_score': min(1, max(0, keyword_score)),  # Ensure 0-1 range
        'skills_score': min(1, max(0, skills_score)),
        'experience_score': min(1, max(0, exp_score)),
    }