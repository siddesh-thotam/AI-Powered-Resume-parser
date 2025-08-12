import spacy
import pdfminer.high_level
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return pdfminer.high_level.extract_text(file_path)
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(file_path, 'r') as f:
            return f.read()

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc 
                    if not token.is_stop and not token.is_punct])

def extract_skills(text):
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            skills.add(ent.text.lower())
    return list(skills)

def calculate_scores(job_text, resume_text, job_skills, resume_skills):
    # Keyword matching
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([job_text, resume_text])
    keyword_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    
    # Skills matching
    common_skills = set(job_skills) & set(resume_skills)
    skills_score = len(common_skills) / len(job_skills) if job_skills else 0
    
    return {
        'keyword_score': keyword_score,
        'skills_score': skills_score,
        'experience_score': 0.5  # Placeholder
    }