📄 AI Resume Parser

An intelligent web application that automatically **ranks resumes against job descriptions** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. Get instant **skill gap analysis** and **personalized improvement suggestions**.

---

## ✨ Features

* **Smart Resume Parsing**: Extract text from **PDF, DOCX, TXT** files
* **AI-Powered Matching**: Advanced NLP algorithms for skill extraction & matching
* **Comprehensive Scoring**: Multi-factor scoring system (*keywords, skills, experience*)
* **Skill Gap Analysis**: Detailed report on missing skills with actionable suggestions
* **Job Description Summarization**: Automatic extraction of key requirements from JDs
* **Production Ready**: Deployed on **Render.com** with error handling

---

## 🚀 Live Demo  
🔗 [Resume Ranker Live Demo](https://resume-ranker-gxvs.onrender.com)

It will take few minutes to load the website because i am using render's free tier plan.
---

## 🛠️ Technology Stack

**Backend**

* Framework: **Django 5.2.5** + **Django REST Framework**
* NLP: **NLTK**, **spaCy** (fallback support)
* File Processing: **pdfminer.six**, **python-docx**
* ML Processing: **scikit-learn**, **numpy**
* API: RESTful APIs with authentication

**Frontend**

* Templating: **Django Templates**
* Styling: **Custom CSS** (responsive)
* JS: **Vanilla JavaScript**

**Deployment**

* Platform: **Render.com**
* WSGI Server: **Gunicorn**
* Static Files: **WhiteNoise**
* Database: **PostgreSQL (Production)** / **SQLite (Development)**

---

## 📦 Installation

### **Prerequisites**

* Python **3.11+**
* **pip**
* **Virtual environment**

### **Setup (Local Development)**

```bash
# 1. Clone repository
git clone https://github.com/your-username/ai-resume-ranker.git
cd ai-resume-ranker

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# 5. Run migrations
python manage.py migrate

# 6. Start development server
python manage.py runserver
```

Visit 👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🎯 How It Works

1. **Text Extraction**

   * PDF → `pdfminer`
   * DOCX → `python-docx`
   * Fallback for multiple formats

2. **NLP Processing**

   * Tokenization & Cleaning → **NLTK**
   * Skill Extraction → Regex + contextual analysis
   * Skill Normalization → Custom mapping
   * Experience Extraction → Pattern matching

3. **Scoring Algorithm**

```python
overall_score = (0.5 * keyword_score +
                 0.3 * skills_score +
                 0.2 * experience_score) * 100
```

4. **AI Features**

   * **TF-IDF** for keyword matching
   * **Cosine similarity** for comparison
   * Intelligent skill gap analysis
   * Personalized improvement suggestions

---

## 📊 API Endpoints

* **POST /api/job-description/** → Upload job description

```json
{
  "text": "Job description content...",
  "title": "Software Engineer"
}
```

* **POST /api/resume/** → Upload resume

```json
{
  "file": "resume.pdf",
  "user": 1
}
```

* **POST /api/rank/** → Rank resumes

```json
{
  "job_description": "JD text...",
  "resumes": [file1, file2, ...]
}
```

---

## 🏗️ Project Structure

```
ai-resume-ranker/
├── core/                 # Django project settings
├── ranker/               # Main application
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   ├── nlp_processor.py  # AI + NLP logic
│   └── templates/        # Frontend templates
├── requirements.txt
├── render.yaml
└── README.md
```

---

## 🔧 Configuration

### **Environment Variables**

```
SECRET_KEY=your-django-secret-key
DEBUG=False
HUGGINGFACE_API_KEY=your-huggingface-key   # Optional
DATABASE_URL=your-database-url
```

### **Settings**

* **Development** → SQLite, Debug ON
* **Production** → PostgreSQL, Debug OFF, Static file handling

---

## 🚀 Deployment

**Render.com Deployment**

1. Connect GitHub repo to Render
2. Add environment variables
3. Deploy using `render.yaml`

**Manual Deployment**

```bash
python manage.py collectstatic --noinput
python manage.py migrate
gunicorn core.wsgi:application --bind 0.0.0.0:$PORT
```

---

## 🧪 Testing

```bash
# Run all tests
python manage.py test

# Test NLP processing
python manage.py test ranker.tests.test_nlp

# Test API endpoints
python manage.py test ranker.tests.test_api
```

---

## 📈 Performance Optimizations

* Efficient **chunk-based text processing**
* **Caching** processed results
* Async-ready (Celery support)
* Optimized DB queries & indexing

---

## 🔮 Future Enhancements

* ML models for improved matching
* Real-time WebSockets processing
* Advanced analytics dashboard
* Multi-language support
* LinkedIn integration
* ATS system integration
* Batch candidate ranking

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch → `git checkout -b feature/amazing-feature`
3. Commit changes → `git commit -m "Add amazing feature"`
4. Push → `git push origin feature/amazing-feature`
5. Open a PR 🚀

---

## 📄 License

Licensed under **MIT License** – see [LICENSE](LICENSE)

---

## 🙋 Support

* Open an **issue** on GitHub
* Check the **Wiki**
* Browse existing issues

---

## 🏆 Acknowledgments

* **NLTK** & **spaCy** for NLP
* **Django community**
* **Render.com** for deployment
* Open-source contributors 🙌

---

⭐ **If you found this project helpful, don’t forget to give it a star!**

Built with ❤️ using **Django, Python & modern web technologies**

---

Do you want me to also **add shields.io badges** (Python version, Django version, MIT License, Deployment status) at the **top of the README** to make it look even more professional?
