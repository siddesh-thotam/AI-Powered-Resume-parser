document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyze-btn');
    
    analyzeBtn.addEventListener('click', async function() {
        console.log("Button clicked!"); // Verify click works
        
        const jobDesc = document.getElementById('job-description').value;
        const resumeUpload = document.getElementById('resume-upload');
        
        if (!jobDesc || resumeUpload.files.length === 0) {
            alert('Please enter a job description and upload at least one resume');
            return;
        }
        
        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Processing...';
        
        try {
            const formData = new FormData();
            formData.append('job_description', jobDesc);
            
            // Add all files
            for (let i = 0; i < resumeUpload.files.length; i++) {
                formData.append('resumes', resumeUpload.files[i]);
            }
            
            const response = await fetch('/api/rank/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                }
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            console.log("API response:", data);
            displayResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing resumes: ' + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Resumes';
        }
    });
    
    function displayResults(data) {
        const resultsContainer = document.getElementById('results-container');
        resultsContainer.innerHTML = '';
        
        if (data && data.results && data.results.length > 0) {
            data.results.forEach(result => {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `
                    <h3>${result.filename}</h3>
                    <div>Score: ${result.score.toFixed(1)}%</div>
                `;
                resultsContainer.appendChild(card);
            });
        } else {
            resultsContainer.innerHTML = '<p>No results found</p>';
        }
    }
    
    // CSRF token helper
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});