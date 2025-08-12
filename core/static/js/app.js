document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyze-btn');
    
    analyzeBtn.addEventListener('click', function() {
        console.log("Analyze button clicked!");  // Check if this appears in browser console
        
        const jobDesc = document.getElementById('job-description').value;
        const resumeUpload = document.getElementById('resume-upload');
        
        if (!jobDesc || resumeUpload.files.length === 0) {
            alert('Please enter a job description and upload at least one resume');
            return;
        }
        
        // Show processing state
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Processing...';
        
        // Create FormData for the API request
        const formData = new FormData();
        formData.append('job_description', jobDesc);
        
        // Add all files
        for (let i = 0; i < resumeUpload.files.length; i++) {
            formData.append('resumes', resumeUpload.files[i]);
        }
        
        // Make the API call
        fetch('/api/rank/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCookie('csrftoken'),
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log("API response:", data);
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis');
        })
        .finally(() => {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Resumes';
        });
    });
    
    function displayResults(data) {
        const resultsContainer = document.getElementById('results-container');
        resultsContainer.innerHTML = '';
        
        if (data && data.length > 0) {
            data.forEach(result => {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `
                    <h3>${result.resume_name || 'Resume'}</h3>
                    <div class="score-details">
                        Overall Score: ${result.score.toFixed(1)}%
                    </div>
                    <div class="score-bar">
                        <div class="score-progress" style="width: ${result.score}%"></div>
                    </div>
                `;
                resultsContainer.appendChild(card);
            });
        } else {
            resultsContainer.innerHTML = '<p class="text-center">No results found</p>';
        }
    }
    
    // CSRF token helper function
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