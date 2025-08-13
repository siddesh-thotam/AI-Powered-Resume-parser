document.addEventListener('DOMContentLoaded', function() {
    // Get all elements with null checks
    const analyzeBtn = document.getElementById('analyze-btn');
    const jobDescInput = document.getElementById('job-description');
    const resumeUpload = document.getElementById('resume-upload');
    const resultsContainer = document.getElementById('results-container');

    // Verify all required elements exist
    if (!analyzeBtn || !jobDescInput || !resumeUpload || !resultsContainer) {
        console.error('Critical elements missing from page!');
        return;
    }

    analyzeBtn.addEventListener('click', async function() {
        console.log("Analyze button clicked");
        
        try {
            // Validate inputs
            const jobDesc = jobDescInput.value.trim();
            const files = resumeUpload.files;
            
            if (!jobDesc) {
                throw new Error('Please enter a job description');
            }
            
            if (!files || files.length === 0) {
                throw new Error('Please upload at least one resume');
            }

            // Show loading state
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Processing...';
            resultsContainer.innerHTML = '<p class="loading">Analyzing resumes...</p>';

            // Prepare form data
            const formData = new FormData();
            formData.append('job_description', jobDesc);
            
            for (let i = 0; i < files.length; i++) {
                formData.append('resumes', files[i]);
            }

            // API call
            const response = await fetch('/api/rank/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                    'Accept': 'application/json'
                }
            });

            // Handle response
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log("API response data:", data);
            
            if (!data) {
                throw new Error('No data received from server');
            }

            displayResults(data);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            showError(error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Resumes';
        }
    });

    function displayResults(data) {
        try {
            // Clear previous results
            resultsContainer.innerHTML = '';
            
            // Validate data structure
            if (!data?.results || !Array.isArray(data.results)) {
                throw new Error('Invalid results format');
            }

            if (data.results.length === 0) {
                resultsContainer.innerHTML = '<p class="no-results">No matching results found</p>';
                return;
            }

            // Create result cards
            data.results.forEach(result => {
                if (!result || typeof result !== 'object') return;
                
                const card = document.createElement('div');
                card.className = 'result-card';
                
                // Safely handle possible missing properties
                const filename = result.filename || 'Untitled Resume';
                const score = typeof result.score === 'number' ? result.score.toFixed(1) : '0.0';
                
                card.innerHTML = `
                    <h3>${escapeHtml(filename)}</h3>
                    <div class="score-display">
                        <span class="score-label">Match Score:</span>
                        <span class="score-value">${score}%</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-progress" style="width: ${score}%"></div>
                    </div>
                `;
                
                resultsContainer.appendChild(card);
            });

        } catch (error) {
            console.error('Error displaying results:', error);
            showError('Could not display results properly');
        }
    }

    function showError(message) {
        resultsContainer.innerHTML = `
            <div class="error-message">
                <p>${escapeHtml(message)}</p>
                <button onclick="location.reload()">Try Again</button>
            </div>
        `;
    }

    // Basic HTML escaping for security
    function escapeHtml(unsafe) {
        return unsafe?.toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;") || '';
    }

    // CSRF token helper
    function getCookie(name) {
        if (!document.cookie) return null;
        
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.startsWith(name + '=')) {
                return decodeURIComponent(cookie.substring(name.length + 1));
            }
        }
        return null;
    }
});