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

    // Add enhanced loading animation
    function showLoadingState() {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = `
            <span style="display: inline-flex; align-items: center; gap: 0.5rem;">
                <span style="animation: spin 1s linear infinite; display: inline-block;">⚡</span>
                Processing...
            </span>
        `;
        
        resultsContainer.innerHTML = `
            <div class="loading">
                <div style="font-size: 2rem; margin-bottom: 1rem; animation: pulse 1.5s infinite;">🔄</div>
                <p>Analyzing resumes with AI...</p>
                <div style="margin-top: 1rem; color: var(--gray-400); font-size: 0.9rem;">
                    This may take a few moments depending on file size
                </div>
            </div>
        `;

        // Update progress step
        const step3 = document.getElementById('step-3');
        if (step3) {
            step3.classList.add('active');
        }
    }

    function hideLoadingState() {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span>🚀 Analyze Resumes</span>';
        
        // Complete progress step
        const step3 = document.getElementById('step-3');
        if (step3) {
            step3.classList.remove('active');
            step3.classList.add('completed');
        }
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

            // Show enhanced loading state
            showLoadingState();

            // Prepare form data
            const formData = new FormData();
            formData.append('job_description', jobDesc);
            
            for (let i = 0; i < files.length; i++) {
                formData.append('resumes', files[i]);
            }

            // API call with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minutes timeout

            const response = await fetch('/api/rank/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                    'Accept': 'application/json'
                },
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            // Handle response
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || errorData.message || `Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log("API response data:", data);
            
            if (!data) {
                throw new Error('No data received from server');
            }

            displayEnhancedResults(data);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            
            if (error.name === 'AbortError') {
                showError('Request timed out. Please try again with smaller files.');
            } else {
                showError(error.message);
            }
        } finally {
            hideLoadingState();
        }
    });

    function displayEnhancedResults(data) {
        try {
            // Clear previous results
            resultsContainer.innerHTML = '';
            
            // Validate data structure
            if (!data?.results || !Array.isArray(data.results)) {
                throw new Error('Invalid results format');
            }

            if (data.results.length === 0) {
                resultsContainer.innerHTML = `
                    <div class="no-results">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">📭</div>
                        <p>No matching results found</p>
                        <p style="color: var(--gray-400); font-size: 0.9rem; margin-top: 0.5rem;">
                            Try adjusting your job description or uploading different resume formats
                        </p>
                    </div>
                `;
                return;
            }

            // Sort results by score (highest first)
            const sortedResults = data.results.sort((a, b) => (b.score || 0) - (a.score || 0));

            // Add results header
            const resultsHeader = document.createElement('div');
            resultsHeader.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(166, 125, 67, 0.3);">
                    <h3 style="color: var(--primary-gold); margin: 0;">📊 Analysis Complete</h3>
                    <span style="color: var(--gray-400); font-size: 0.9rem;">${sortedResults.length} candidate(s) analyzed</span>
                </div>
            `;
            resultsContainer.appendChild(resultsHeader);

            // Create enhanced result cards
            sortedResults.forEach((result, index) => {
                if (!result || typeof result !== 'object') return;
                
                const card = document.createElement('div');
                card.className = 'result-card-enhanced';
                card.style.animationDelay = `${index * 0.1}s`;
                card.style.animation = 'fadeInUp 0.6s ease forwards';
                
                // Safely handle possible missing properties
                const filename = result.filename || 'Untitled Resume';
                const score = typeof result.score === 'number' ? result.score : 0;
                const details = result.details || {};
                
                // Determine score color and rank
                let scoreColor = 'var(--primary-red)';
                let rank = '🥉';
                if (score >= 80) {
                    scoreColor = 'var(--success)';
                    rank = '🥇';
                } else if (score >= 60) {
                    scoreColor = 'var(--warning)';
                    rank = '🥈';
                }

                card.innerHTML = `
                    <div class="result-header">
                        <div class="result-filename">
                            ${rank} ${escapeHtml(filename)}
                        </div>
                        <div class="result-score" style="background: ${scoreColor};">
                            ${score.toFixed(1)}%
                        </div>
                    </div>
                    
                    <div class="score-bar">
                        <div class="score-progress" style="width: ${score}%; background: ${scoreColor};"></div>
                    </div>
                    
                    <div class="result-details">
                        <div class="detail-item">
                            <div class="detail-label">Keywords</div>
                            <div class="detail-value">${details.keywords || 'N/A'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Skills</div>
                            <div class="detail-value">${details.skills || 'N/A'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Experience</div>
                            <div class="detail-value">${details.experience || 'N/A'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Rank</div>
                            <div class="detail-value">#${index + 1}</div>
                        </div>
                    </div>
                    
                    ${details.matched_skills && details.matched_skills.length > 0 ? `
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(166, 125, 67, 0.2);">
                            <div style="color: var(--gray-400); font-size: 0.8rem; margin-bottom: 0.5rem;">MATCHED SKILLS:</div>
                            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                                ${details.matched_skills.map(skill => 
                                    `<span style="background: rgba(166, 125, 67, 0.2); color: var(--primary-gold); padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">${escapeHtml(skill)}</span>`
                                ).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${result.error ? `
                        <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; color: #fca5a5;">
                            ⚠️ Error: ${escapeHtml(result.error)}
                        </div>
                    ` : ''}
                `;
                
                resultsContainer.appendChild(card);
            });

            // Add summary statistics
            if (sortedResults.length > 1) {
                const avgScore = sortedResults.reduce((sum, r) => sum + (r.score || 0), 0) / sortedResults.length;
                const highScore = Math.max(...sortedResults.map(r => r.score || 0));
                
                const summary = document.createElement('div');
                summary.innerHTML = `
                    <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(166, 125, 67, 0.1); border: 1px solid rgba(166, 125, 67, 0.3); border-radius: 12px;">
                        <h4 style="color: var(--primary-gold); margin: 0 0 1rem 0;">📈 Summary Statistics</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                            <div style="text-align: center;">
                                <div style="color: var(--gray-400); font-size: 0.8rem;">Average Score</div>
                                <div style="color: var(--primary-gold); font-size: 1.2rem; font-weight: 600;">${avgScore.toFixed(1)}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: var(--gray-400); font-size: 0.8rem;">Highest Score</div>
                                <div style="color: var(--success); font-size: 1.2rem; font-weight: 600;">${highScore.toFixed(1)}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: var(--gray-400); font-size: 0.8rem;">Total Candidates</div>
                                <div style="color: var(--primary-gold); font-size: 1.2rem; font-weight: 600;">${sortedResults.length}</div>
                            </div>
                        </div>
                    </div>
                `;
                resultsContainer.appendChild(summary);
            }

        } catch (error) {
            console.error('Error displaying results:', error);
            showError('Could not display results properly');
        }
    }

    function showError(message) {
        resultsContainer.innerHTML = `
            <div class="error-message">
                <div style="font-size: 2rem; margin-bottom: 1rem;">❌</div>
                <p><strong>Error:</strong> ${escapeHtml(message)}</p>
                <button onclick="location.reload()" class="btn" style="margin-top: 1rem; background: var(--error);">
                    🔄 Try Again
                </button>
            </div>
        `;
    }

    // Enhanced HTML escaping for security
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe.toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
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

    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-card-enhanced {
            opacity: 0;
        }
    `;
    document.head.appendChild(style);

    // Add drag and drop functionality for file upload
    const fileUploadArea = resumeUpload.parentElement;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileUploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        fileUploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileUploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        fileUploadArea.style.borderColor = 'var(--primary-gold)';
        fileUploadArea.style.backgroundColor = 'rgba(166, 125, 67, 0.1)';
    }

    function unhighlight(e) {
        fileUploadArea.style.borderColor = '';
        fileUploadArea.style.backgroundColor = '';
    }

    fileUploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        resumeUpload.files = files;
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        resumeUpload.dispatchEvent(event);
    }
});