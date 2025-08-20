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

    // Enhanced loading animation with segmented progress bar
    function showLoadingState() {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = `
            <div class="loading-spinner" style="width: 16px; height: 16px; margin-right: 8px; display: inline-block;"></div>
            Processing...
        `;
        
        resultsContainer.innerHTML = `
            <div class="loading">
                <div class="loading-spinner"></div>
                <p style="margin-bottom: 12px; font-weight: 500;">Analyzing resumes...</p>
                <div class="progress-bar">
                    <div class="segmented-progress">
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                        <div class="segment"></div>
                    </div>
                </div>
                <p style="color: #666; font-size: 12px; margin-top: 12px;">
                    This may take a few moments depending on file size
                </p>
            </div>
        `;
    }

    function hideLoadingState() {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `
            <span class="window-icon icon-search"></span>
            Analyze Resumes
        `;
    }

    analyzeBtn.addEventListener('click', async function() {
        console.log("Analyze button clicked");
        
        try {
            // Validate inputs
            const jobDesc = jobDescInput.value.trim();
            const files = resumeUpload.files;
            
            if (!jobDesc) {
                showErrorDialog('Please enter a job description');
                return;
            }
            
            if (!files || files.length === 0) {
                showErrorDialog('Please upload at least one resume');
                return;
            }

            // Show loading state
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

            const response = await fetch('/ranker/rank/', {
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

            displayResults(data);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            
            if (error.name === 'AbortError') {
                showErrorDialog('Request timed out. Please try again with smaller files.');
            } else {
                showErrorDialog(error.message);
            }
        } finally {
            hideLoadingState();
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
        
         if (data.jd_summary) {
            const jdSummaryCard = document.createElement('div');
            jdSummaryCard.className = 'feedback-card';
            jdSummaryCard.innerHTML = `
                <div class="feedback-header">
                    <span class="feedback-icon">📋</span>
                    Job Description Summary
                </div>
                <div class="feedback-section">
                    <h3>What They're Looking For</h3>
                    <div class="jd-summary">
                        <p>${escapeHtml(data.jd_summary)}</p>
                    </div>
                </div>
            `;
            resultsContainer.appendChild(jdSummaryCard);
        }


        if (data.results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="no-results">
                    <p>No matching results found</p>
                    <p style="color: #666; font-size: 12px; margin-top: 8px;">
                        Try adjusting your job description or uploading different resume formats
                    </p>
                </div>
            `;
            return;
        }

        // Sort results by score (highest first)
        const sortedResults = data.results.sort((a, b) => (b.score || 0) - (a.score || 0));

        // Add results summary
        const resultsHeader = document.createElement('div');
        resultsHeader.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding: 12px; background: linear-gradient(to bottom, #f8f8f8, #e8e8e8); border: 1px solid #c0c0c0;">
                <span style="font-weight: 500;">Analysis Complete</span>
                <span style="color: #666; font-size: 12px;">${sortedResults.length} candidate(s) analyzed</span>
            </div>
        `;
        resultsContainer.appendChild(resultsHeader);

        // Create result cards
        sortedResults.forEach((result, index) => {
            if (!result || typeof result !== 'object') return;
            
            const card = document.createElement('div');
            card.className = 'result-card-enhanced';
            
            // Safely handle possible missing properties
            const filename = result.filename || 'Untitled Resume';
            const score = typeof result.score === 'number' ? result.score : 0;
            const details = result.details || {};
            
            // Determine rank emoji and score color
            let rankEmoji = '📄';
            let scoreClass = 'result-score';
            if (index === 0 && score >= 70) rankEmoji = '🥇';
            else if (index === 1 && score >= 60) rankEmoji = '🥈';
            else if (index === 2 && score >= 50) rankEmoji = '🥉';

            // Set score color based on performance
            let scoreStyle = '';
            if (score >= 80) scoreStyle = 'background: linear-gradient(to bottom, #27ae60, #239954); border-color: #239954;';
            else if (score >= 60) scoreStyle = 'background: linear-gradient(to bottom, #f39c12, #e67e22); border-color: #e67e22;';
            else scoreStyle = 'background: linear-gradient(to bottom, #e74c3c, #c0392b); border-color: #c0392b;';

            card.innerHTML = `
                <div class="result-header">
                    <div class="result-filename">
                        ${rankEmoji} ${escapeHtml(filename)}
                    </div>
                    <div class="result-score" style="${scoreStyle}">
                        ${score.toFixed(1)}%
                    </div>
                </div>
                
                <div class="score-bar">
                    <div class="score-progress" style="width: ${score}%;"></div>
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
                
                ${details.skill_weights && Object.keys(details.skill_weights).length > 0 ? `
                    <div style="margin-top: 12px; padding: 12px; background: #f0f8ff; border-top: 1px solid #ddd;">
                        <div style="color: #666; font-size: 11px; margin-bottom: 8px; text-transform: uppercase; font-weight: 500;">Skill Importance:</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                            ${Object.entries(details.skill_weights).map(([skill, weight]) => {
                                const isMatched = details.matched_skills && details.matched_skills.includes(skill);
                                const isMissing = details.missing_skills && details.missing_skills.includes(skill);
                                const weightPercent = Math.round((weight / 5) * 100);
                                const bgColor = isMatched ? 
                                    `hsl(120, 100%, ${85 - weightPercent/4}%)` : 
                                    `hsl(0, 100%, ${85 - weightPercent/4}%)`;
                                return `
                                    <span style="
                                        background: ${bgColor};
                                        border: 1px solid #ddd;
                                        padding: 2px 6px;
                                        border-radius: 2px;
                                        font-size: 11px;
                                        display: inline-block;
                                        margin: 2px;
                                        position: relative;
                                    ">
                                        ${escapeHtml(skill)}
                                        <span style="
                                            position: absolute;
                                            bottom: -5px;
                                            left: 0;
                                            right: 0;
                                            height: 2px;
                                            background: ${isMatched ? '#27ae60' : '#e74c3c'};
                                            width: ${weightPercent}%;
                                        "></span>
                                    </span>
                                `;
                            }).join('')}
                        </div>
                        <div style="margin-top: 8px; font-size: 11px; color: #666;">
                            <span style="color: #27ae60;">■</span> = Has skill • 
                            <span style="color: #e74c3c;">■</span> = Missing skill •
                            Bar length = Importance
                        </div>
                    </div>
                ` : ''}
                
                ${result.error ? `
                    <div style="margin-top: 12px; padding: 12px; background: #fff5f5; border: 1px solid #e74c3c; color: #c53030;">
                        ⚠️ Error: ${escapeHtml(result.error)}
                    </div>
                ` : ''}
            `;
            
             resultsContainer.appendChild(card);

            // Add Skill Gap Analysis Feedback Section
            if (details.gap_analysis && details.gap_analysis.length > 0) {
                const feedbackCard = document.createElement('div');
                feedbackCard.className = 'feedback-card';
                feedbackCard.innerHTML = `
                    <div class="feedback-header">
                        <span class="feedback-icon">📊</span>
                        Feedback Report
                    </div>
                    
                    <div class="feedback-section">
                        <h3>Skill Gap Analysis</h3>
                        <p>The candidate is missing these important skills:</p>
                        
                        <div class="skill-gap-container">
                            ${details.gap_analysis.map(gap => `
                                <div class="skill-gap-item">
                                    <div class="skill-gap-skill">
                                        <span class="skill-category ${gap.category.toLowerCase().replace(' ', '-')}">
                                            ${gap.category}
                                        </span>
                                        ${escapeHtml(gap.skill)}
                                    </div>
                                    <div class="skill-gap-importance">
                                        Importance: 
                                        <span class="importance-level" style="width: ${gap.importance * 20}%">
                                            ${gap.importance.toFixed(1)}/5
                                        </span>
                                    </div>
                                    <div class="skill-suggestion">
                                        ${generateSkillSuggestion(gap.skill)}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
                resultsContainer.appendChild(feedbackCard);
            }

            // Add Strength Highlighting Section - THIS SHOULD BE INSIDE THE LOOP
            if (details.strengths && details.strengths.length > 0) {
                const strengthsCard = document.createElement('div');
                strengthsCard.className = 'feedback-card';
                strengthsCard.innerHTML = `
                    <div class="feedback-header">
                        <span class="feedback-icon">💪</span>
                        Key Strengths
                    </div>
                    
                    <div class="feedback-section">
                        <h3>Candidate Strengths</h3>
                        <p>Areas where this candidate excels:</p>
                        
                        <div class="strengths-container">
                            ${details.strengths.map(strength => `
                                <div class="strength-item">
                                    <div class="strength-skill">
                                        <span class="skill-category ${strength.category.toLowerCase().replace(' ', '-')}">
                                            ${strength.category}
                                        </span>
                                        ${escapeHtml(strength.skill)}
                                        ${strength.relevance === 'job-specific' ? 
                                            '<span class="relevance-badge job-specific" title="Directly mentioned in job description">JD</span>' : 
                                            '<span class="relevance-badge general" title="Generally valuable skill">GEN</span>'}
                                    </div>
                                    <div class="strength-level">
                                        Strength: 
                                        <span class="level-indicator" style="width: ${strength.strength_level * 20}%">
                                            ${strength.strength_level.toFixed(1)}/5
                                        </span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
                resultsContainer.appendChild(strengthsCard);
            }
        });

        // Add summary statistics if more than one result
        if (sortedResults.length > 1) {
            const avgScore = sortedResults.reduce((sum, r) => sum + (r.score || 0), 0) / sortedResults.length;
            const highScore = Math.max(...sortedResults.map(r => r.score || 0));
            
            const summary = document.createElement('div');
            summary.className = 'summary-stats';
            summary.innerHTML = `
                <h4>📈 Summary Statistics</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Average Score</div>
                        <div class="stat-value">${avgScore.toFixed(1)}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Highest Score</div>
                        <div class="stat-value" style="color: #27ae60;">${highScore.toFixed(1)}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Total Candidates</div>
                        <div class="stat-value">${sortedResults.length}</div>
                    </div>
                </div>
            `;
            resultsContainer.appendChild(summary);
        }

    } catch (error) {
        console.error('Error displaying results:', error);
        showErrorDialog('Could not display results properly');
    }
}


    // Helper function for skill suggestions
    function generateSkillSuggestion(skill) {
    const suggestions = {
        'python': 'Consider highlighting any Python coursework or personal projects',
        'aws': 'Look for cloud computing certifications or online courses',
        'react': 'Build a small React project to demonstrate understanding',
        'communication': 'Add examples of presentations or collaborative projects'
    };
    
    const defaultSuggestion = 'This skill could be developed through online courses or practical projects';
    
    return suggestions[skill] || defaultSuggestion;
}


    function showErrorDialog(message) {
        // Create Windows-style error dialog
        const dialog = document.createElement('div');
        dialog.className = 'status-dialog';
        dialog.innerHTML = `
            <div class="status-dialog-header">
                <span style="color: #e74c3c;">⚠️</span>
                Error
            </div>
            <div class="status-dialog-content">
                <p style="margin-bottom: 16px;">${escapeHtml(message)}</p>
            </div>
            <div class="status-dialog-buttons">
                <button class="btn" onclick="this.closest('.status-dialog').remove()">
                    OK
                </button>
            </div>
        `;
        
        // Add backdrop
        const backdrop = document.createElement('div');
        backdrop.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.3);
            z-index: 999;
        `;
        backdrop.onclick = () => {
            backdrop.remove();
            dialog.remove();
        };
        
        document.body.appendChild(backdrop);
        document.body.appendChild(dialog);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (dialog.parentNode) {
                backdrop.remove();
                dialog.remove();
            }
        }, 5000);
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

    // Add CSS animations for the new design
    const style = document.createElement('style');
    style.textContent = `
        @keyframes progress-loading {
            0% { width: 0%; }
            50% { width: 70%; }
            100% { width: 100%; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-card-enhanced {
            animation: fadeIn 0.4s ease forwards;
        }
        
        .status-dialog {
            animation: fadeIn 0.2s ease forwards;
        }
    `;
    document.head.appendChild(style);

    // Enhanced file upload with drag and drop
    const fileUploadArea = resumeUpload.parentElement;
    
    // Drag and drop events
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
        fileUploadArea.style.borderColor = '#0078d4';
        fileUploadArea.style.backgroundColor = '#f0f8ff';
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
        
        // Show file count
        updateFileCounter();
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        resumeUpload.dispatchEvent(event);
    }

    // File counter
    resumeUpload.addEventListener('change', updateFileCounter);

    function updateFileCounter() {
        const fileCount = resumeUpload.files.length;
        const existingCounter = fileUploadArea.querySelector('.file-counter');
        
        if (existingCounter) {
            existingCounter.remove();
        }
        
        if (fileCount > 0) {
            const counter = document.createElement('div');
            counter.className = 'file-counter';
            counter.style.cssText = `
                margin-top: 8px;
                color: #666;
                font-size: 12px;
                text-align: center;
            `;
            counter.textContent = `${fileCount} file${fileCount > 1 ? 's' : ''} selected`;
            fileUploadArea.appendChild(counter);
        }
    }

    // Initialize file counter on page load
    updateFileCounter();
});