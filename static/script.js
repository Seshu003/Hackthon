// script.js
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultCard = document.getElementById('resultCard');
    const predictionResult = document.getElementById('predictionResult');
    const demandStatus = document.getElementById('demandStatus');
    const toast = document.getElementById('toast');

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading overlay
        showLoading();
        
        // Get form data
        const formData = new FormData(form);
        const data = {
            seasons: formData.get('seasons'),
            day: formData.get('day'),
            temp: parseInt(formData.get('temp')),
            a_temp: parseInt(formData.get('a_temp')),
            humidity: parseInt(formData.get('humidity')),
            wind: parseFloat(formData.get('wind'))
        };
        
        try {
            // Make API call to Flask backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Hide loading overlay
            hideLoading();
            
            // Display result
            displayResult(result.prediction[0]);
            
            // Show success toast
            showToast('Prediction completed successfully!', 'success');
            
        } catch (error) {
            console.error('Error:', error);
            hideLoading();
            showToast('Error making prediction. Please try again.', 'error');
        }
    });

    // Display prediction result
    function displayResult(prediction) {
        const roundedPrediction = Math.round(prediction);
        
        // Update result number
        const resultNumber = predictionResult.querySelector('.result-number');
        resultNumber.textContent = roundedPrediction;
        
        // Animate the number
        animateNumber(resultNumber, 0, roundedPrediction, 1000);
        
        // Update demand status
        updateDemandStatus(roundedPrediction);
        
        // Scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Update demand status based on prediction
    function updateDemandStatus(prediction) {
        const statusIndicator = demandStatus.querySelector('.status-indicator');
        const statusText = demandStatus.querySelector('.status-text');
        
        if (prediction < 100) {
            statusIndicator.className = 'status-indicator low';
            statusText.textContent = 'Low demand expected';
        } else if (prediction < 300) {
            statusIndicator.className = 'status-indicator medium';
            statusText.textContent = 'Medium demand expected';
        } else {
            statusIndicator.className = 'status-indicator high';
            statusText.textContent = 'High demand expected';
        }
    }

    // Animate number counting
    function animateNumber(element, start, end, duration) {
        const startTime = performance.now();
        
        function updateNumber(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = Math.floor(start + (end - start) * easeOutQuart(progress));
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            }
        }
        
        requestAnimationFrame(updateNumber);
    }

    // Easing function for smooth animation
    function easeOutQuart(t) {
        return 1 - Math.pow(1 - t, 4);
    }

    // Show loading overlay
    function showLoading() {
        loadingOverlay.classList.add('show');
        document.body.style.overflow = 'hidden';
    }

    // Hide loading overlay
    function hideLoading() {
        loadingOverlay.classList.remove('show');
        document.body.style.overflow = 'auto';
    }

    // Show toast notification
    function showToast(message, type) {
        const toastContent = toast.querySelector('.toast-content');
        const toastIcon = toast.querySelector('.toast-icon');
        const toastMessage = toast.querySelector('.toast-message');
        
        // Set icon based on type
        if (type === 'success') {
            toastIcon.className = 'toast-icon fas fa-check-circle';
            toast.className = 'toast success';
        } else if (type === 'error') {
            toastIcon.className = 'toast-icon fas fa-exclamation-circle';
            toast.className = 'toast error';
        }
        
        toastMessage.textContent = message;
        toast.classList.add('show');
        
        // Auto hide after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    // Smooth scrolling for navigation
    window.scrollToSection = function(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth' });
        }
    };

    // Form validation
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearFieldError);
    });

    function validateField(e) {
        const field = e.target;
        const value = field.value.trim();
        
        // Remove existing error styling
        field.classList.remove('error');
        
        // Basic validation
        if (field.hasAttribute('required') && !value) {
            showFieldError(field, 'This field is required');
            return false;
        }
        
        // Number validation
        if (field.type === 'number') {
            const num = parseFloat(value);
            const min = parseFloat(field.getAttribute('min'));
            const max = parseFloat(field.getAttribute('max'));
            
            if (isNaN(num)) {
                showFieldError(field, 'Please enter a valid number');
                return false;
            }
            
            if (min !== null && num < min) {
                showFieldError(field, `Value must be at least ${min}`);
                return false;
            }
            
            if (max !== null && num > max) {
                showFieldError(field, `Value must be at most ${max}`);
                return false;
            }
        }
        
        return true;
    }

    function showFieldError(field, message) {
        field.classList.add('error');
        
        // Remove existing error message
        const existingError = field.parentNode.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }
        
        // Add new error message
        const errorElement = document.createElement('span');
        errorElement.className = 'field-error';
        errorElement.textContent = message;
        field.parentNode.appendChild(errorElement);
    }

    function clearFieldError(e) {
        const field = e.target;
        field.classList.remove('error');
        
        const errorElement = field.parentNode.querySelector('.field-error');
        if (errorElement) {
            errorElement.remove();
        }
    }
});

// Add CSS for field validation
const style = document.createElement('style');
style.textContent = `
    .form-group input.error,
    .form-group select.error {
        border-color: var(--error-color);
        box-shadow: 0 0 0 3px rgba(255, 68, 68, 0.1);
    }
    
    .field-error {
        color: var(--error-color);
        font-size: 0.85rem;
        margin-top: 0.25rem;
        display: block;
    }
`;
document.head.appendChild(style);
