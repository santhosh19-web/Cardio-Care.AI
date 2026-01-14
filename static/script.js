document.getElementById('predictionForm').addEventListener('submit', function() {
    const btn = document.querySelector('.btn-predict');
    btn.innerHTML = 'Processing...';
    btn.style.opacity = '0.7';
});