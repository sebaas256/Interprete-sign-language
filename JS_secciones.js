document.addEventListener('DOMContentLoaded', function() {
    const footer = document.querySelector('.footer-container');

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                footer.classList.add('visible'); 
            }else{
                footer.classList.remove('visible');
            }
        });
    }, {
        threshold: 0.8
    });

    observer.observe(footer);
});