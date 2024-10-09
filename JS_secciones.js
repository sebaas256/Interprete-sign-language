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

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});