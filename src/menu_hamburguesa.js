const toggleButton = document.getElementById('navbarToggle');
const menu = document.getElementById('navbarMenu');
const body = document.body;

function closeMenu() {
    menu.classList.remove('active');
    toggleButton.classList.remove('active');
    body.classList.remove('menu-active'); 
}

toggleButton.addEventListener('click', () => {
    menu.classList.toggle('active');
    toggleButton.classList.toggle('active');
    body.classList.toggle('menu-active'); 
});

const menuLinks = menu.querySelectorAll('a');
menuLinks.forEach(link => {
    link.addEventListener('click', closeMenu); 
});
