document.addEventListener("DOMContentLoaded", function() {
  fetch('/api/footer')
    .then(response => response.json())
    .then(data => {
      document.getElementById('footer-content').innerHTML = data.footer_content;
    })
    .catch(error => {
      console.error('Error fetching footer content:', error);
      document.getElementById('footer-content').innerHTML = "Error loading footer content.";
    });
});
