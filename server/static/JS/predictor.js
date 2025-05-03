function message(message_element, type, message){
    message_element.className = '';
    message_element.classList.add("alert", type, "mt-4");
    message_element.innerHTML = message;
    message_element.hidden = false;
}

document.addEventListener("DOMContentLoaded", function() {
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById("uploadForm");
    const message_element = document.getElementById("message")

    uploadForm.addEventListener("submit", function(event) {
        event.preventDefault();
        const fileExtension = fileInput.files[0].name.split('.').pop().toLowerCase();
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        
        if (fileExtension != "csv") {
            message(message_element, "alert-danger", "El archivo debe ser un csv.")
            return; 
        } else {
            message(message_element, "alert-info", "Procesando fichero csv...")
        }
   
        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then((data) => {
            if (data['error']){
                message(message_element, "alert-danger", data['error'])
            } else {
                message(message_element, "alert-success", data['message'])
            }
        })
        .catch((e) => {
            message(message_element, "alert-danger", e.message)
        });
    });   
});