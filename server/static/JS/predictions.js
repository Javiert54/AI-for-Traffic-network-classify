function message(message_element, type, message){
    message_element.className = '';
    message_element.classList.add("alert", type, "mt-4");
    message_element.innerHTML = message;
    message_element.hidden = false;
}

document.addEventListener("DOMContentLoaded", function () {
    const message_element = document.getElementById("message")
    fetch("/show_predictions")
    .then(response => response.json())
    .then((data) => {
        if (data['error']) {
            message(message_element, "alert-danger", data['error'])
        } else {
            message(message_element, "alert-success", data['predictions'])
        }
    })
    .catch((e) => {
        message(message_element, "alert-danger", e.message)
    })
});