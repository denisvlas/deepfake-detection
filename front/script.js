document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("verifyButton");
  
    btn.addEventListener("click", function () {
        var waitingMessage = document.createElement("h1");
        waitingMessage.textContent = "Așteptare...";
        document.getElementsByClassName("button-wrapper")[0].appendChild(waitingMessage); // Afișați mesajul de așteptare
        document.getElementsByClassName("button-wrapper")[0].removeChild(btn); // Afișați mesajul de așteptare

        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            if (tabs.length === 0) {
                console.log("No active tab found");
                return; // Exit the function if no tabs
            }
            var url = tabs[0].url;

            fetch("http://localhost:5000/predict", {
                method: "POST",
                body: JSON.stringify({ video_url: url }),
                headers: {
                    "Content-Type": "application/json",
                },
            })
            .then((response) => response.json())
            .then((data) => {
                console.log(data);
                // După ce primiți răspunsul de la server, actualizați mesajul de așteptare cu rezultatul
                waitingMessage.textContent = "Deepfake prediction: " + data.average_percentage.toFixed(2) + "%";
            })
            .catch((error) => {
                // În caz de eroare, actualizați mesajul de așteptare cu mesajul de eroare
                waitingMessage.textContent = error;
            });
        });
    });
});
