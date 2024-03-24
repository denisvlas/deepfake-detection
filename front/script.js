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
                console.log(data);9
                // După ce primiți răspunsul de la server, actualizați mesajul de așteptare cu rezultatul
                waitingMessage.innerHTML = data.percentage.toFixed(2) + "<br>" + data.final_prediction})
                // waitingMessage.innerHTML = "Real Percentage: " + data.real_percentage.toFixed(2) + "<br>" + 
                // "Fake percentage:" + data.fake_percentage.toFixed(2) + "<br>" + 
                // "Deepfake prediction: " + data.final_prediction;})
                .catch((error) => {
                // În caz de eroare, actualizați mesajul de așteptare cu mesajul de eroare
                waitingMessage.textContent = error;
            });
        });
    });
});
