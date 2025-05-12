function startDictation() {
    if (window.hasOwnProperty('webkitSpeechRecognition')) {
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        recognition.start();

        recognition.onresult = function(e) {
            document.getElementById('text_input').value = e.results[0][0].transcript;
            document.getElementById('speech_output').innerText = 'Captured: ' + e.results[0][0].transcript;
            recognition.stop();
        };

        recognition.onerror = function(e) {
            recognition.stop();
            alert('Error occurred in recognition: ' + e.error);
        };
    } else {
        alert('Speech recognition not supported in this browser.');
    }
}
