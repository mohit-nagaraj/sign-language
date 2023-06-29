fromText = document.querySelector(".from-text"),
toText = document.querySelector("#translatedbro"),
exchageIcon = document.querySelector(".exchange"),
selectTag = document.querySelectorAll("select"),
icons = document.querySelectorAll(".row i");
translateBtn = document.querySelector("#clickbro"),

selectTag.forEach((tag, id) => {
    for (let country_code in countries) {
        let selected = id == 0 ? country_code == "en-GB" ? "selected" : "" : country_code == "hi-IN" ? "selected" : "";
        let option = `<option ${selected} value="${country_code}">${countries[country_code]}</option>`;
        tag.insertAdjacentHTML("beforeend", option);
    }
});

exchageIcon.addEventListener("click", () => {
    let tempText = fromText.value,
    tempLang = selectTag[0].value;
    fromText.value = toText.value;
    toText.value = tempText;
    selectTag[0].value = selectTag[1].value;
    selectTag[1].value = tempLang;
});

fromText.addEventListener("keyup", () => {
    if(!fromText.value) {
        toText.value = "";
    }
});

  


translateBtn.addEventListener("click", () => {
    const text = fromText.value.trim();
    translateFrom = selectTag[0].value;
    translateTo = selectTag[1].value;
    if(!text) return;
    toText.setAttribute("placeholder", "Translating...");
    var apiUrl = 'https://google-translate1.p.rapidapi.com/language/translate/v2';
    const options = {
        method: 'POST', 
        headers: {
            'content-type': 'application/x-www-form-urlencoded',
            'Accept-Encoding': 'application/gzip',
            'X-RapidAPI-Key': 'b1fea9e3e8msh662920884bee0c3p10bde2jsnb788be0810fc',
            'X-RapidAPI-Host': 'google-translate1.p.rapidapi.com'
        },
        body: new URLSearchParams({
            q: text,
            target: translateTo,
            source: translateFrom
        })
    };      
        try {
          fetch(apiUrl, options)
            .then(res => res.json())
            .then(data => {
              toText.value = data.data.translations[0].translatedText;
            })
            .catch(error => {
              toText.value = "Translation failed "+error;
            });
        } catch (error) {
          toText.value = "Translation failed1.";
        }
        //toText.setAttribute("placeholder", "Translation");
      });
      
    

icons.forEach(icon => {
    icon.addEventListener("click", ({target}) => {
        if(!fromText.value || !toText.value) return;
        if(target.classList.contains("fa-copy")) {
            if(target.id == "fromm") {
                navigator.clipboard.writeText(fromText.value);
            }if(target.id == "too") {
                navigator.clipboard.writeText(toText.value);
            }
        } else {
            let utterance;
            if(target.id == "from") {
                utterance = new SpeechSynthesisUtterance(fromText.value);
                utterance.lang = selectTag[0].value;
            } else {
                utterance = new SpeechSynthesisUtterance(toText.value);
                utterance.lang = selectTag[1].value;
            }
            speechSynthesis.speak(utterance);
        }
    });
});