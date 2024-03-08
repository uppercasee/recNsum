const form = document.getElementById('summaryForm');
const summaryInput = document.getElementById('summaryInput');
const summary = document.getElementById('summary');
const recform = document.getElementById('recommendationForm');
const userId = document.getElementById('userId');
const recommendation = document.getElementById('recommendation');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    fetch('/summarize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: summaryInput.value })
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            summary.innerHTML = data.summary;
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

recform.addEventListener('submit', (e) => {
    e.preventDefault();
    // console.log(userId.value);
    fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({userid: userId.value})
    })
        .then(response => response.json())
        .then(data=> {
            userId.value = ''
            // console.log(data)
            const parsedData = JSON.parse(data.recommendations);
            // console.log(parsedData)
            recommendation.innerHTML = '';
            parsedData.forEach(item => {
                const div = document.createElement('div');
                div.classList.add('recommendation-item');
                div.innerHTML = `
                <h3 class="recommendation-item-id">News ID: ${item.itemId}</h3>
                <h4 class="recommendation-item-title">${item.title}</h4>
                <p class="recommendation-item-abstract">${item.abstract}</p>
                `;
                recommendation.appendChild(div);
            });
        })
        .catch(error=> {
            console.error('Error:', error);
        });
});

