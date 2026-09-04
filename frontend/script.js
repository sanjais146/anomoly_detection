document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('run-btn');
    btn.textContent = 'Processing...';
    btn.disabled = true;

    const payload = {
        TransactionAmt: parseFloat(document.getElementById('TransactionAmt').value),
        ProductCD: document.getElementById('ProductCD').value,
        card1: parseInt(document.getElementById('card1').value),
        P_emaildomain: document.getElementById('P_emaildomain').value || null,
        R_emaildomain: document.getElementById('R_emaildomain').value || null
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        
        document.getElementById('result-container').classList.remove('hidden');
        
        const probPct = (data.fraud_probability * 100).toFixed(1);
        const circle = document.getElementById('prob-circle');
        document.getElementById('prob-value').textContent = probPct + '%';
        
        const isFraud = data.prediction === 'fraud';
        circle.style.borderColor = isFraud ? '#ef4444' : '#10b981';
        
        document.getElementById('res-decision').textContent = isFraud ? 'HIGH RISK — POTENTIAL FRAUD' : 'Legitimate';
        document.getElementById('res-decision').style.color = isFraud ? '#ef4444' : '#10b981';
        
        const riskEl = document.getElementById('res-risk');
        riskEl.textContent = data.risk_level;
        riskEl.className = 'value badge ' + data.risk_level.toLowerCase();
        
        const list = document.getElementById('signal-list');
        list.innerHTML = '';
        data.signals.forEach(s => {
            let li = document.createElement('li');
            li.textContent = s;
            list.appendChild(li);
        });

    } catch (err) {
        alert("Error making prediction request.");
    } finally {
        btn.textContent = 'Run Prediction';
        btn.disabled = false;
    }
});
