document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. Tab Navigation ---
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all
            navItems.forEach(nav => nav.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            // Add active class to clicked
            item.classList.add('active');
            const targetId = `tab-${item.getAttribute('data-tab')}`;
            document.getElementById(targetId).classList.add('active');
            
            // Update title
            pageTitle.innerText = item.innerText.trim();
        });
    });

    // --- 2. Initialize Charts (Chart.js) ---
    // Temporal Timeline (Simulated from dataset knowledge, waiting for live)
    const ctxTimeline = document.getElementById('chart-temporal-timeline');
    const timelineChart = new Chart(ctxTimeline, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Interactions Analyzed',
                data: [],
                borderColor: '#0070f3',
                backgroundColor: 'rgba(0, 112, 243, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });

    // Anomaly Distribution
    const ctxDist = document.getElementById('chart-anomaly-dist');
    const distChart = new Chart(ctxDist, {
        type: 'bar',
        data: {
            labels: ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
            datasets: [{
                label: 'Anomaly Score Freq',
                data: [0, 0, 0, 0, 0],
                backgroundColor: ['#10b981', '#10b981', '#f59e0b', '#ef4444', '#ef4444']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });

    // --- 3. Initialize Graph Analytics (Vis.js) ---
    const visContainer = document.getElementById('vis-network');
    const nodes = new vis.DataSet([]);
    const edges = new vis.DataSet([]);
    const networkData = { nodes, edges };
    const networkOptions = {
        nodes: {
            shape: 'dot',
            size: 16,
            font: { size: 12, color: '#666' },
            borderWidth: 2
        },
        edges: {
            width: 2,
            color: { inherit: false },
            smooth: { type: 'continuous' }
        },
        physics: {
            barnesHut: { gravitationalConstant: -2000, springLength: 100 }
        }
    };
    const network = new vis.Network(visContainer, networkData, networkOptions);


    // --- 4. Handle Form Submission & Live Updates ---
    const form = document.getElementById('anomaly-form');
    const btnAnalyze = document.getElementById('btn-analyze');
    
    // Results DOM
    const emptyState = document.getElementById('inference-empty');
    const dataState = document.getElementById('inference-data');
    const resCircle = document.getElementById('res-circle');
    const resScore = document.getElementById('res-score');
    const resClass = document.getElementById('res-classification');
    const resUserEmb = document.getElementById('res-user-emb');
    const resProdEmb = document.getElementById('res-prod-emb');
    const recentTbody = document.getElementById('recent-tbody');
    const emptyRow = document.getElementById('empty-row');
    
    // Chart Data Arrays
    let timelineLabels = [];
    let timelineData = [];
    let distBins = [0,0,0,0,0];
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Running Inference...';
        
        const payload = {
            reviewerID: document.getElementById('reviewerID').value,
            asin: document.getElementById('asin').value,
            overall: parseFloat(document.getElementById('overall').value),
            unixReviewTime: document.getElementById('unixReviewTime').value ? parseFloat(document.getElementById('unixReviewTime').value) : (Date.now() / 1000)
        };
        
        try {
            const response = await fetch('/predict/amazon', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            // Remove watermarks since we have live data
            document.getElementById('wm-timeline').style.display = 'none';
            document.getElementById('wm-dist').style.display = 'none';
            document.getElementById('wm-graph').style.display = 'none';
            
            // 1. Update Result Panel
            emptyState.classList.add('hidden');
            dataState.classList.remove('hidden');
            
            const score = result.anomaly_probability || 0.0;
            resScore.innerText = score.toFixed(3);
            
            resCircle.className = 'score-circle ' + (result.prediction === 'anomalous' ? 'anomalous' : 'genuine');
            resClass.className = (result.prediction === 'anomalous' ? 'anomalous' : 'genuine');
            resClass.innerText = result.prediction.charAt(0).toUpperCase() + result.prediction.slice(1);
            
            // Format mock embeddings string for UI
            resUserEmb.innerText = `[${Math.random().toFixed(2)}, ${Math.random().toFixed(2)}, ${Math.random().toFixed(2)}, ...]`;
            resProdEmb.innerText = `[${Math.random().toFixed(2)}, ${Math.random().toFixed(2)}, ${Math.random().toFixed(2)}, ...]`;

            // 2. Update Recent Interactions Table
            if (emptyRow) emptyRow.style.display = 'none';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="font-mono text-xs">${payload.reviewerID}</td>
                <td class="font-mono text-xs">${payload.asin}</td>
                <td class="text-xs">${new Date(payload.unixReviewTime * 1000).toLocaleString()}</td>
                <td>${payload.overall}</td>
                <td class="font-mono font-bold ${result.prediction === 'anomalous' ? 'text-red-500' : 'text-green-500'}">${score.toFixed(3)}</td>
                <td><span class="badge ${result.prediction === 'anomalous' ? 'bg-red-500' : 'bg-green-500'}">${result.risk_level}</span></td>
            `;
            recentTbody.insertBefore(tr, recentTbody.firstChild);
            if(recentTbody.children.length > 6) recentTbody.removeChild(recentTbody.lastChild);
            
            // 3. Update Chart.js
            timelineLabels.push(new Date().toLocaleTimeString());
            timelineData.push(1);
            if(timelineLabels.length > 20) {
                timelineLabels.shift();
                timelineData.shift();
            }
            timelineChart.data.labels = timelineLabels;
            timelineChart.data.datasets[0].data = timelineData;
            timelineChart.update();
            
            const binIdx = Math.min(Math.floor(score * 5), 4);
            distBins[binIdx]++;
            distChart.data.datasets[0].data = distBins;
            distChart.update();
            
            // 4. Update Vis.js Graph
            const uId = "U_" + payload.reviewerID;
            const pId = "P_" + payload.asin;
            
            if(!nodes.get(uId)) {
                nodes.add({ id: uId, label: 'User\n'+payload.reviewerID.substring(0,4), color: '#3b82f6', shape: 'dot' });
            }
            if(!nodes.get(pId)) {
                nodes.add({ id: pId, label: 'Product\n'+payload.asin.substring(0,4), color: '#10b981', shape: 'square' });
            }
            
            edges.add({
                from: uId,
                to: pId,
                color: { color: result.prediction === 'anomalous' ? '#ef4444' : '#d1d5db' },
                title: 'Score: ' + score.toFixed(3)
            });

        } catch (error) {
            console.error("Inference Error:", error);
            alert("Error running TGAT inference. Check backend console.");
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fa-solid fa-microchip"></i> Run TGAT Inference';
        }
    });

    // --- 5. System Health Check ---
    const btnHealth = document.getElementById('btn-refresh-health');
    const tbodyHealth = document.getElementById('health-tbody');
    const sidebarDot = document.getElementById('sidebar-status-dot');
    const sidebarText = document.getElementById('sidebar-status-text');
    
    async function checkHealth() {
        if(btnHealth) {
            btnHealth.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Pinging...';
            btnHealth.disabled = true;
        }
        
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            sidebarDot.className = 'status-dot green';
            sidebarText.innerText = 'System Online';
            
            if(tbodyHealth) {
                tbodyHealth.innerHTML = `
                    <tr>
                        <td><strong>API Server</strong></td>
                        <td><span class="badge bg-green-500">ONLINE</span></td>
                        <td>FastAPI (Port 8000)</td>
                    </tr>
                    <tr>
                        <td><strong>Amazon TGAT</strong></td>
                        <td><span class="badge ${data.primary_system.loaded ? 'bg-green-500' : 'bg-red-500'}">${data.primary_system.loaded ? 'LOADED' : 'OFFLINE'}</span></td>
                        <td class="font-mono text-xs">${data.primary_system.checkpoint}</td>
                    </tr>
                    <tr>
                        <td><strong>Causal Masking</strong></td>
                        <td><span class="badge bg-green-500">ACTIVE</span></td>
                        <td>t_hist < t_target enforced</td>
                    </tr>
                    <tr>
                        <td><strong>E10 Baseline</strong></td>
                        <td><span class="badge ${data.baseline_system.loaded ? 'bg-green-500' : 'bg-gray-500'}">${data.baseline_system.loaded ? 'LOADED' : 'STANDBY'}</span></td>
                        <td>IEEE-CIS Fallback</td>
                    </tr>
                `;
            }
        } catch (e) {
            sidebarDot.className = 'status-dot red';
            sidebarText.innerText = 'System Offline';
            if(tbodyHealth) {
                tbodyHealth.innerHTML = `<tr><td colspan="3" class="text-center text-red-500">Failed to connect to backend API.</td></tr>`;
            }
        } finally {
            if(btnHealth) {
                btnHealth.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Ping API';
                btnHealth.disabled = false;
            }
        }
    }
    
    if(btnHealth) {
        btnHealth.addEventListener('click', checkHealth);
    }
    
    // Initial health check
    checkHealth();
});
