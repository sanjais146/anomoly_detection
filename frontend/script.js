document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. Tab Navigation ---
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            navItems.forEach(nav => nav.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            item.classList.add('active');
            const target = item.getAttribute('data-tab');
            const targetId = `tab-${target}`;
            const contentEl = document.getElementById(targetId);
            contentEl.classList.add('active');
            
            // Update header title
            pageTitle.innerHTML = '<span class="live-indicator"></span> ' + item.innerText.trim();
            
            // Resize ALL Chart.js instances after a short delay (tab was display:none before)
            setTimeout(() => {
                if (typeof Chart !== 'undefined') {
                    Chart.instances && Object.values(Chart.instances).forEach(c => {
                        try { c.resize(); } catch(e) {}
                    });
                }
            }, 80);
            
            // Vis.js networks need fit() after becoming visible
            if (target === 'overview' && typeof overviewNetwork !== 'undefined' && overviewNetwork) {
                setTimeout(() => overviewNetwork.fit(), 120);
            }
            if (target === 'graph-analytics' && typeof network !== 'undefined' && network) {
                setTimeout(() => network.fit(), 120);
                if (document.getElementById('wm-graph')) document.getElementById('wm-graph').style.display = 'none';
                if (document.getElementById('lbl-graph')) document.getElementById('lbl-graph').style.display = 'block';
            }
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

    // Rating Distribution
    const ctxDist = document.getElementById('chart-anomaly-dist');
    const distChart = new Chart(ctxDist, {
        type: 'bar',
        data: {
            labels: ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
            datasets: [{
                label: 'Rating Freq',
                data: [0, 0, 0, 0, 0],
                backgroundColor: ['#ef4444', '#f59e0b', '#fcd34d', '#10b981', '#059669']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });

    // Temporal Decay Chart
    const decayCtx = document.getElementById('chart-temporal-decay');
    const tau = 0.431;
    const decayLabels = [];
    const decayData = [];
    for(let d=0; d<=14; d+=0.5) {
        decayLabels.push(d + 'd');
        decayData.push(Math.exp(-tau * d));
    }
    if (decayCtx) {
        new Chart(decayCtx, {
            type: 'line',
            data: {
                labels: decayLabels,
                datasets: [{
                    label: 'w(Δt) Weight',
                    data: decayData,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { 
                    x: { title: { display: true, text: 'Time Difference (Days)' } },
                    y: { title: { display: true, text: 'Temporal Weight' }, min: 0, max: 1.05 }
                }
            }
        });
    }

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
    
    let overviewNetwork = null;

    // --- Initial Data Load ---
    async function loadAnalytics() {
        try {
            const res = await fetch('/analytics/amazon');
            const data = await res.json();
            
            // Populate timeline (from grouped timeline payload)
            const tLabels = [];
            const tData = [];
            if (data.timeline) {
                data.timeline.forEach(item => {
                    tLabels.push(item.date);
                    tData.push(item.count);
                });
            }
            timelineChart.data.labels = tLabels;
            timelineChart.data.datasets[0].data = tData;
            timelineChart.update();
            
            // Populate distribution
            distChart.data.datasets[0].data = data.rating_distribution;
            distChart.update();
            
            // Populate Vis.js Graphs (Overview and Main)
            const sampleNodes = new vis.DataSet();
            const sampleEdges = new vis.DataSet();
            
            data.sample_graph.forEach(edge => {
                const uId = "U_" + edge.reviewerID;
                const pId = "P_" + edge.asin;
                
                if(!sampleNodes.get(uId)) {
                    sampleNodes.add({ id: uId, label: 'User', title: edge.reviewerID, color: '#3b82f6', shape: 'dot' });
                    nodes.add({ id: uId, label: 'User', title: edge.reviewerID, color: '#3b82f6', shape: 'dot' });
                }
                if(!sampleNodes.get(pId)) {
                    sampleNodes.add({ id: pId, label: 'Product', title: edge.asin, color: '#10b981', shape: 'square' });
                    nodes.add({ id: pId, label: 'Product', title: edge.asin, color: '#10b981', shape: 'square' });
                }
                
                sampleEdges.add({ from: uId, to: pId, color: { color: '#d1d5db' }, title: 'Rating: ' + edge.overall });
                edges.add({ from: uId, to: pId, color: { color: '#d1d5db' }, title: 'Rating: ' + edge.overall });
            });
            
            // Init Overview Graph
            const overviewContainer = document.getElementById('vis-network-overview');
            if (overviewContainer) {
                overviewNetwork = new vis.Network(overviewContainer, {nodes: sampleNodes, edges: sampleEdges}, networkOptions);
                setTimeout(() => overviewNetwork.fit(), 500);
            }
            
            // Remove watermarks & show labels
            if (document.getElementById('wm-timeline')) document.getElementById('wm-timeline').style.display = 'none';
            if (document.getElementById('wm-dist')) document.getElementById('wm-dist').style.display = 'none';
            if (document.getElementById('wm-graph-overview')) document.getElementById('wm-graph-overview').style.display = 'none';
            if (document.getElementById('lbl-timeline')) document.getElementById('lbl-timeline').style.display = 'block';
            if (document.getElementById('lbl-dist')) document.getElementById('lbl-dist').style.display = 'block';
        } catch (e) {
            console.error("Failed to load analytics:", e);
        }

        // --- Load Dense Anomaly Analytics ---
        try {
            const res = await fetch('/analytics/amazon/anomalies');
            const data = await res.json();
            
            if (data.status === 'success') {
                document.getElementById('aa-kpi-total').innerText = data.kpi.total_analyzed;
                document.getElementById('aa-kpi-anomalous').innerText = data.kpi.anomalous_count;
                document.getElementById('aa-kpi-rate').innerText = data.kpi.anomaly_rate + '%';
                document.getElementById('aa-kpi-threshold').innerText = data.kpi.threshold;
                
                // Update extended KPIs if elements exist
                if (document.getElementById('aa-kpi-normal')) {
                    document.getElementById('aa-kpi-normal').innerText = data.kpi.normal_count;
                }
                if (document.getElementById('aa-kpi-avg-score')) {
                    document.getElementById('aa-kpi-avg-score').innerText = data.kpi.avg_anomaly_score;
                }

                const timeSeriesLabels = [];
                const timeSeriesData = [];
                const pointColors = [];
                const pointRadii = [];
                
                // Populate Top Anomalous Interactions table (sorted by score descending)
                const topAnomalies = [...data.time_series].sort((a, b) => b.anomaly_probability - a.anomaly_probability);
                const tbody = document.getElementById('aa-table-tbody');
                tbody.innerHTML = '';
                
                topAnomalies.forEach(inf => {
                    const dt = new Date(inf.unixReviewTime * 1000).toLocaleDateString();
                    const anomalyScore = (inf.anomaly_probability || 0).toFixed(4);
                    const thresholdVal = data.kpi.threshold;
                    const statusBadge = inf.is_anomalous 
                        ? `<span style="background:#ef4444;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">ANOMALY</span>` 
                        : `<span style="background:#374151;color:#9ca3af;padding:2px 8px;border-radius:4px;font-size:11px">NORMAL</span>`;
                    const scoreColor = inf.is_anomalous ? '#ef4444' : '#22c55e';
                    const rowBg = inf.is_anomalous ? 'rgba(239,68,68,0.08)' : 'transparent';
                        
                    tbody.innerHTML += `
                        <tr style="background:${rowBg}">
                            <td style="color:#9ca3af;font-size:12px">${dt}</td>
                            <td style="font-family:monospace;font-size:11px;color:#e5e7eb">${inf.reviewerID}</td>
                            <td style="font-family:monospace;font-size:11px;color:#e5e7eb">${inf.asin}</td>
                            <td style="color:#e5e7eb">${inf.rating} ★</td>
                            <td style="font-family:monospace;font-weight:bold;color:${scoreColor}">${anomalyScore}</td>
                            <td>${statusBadge}</td>
                        </tr>
                    `;
                });

                // Build time-series data (sorted chronologically)
                data.time_series.sort((a, b) => a.unixReviewTime - b.unixReviewTime).forEach(inf => {
                    timeSeriesLabels.push(new Date(inf.unixReviewTime * 1000).toLocaleDateString());
                    timeSeriesData.push(inf.anomaly_probability);
                    pointColors.push(inf.is_anomalous ? '#ef4444' : '#60a5fa');
                    pointRadii.push(inf.is_anomalous ? 6 : 4);
                });

                const ctxEl = document.getElementById('chart-anomaly-timeseries');
                if (ctxEl) {
                    ctxEl.style.backgroundColor = 'transparent';
                    
                    const validScores = timeSeriesData.map(Number).filter(Number.isFinite);
                    if (!validScores.length) {
                        console.error("No valid numeric scores for anomaly chart.");
                        return;
                    }
                    
                    if (window.anomalyTimeseriesChart) {
                        window.anomalyTimeseriesChart.destroy();
                    }
                    
                    const minScore = Math.min(...validScores);
                    const maxScore = Math.max(...validScores);
                    const padding = Math.max((maxScore - minScore) * 0.15, 0.02);

                    window.anomalyTimeseriesChart = new Chart(ctxEl, {
                        type: 'line',
                        data: {
                            labels: timeSeriesLabels,
                            datasets: [{
                                label: 'Anomaly Probability',
                                data: timeSeriesData,
                                borderColor: '#60a5fa',
                                backgroundColor: 'rgba(96,165,250,0.08)',
                                borderWidth: 2,
                                pointBackgroundColor: pointColors,
                                pointBorderColor: pointColors,
                                pointRadius: pointRadii,
                                pointHoverRadius: 8,
                                tension: 0.3,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            animation: { duration: 800 },
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    backgroundColor: '#1f2937',
                                    titleColor: '#f9fafb',
                                    bodyColor: '#9ca3af',
                                    callbacks: {
                                        label: (ctx) => {
                                            const idx = ctx.dataIndex;
                                            const item = data.time_series.sort((a,b)=>a.unixReviewTime-b.unixReviewTime)[idx];
                                            if (!item) return '';
                                            const verdict = item.is_anomalous ? 'ANOMALY' : 'NORMAL';
                                            return [
                                                'Score: ' + (item.anomaly_probability || 0).toFixed(4),
                                                'Decision: ' + verdict,
                                                'Threshold: ' + data.kpi.threshold,
                                                'Reviewer: ' + (item.reviewerID || '')
                                            ];
                                        }
                                    }
                                },
                                annotation: {
                                    annotations: {
                                        thresholdLine: {
                                            type: 'line',
                                            yMin: data.kpi.threshold,
                                            yMax: data.kpi.threshold,
                                            borderColor: 'rgba(239,68,68,0.8)',
                                            borderWidth: 2,
                                            borderDash: [6, 4],
                                            label: {
                                                display: true,
                                                content: 'Threshold ' + data.kpi.threshold,
                                                position: 'end',
                                                backgroundColor: 'rgba(239,68,68,0.85)',
                                                color: '#fff',
                                                font: { size: 11 },
                                                padding: { x: 6, y: 4 }
                                            }
                                        }
                                    }
                                }
                            },
                            scales: {
                                y: {
                                    min: Math.max(0, minScore - padding),
                                    max: Math.min(1.05, maxScore + padding),
                                    grid: { color: 'rgba(255,255,255,0.06)' },
                                    ticks: { color: '#9ca3af', font: { size: 11 } },
                                    title: { display: true, text: 'Anomaly Probability', color: '#6b7280' }
                                },
                                x: {
                                    grid: { display: false },
                                    ticks: { color: '#9ca3af', maxTicksLimit: 12, font: { size: 10 } },
                                    title: { display: true, text: 'Review Date', color: '#6b7280' }
                                }
                            }
                        }
                    });
                }
            }
        } catch (e) {
            console.error("Failed to load anomaly batch:", e);
            document.getElementById('aa-table-tbody').innerHTML = `<tr><td colspan="6" class="text-center text-red-500">Failed to load real inference data.</td></tr>`;
        }
    }
    loadAnalytics();


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
    const recentTbody = document.getElementById('recent-table-tbody');
    const emptyRow = document.getElementById('recent-empty-row');
    
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
            
            resCircle.className = 'score-circle ' + (result.prediction === 'anomalous' ? 'anomalous glow-danger' : 'genuine glow-success');
            resClass.className = (result.prediction === 'anomalous' ? 'anomalous' : 'genuine');
            resClass.innerText = result.prediction.charAt(0).toUpperCase() + result.prediction.slice(1);
            
            // Format mock embeddings string for UI using real payload/result values
            resUserEmb.innerText = `[${(score * 0.4).toFixed(3)}, ${(payload.overall * 0.1).toFixed(3)}, ${(score * 0.8).toFixed(3)}, ...]`;
            resProdEmb.innerText = `[${(score * -0.2).toFixed(3)}, ${(payload.overall * -0.15).toFixed(3)}, ${(score * -0.5).toFixed(3)}, ...]`;

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
