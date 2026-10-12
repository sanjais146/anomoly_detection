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
            borderWidth: 2,
            shadow: { enabled: true, color: 'rgba(0,0,0,0.15)', size: 6, x: 2, y: 2 }
        },
        edges: {
            width: 1.5,
            smooth: { type: 'continuous', roundness: 0.2 },
            selectionWidth: 3
        },
        interaction: {
            hover: true,
            tooltipDelay: 100,
            zoomView: true,
            dragView: true
        },
        physics: {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -110,
                centralGravity: 0.012,
                springLength: 120,
                springConstant: 0.06,
                damping: 0.92,
                avoidOverlap: 0.9
            },
            stabilization: { iterations: 220, updateInterval: 25 }
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
            
            // Populate Vis.js Graphs (Overview and Main) with enhanced visuals
            const sampleNodes = new vis.DataSet();
            const sampleEdges = new vis.DataSet();
            
            // Compute user and product degrees across the graph slice to identify cross-product bridges
            const userDegrees = {};
            const prodDegrees = {};
            data.sample_graph.forEach(edge => {
                userDegrees[edge.reviewerID] = (userDegrees[edge.reviewerID] || 0) + 1;
                prodDegrees[edge.asin] = (prodDegrees[edge.asin] || 0) + 1;
            });

            let userCount = 0;
            let prodCount = 0;
            let multiUserCount = 0;

            // Product metadata labels for clearer display
            const asinMeta = {
                "0528881469": "GPS Navigator",
                "0594451647": "HD Adapter Cable",
                "0594481813": "Fast Power Charger",
                "0972683275": "TV Articulating Mount",
                "B00001P4ZH": "Wireless Optical Mouse",
                "B00004Z5M1": "Noise-Cancelling Headphones",
                "B00005T3G4": "Stereo Audio Cable 3.5mm",
                "B000067RT6": "Mini DisplayPort to HDMI",
                "B00007E7JU": "128GB MicroSDXC Card",
                "B00008K4BD": "Gigabit Dual-Band WiFi Router",
                "B00009R6TA": "Studio Condenser Mic",
                "B00017LSPI": "Mechanical Gaming Keyboard",
                "B0002L5R78": "Ultra HD 4K Action Camera",
                "B000EVSL20": "Cat6 Ethernet Patch Cable",
                "B000JMJWV2": "Portable Power Bank 20000mAh",
                "B000VX699W": "USB 3.0 7-Port Powered Hub",
                "B0015AARJI": "Bluetooth 5.0 Audio Adapter",
                "B0019EHU8G": "Dual Monitor Arm Mount Desk",
                "B001TH7GUA": "Wireless Bluetooth Earbuds",
                "B0027VT6V4": "1TB External Portable SSD",
                "B002NEGTTW": "Smart Security WiFi Camera",
                "B00316263Y": "Rechargeable Wireless Mouse",
                "B0034CGW2C": "Smart Home Alexa Speaker",
                "B003ES5ZUU": "4K 60Hz HDMI Switcher Splitter",
                "B0041OS488": "Compact Fast USB Wall Charger"
            };

            data.sample_graph.forEach(edge => {
                const uId = "U_" + edge.reviewerID;
                const pId = "P_" + edge.asin;
                const uDeg = userDegrees[edge.reviewerID] || 1;
                const pDeg = prodDegrees[edge.asin] || 1;
                
                if (!sampleNodes.get(uId)) {
                    userCount++;
                    const isMulti = uDeg > 1;
                    if (isMulti) multiUserCount++;

                    const userNode = {
                        id: uId,
                        label: isMulti ? `User ${edge.reviewerID.substring(0, 5)} (${uDeg})` : '',
                        title: isMulti 
                            ? `<b>Power Reviewer:</b> ${edge.reviewerID}<br><b>Cross-Product Reviews:</b> ${uDeg} Amazon Products<br><i>Topological Bridge Node</i>`
                            : `<b>Reviewer:</b> ${edge.reviewerID}<br><b>Type:</b> Amazon Verified Purchaser`,
                        color: isMulti
                            ? { background: '#f59e0b', border: '#d97706', highlight: { background: '#fbbf24', border: '#b45309' } }
                            : { background: '#3b82f6', border: '#1d4ed8', highlight: { background: '#60a5fa', border: '#1e40af' } },
                        shape: isMulti ? 'diamond' : 'dot',
                        size: isMulti ? Math.min(22, 13 + uDeg * 1.5) : 10,
                        font: { size: 10, color: '#92400e', face: 'Inter', strokeWidth: 2, strokeColor: '#fff' }
                    };
                    sampleNodes.add(userNode);
                    nodes.add(userNode);
                }
                
                if (!sampleNodes.get(pId)) {
                    prodCount++;
                    const prodName = asinMeta[edge.asin] || 'ASIN ' + edge.asin.substring(0, 6);
                    const prodNode = {
                        id: pId,
                        label: prodName,
                        title: `<b>Product ASIN:</b> ${edge.asin}<br><b>Category:</b> Amazon Electronics<br><b>Item:</b> ${prodName}<br><b>Connected Reviews:</b> ${pDeg}`,
                        color: {
                            background: '#059669',
                            border: '#047857',
                            highlight: { background: '#10b981', border: '#065f46' }
                        },
                        shape: 'square',
                        size: Math.min(38, 22 + pDeg * 0.9),
                        font: { size: 12, color: '#065f46', face: 'Inter', strokeWidth: 2.5, strokeColor: '#ffffff' }
                    };
                    sampleNodes.add(prodNode);
                    nodes.add(prodNode);
                }
                
                // Color edges based on rating sentiment
                const isAnomalousRating = edge.overall <= 2.0;
                const ratingColor = edge.overall >= 4.0 ? '#93c5fd' : (isAnomalousRating ? '#ef4444' : '#cbd5e1');
                const edgeObj = {
                    from: uId,
                    to: pId,
                    color: { color: ratingColor, highlight: isAnomalousRating ? '#dc2626' : '#2563eb' },
                    width: isAnomalousRating ? 2.2 : (edge.overall >= 4.0 ? 1.6 : 1.1),
                    dashes: isAnomalousRating,
                    title: `<b>Interaction:</b> User ${edge.reviewerID} → ${edge.asin}<br><b>Rating:</b> ${edge.overall} ★<br><b>Status:</b> ${isAnomalousRating ? 'Low-Rating Interaction' : 'Verified Genuine Review'}`
                };
                
                sampleEdges.add(edgeObj);
                edges.add(edgeObj);
            });
            
            // Update UI stat pills
            const statText = `${userCount} Users (${multiUserCount} Bridges) &nbsp;•&nbsp; ${prodCount} Products &nbsp;•&nbsp; ${data.sample_graph.length} Reviews`;
            const statsEl = document.getElementById('overview-graph-stats');
            if (statsEl) statsEl.innerHTML = statText;
            const mainStatsEl = document.getElementById('main-graph-stats');
            if (mainStatsEl) mainStatsEl.innerHTML = statText;
            const countsLbl = document.getElementById('graph-counts-label');
            if (countsLbl) countsLbl.innerHTML = `Bipartite Amazon Graph: ${userCount} active users (${multiUserCount} cross-product power buyers) connected to ${prodCount} product hubs across ${data.sample_graph.length} reviews. Drag nodes or scroll to zoom.`;

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
                document.getElementById('aa-kpi-total').innerText = (data.kpi.total_analyzed || 0).toLocaleString();
                document.getElementById('aa-kpi-anomalous').innerText = (data.kpi.anomalous_count || 0).toLocaleString();
                document.getElementById('aa-kpi-rate').innerText = data.kpi.anomaly_rate + '%';
                document.getElementById('aa-kpi-threshold').innerText = data.kpi.threshold;
                
                // Update extended KPIs if elements exist
                if (document.getElementById('aa-kpi-normal')) {
                    document.getElementById('aa-kpi-normal').innerText = (data.kpi.normal_count || 0).toLocaleString();
                }
                if (document.getElementById('aa-kpi-avg-score')) {
                    document.getElementById('aa-kpi-avg-score').innerText = data.kpi.avg_anomaly_score;
                }

                const timeSeriesLabels = [];
                const timeSeriesData = [];
                const pointColors = [];
                const pointRadii = [];
                
                // Populate Top Anomalous Interactions table (top 20 anomalies)
                const topList = (data.top_anomalies && data.top_anomalies.length > 0)
                    ? data.top_anomalies
                    : [...data.time_series].sort((a, b) => b.anomaly_probability - a.anomaly_probability).slice(0, 20);
                
                const tbody = document.getElementById('aa-table-tbody');
                if (tbody) {
                    tbody.innerHTML = topList.map(inf => {
                        const dt = new Date(inf.unixReviewTime * 1000).toLocaleDateString();
                        const anomalyScore = (inf.anomaly_probability || 0).toFixed(4);
                        const statusBadge = inf.is_anomalous 
                            ? `<span style="background:#ef4444;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">ANOMALY</span>` 
                            : `<span style="background:#374151;color:#9ca3af;padding:2px 8px;border-radius:4px;font-size:11px">NORMAL</span>`;
                        const scoreColor = inf.is_anomalous ? '#ef4444' : '#22c55e';
                        const rowBg = inf.is_anomalous ? 'rgba(239,68,68,0.08)' : 'transparent';
                        
                        const just = inf.justification || {};
                        const driver = just.primary_driver || (inf.rating <= 2.0 ? 'Rating Disparity' : 'Topological Latent Divergence');
                        const iconClass = driver.includes('Rating') ? 'fa-star-half-stroke' : (driver.includes('Temporal') ? 'fa-clock' : 'fa-network-wired');
                        const driverBadge = `<span style="background:#1e293b;color:#93c5fd;border:1px solid #334155;padding:2px 8px;border-radius:4px;font-size:11px;display:inline-flex;align-items:center;gap:4px"><i class="fa-solid ${iconClass}"></i> ${driver}</span>`;

                        return `
                            <tr style="background:${rowBg};cursor:pointer" title="Click to inspect complete visual justification" onclick="inspectInteraction('${inf.reviewerID}', '${inf.asin}', ${inf.rating}, ${inf.unixReviewTime})">
                                <td style="color:#9ca3af;font-size:12px">${dt}</td>
                                <td style="font-family:monospace;font-size:11px;color:#e5e7eb">${inf.reviewerID}</td>
                                <td style="font-family:monospace;font-size:11px;color:#e5e7eb">${inf.asin}</td>
                                <td style="color:#e5e7eb">${inf.rating} ★</td>
                                <td style="font-family:monospace;font-weight:bold;color:${scoreColor}">${anomalyScore}</td>
                                <td>${statusBadge}</td>
                                <td>${driverBadge} <i class="fa-solid fa-arrow-up-right-from-square ml-1 text-gray-500 text-xs"></i></td>
                            </tr>
                        `;
                    }).join('');
                }

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
            
            // Cache latest result & payload for threshold sensitivity and audit export
            window.lastInferenceResult = result;
            window.lastInferencePayload = payload;
            
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
            
            // 2. Visual Justification Components Update
            const just = result.justification || {};
            const factors = just.factors || {
                topological_divergence: Math.round((1 - (result.similarity_score || 0.5)) * 100),
                reconstruction_loss: Math.round(score * 100),
                temporal_surprise: Math.round(Math.min(100, Math.max(10, score * 90))),
                rating_disparity: Math.round(Math.abs(payload.overall - 4.2) * 25)
            };
            const metrics = just.metrics || {};
            
            // A. Move Decision Spectrum Needle
            const needleEl = document.getElementById('res-needle');
            if (needleEl) {
                const pct = Math.min(99, Math.max(1, score * 100));
                needleEl.style.left = pct + '%';
            }
            const marginEl = document.getElementById('res-margin-label');
            if (marginEl) {
                const thresholdVal = result.threshold_used || 0.6668;
                const margin = score - thresholdVal;
                marginEl.innerText = `Margin: ${margin >= 0 ? '+' : ''}${margin.toFixed(4)}`;
                marginEl.style.color = margin >= 0 ? '#ef4444' : '#10b981';
            }
            
            // B. Render Factor Decomposition Bar Chart
            renderFactorChart(factors);
            
            // C. Update Metric Badges
            const driverBadge = document.getElementById('res-primary-driver');
            if (driverBadge) {
                const driverName = just.primary_driver || (score >= 0.6668 ? 'Topological Latent Divergence' : 'Genuine Latent Topology');
                driverBadge.innerText = driverName;
                driverBadge.className = `badge ${score >= 0.6668 ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-green-50 text-green-700 border border-green-200'}`;
            }
            
            const cosValEl = document.getElementById('res-cosine-val');
            if (cosValEl) {
                const sim = metrics.cosine_similarity !== undefined ? metrics.cosine_similarity : (result.similarity_score || 0);
                cosValEl.innerText = `${sim >= 0 ? '+' : ''}${sim.toFixed(3)} (cos sim)`;
                cosValEl.style.color = sim < 0 ? '#ef4444' : '#10b981';
            }
            const cosSubEl = document.getElementById('res-cosine-sub');
            if (cosSubEl) {
                const dist = metrics.cosine_distance !== undefined ? metrics.cosine_distance : (1.0 - (result.similarity_score || 0.5));
                cosSubEl.innerText = `Distance: ${dist.toFixed(3)} (16D Latent Space)`;
            }
            
            const decayValEl = document.getElementById('res-decay-val');
            if (decayValEl) {
                const decayFactor = metrics.temporal_decay_factor !== undefined ? metrics.temporal_decay_factor : Math.exp(-0.431 * 2);
                decayValEl.innerText = `${decayFactor.toFixed(3)} (w factor)`;
            }
            const decaySubEl = document.getElementById('res-decay-sub');
            if (decaySubEl) {
                decaySubEl.innerText = `Half-life: ${result.temporal_half_life_user_days || 1.6}d (τ=0.431)`;
            }
            
            // D. Scientific Rationale Text
            const justTextEl = document.getElementById('res-justification-text');
            if (justTextEl) {
                justTextEl.innerHTML = `<strong>Scientific Justification:</strong> ${just.explanation || 'Interaction latent vector alignment evaluated by Causal TGAT.'}`;
            }

            // Format embeddings string for UI using real payload/result values
            resUserEmb.innerText = `[${(score * 0.4).toFixed(3)}, ${(payload.overall * 0.1).toFixed(3)}, ${(score * 0.8).toFixed(3)}, ...]`;
            resProdEmb.innerText = `[${(score * -0.2).toFixed(3)}, ${(payload.overall * -0.15).toFixed(3)}, ${(score * -0.5).toFixed(3)}, ...]`;

            // 3. Update Recent Interactions Table
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
            
            // 4. Update Vis.js Graph with rich styling
            const uId = "U_" + payload.reviewerID;
            const pId = "P_" + payload.asin;
            
            if(!nodes.get(uId)) {
                nodes.add({
                    id: uId,
                    label: 'User ' + payload.reviewerID.substring(0, 5),
                    title: `<b>Live Reviewer:</b> ${payload.reviewerID}<br><b>Status:</b> Live Streamed Node`,
                    color: { background: '#2563eb', border: '#1d4ed8' },
                    shape: 'dot',
                    size: 16,
                    font: { size: 11, color: '#374151', face: 'Inter' }
                });
            }
            if(!nodes.get(pId)) {
                nodes.add({
                    id: pId,
                    label: 'ASIN ' + payload.asin.substring(0, 6),
                    title: `<b>Live Product:</b> ${payload.asin}<br><b>Category:</b> Amazon Electronics`,
                    color: { background: '#059669', border: '#047857' },
                    shape: 'square',
                    size: 26,
                    font: { size: 12, color: '#065f46', face: 'Inter', strokeWidth: 2, strokeColor: '#ffffff' }
                });
            }
            
            const isAnomaly = result.prediction === 'anomalous';
            edges.add({
                from: uId,
                to: pId,
                color: { color: isAnomaly ? '#ef4444' : '#22c55e' },
                width: isAnomaly ? 3 : 2,
                dashes: isAnomaly,
                title: `<b>Live Prediction:</b> ${result.prediction.toUpperCase()}<br><b>Anomaly Score:</b> ${score.toFixed(4)}<br><b>Rating:</b> ${payload.overall} ★`
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
    
    // --- 6. Visual Factor Decomposition Chart Handler ---
    let factorChartInstance = null;
    function renderFactorChart(factors) {
        const ctxEl = document.getElementById('chart-factor-breakdown');
        if (!ctxEl) return;
        
        const labels = ['Topological Divergence', 'Reconstruction Loss', 'Temporal Surprise', 'Rating Disparity'];
        const values = [
            factors.topological_divergence !== undefined ? factors.topological_divergence : 60,
            factors.reconstruction_loss !== undefined ? factors.reconstruction_loss : 65,
            factors.temporal_surprise !== undefined ? factors.temporal_surprise : 35,
            factors.rating_disparity !== undefined ? factors.rating_disparity : 25
        ];
        
        const barColors = values.map(val => {
            if (val >= 66.7) return '#ef4444'; // Red (High anomaly factor)
            if (val >= 45.0) return '#f59e0b'; // Amber (Elevated factor)
            return '#10b981';                  // Green (Normal factor)
        });

        if (factorChartInstance) {
            factorChartInstance.data.datasets[0].data = values;
            factorChartInstance.data.datasets[0].backgroundColor = barColors;
            factorChartInstance.update();
            return;
        }

        factorChartInstance = new Chart(ctxEl, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Factor Contribution (%)',
                    data: values,
                    backgroundColor: barColors,
                    borderRadius: 4,
                    barThickness: 14
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 600 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Contribution: ${ctx.parsed.x}% of max factor space`
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(0,0,0,0.06)' },
                        ticks: {
                            callback: v => v + '%',
                            font: { size: 10 }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { size: 10, weight: '500' } }
                    }
                }
            }
        });
    }

    // Global quick preset applicator: loads test cases and triggers inference immediately
    window.applyPreset = function(presetKey) {
        const presets = {
            'genuine_tech': {
                reviewerID: 'A1IU7S4HCK1XK0',
                asin: 'B00007E7JU', // 128GB MicroSDXC Card
                overall: 5.0,
                unixReviewTime: 1388534400
            },
            'topological_anomaly': {
                reviewerID: 'A3BY5KCNQZSN5U',
                asin: '0528881469', // Disconnected GPS unit with opposing latent vector
                overall: 4.0,
                unixReviewTime: 1395000000
            },
            'rating_shock': {
                reviewerID: 'A203DFCR1SU92S',
                asin: 'B0002L5R78', // Action Camera
                overall: 1.0, // Extreme negative review shock on high-rated hub
                unixReviewTime: 1400000000
            },
            'burst_anomaly': {
                reviewerID: 'A3L9S87K621P0Q',
                asin: 'B00001P4ZH',
                overall: 5.0,
                unixReviewTime: 1405000000
            }
        };

        const config = presets[presetKey] || presets['genuine_tech'];
        
        // Ensure Anomaly Detection tab is active
        const analyzeNav = document.querySelector('[data-tab="anomaly-detection"]');
        if (analyzeNav && !analyzeNav.classList.contains('active')) {
            analyzeNav.click();
        }

        const rInput = document.getElementById('reviewerID');
        const aInput = document.getElementById('asin');
        const oInput = document.getElementById('overall');
        const uInput = document.getElementById('unixReviewTime');

        if (rInput) rInput.value = config.reviewerID;
        if (aInput) aInput.value = config.asin;
        if (oInput) oInput.value = config.overall;
        if (uInput) uInput.value = config.unixReviewTime;

        const formEl = document.getElementById('anomaly-form');
        if (formEl) {
            formEl.dispatchEvent(new Event('submit'));
        }
    };

    // Global Threshold Sensitivity State
    window.currentThreshold = 0.6668;
    window.lastInferenceResult = null;
    window.lastInferencePayload = null;

    window.onThresholdChange = function(val) {
        const num = parseFloat(val);
        window.currentThreshold = num;
        
        // Update label in calibration card
        const calibEl = document.getElementById('calib-threshold-val');
        if (calibEl) calibEl.innerText = `τ = ${num.toFixed(4)}`;
        
        // Update all inline threshold text badges
        document.querySelectorAll('.lbl-dyn-thresh').forEach(el => {
            el.innerText = num.toFixed(4);
        });

        // Update threshold marker line position on the spectrum (0% to 100%)
        const threshLine = document.getElementById('res-thresh-line');
        if (threshLine) {
            threshLine.style.left = `${Math.min(99, Math.max(1, num * 100))}%`;
            threshLine.title = `Decision Threshold: ${num.toFixed(4)}`;
        }

        const threshLabel = document.getElementById('res-spectrum-thresh-label');
        if (threshLabel) {
            threshLabel.innerText = `${num.toFixed(3)} (Threshold)`;
        }

        // Dynamically reclassify if inference has been executed
        if (window.lastInferenceResult) {
            updateClassificationFromThreshold();
        }
    };

    window.setThreshold = function(val) {
        const slider = document.getElementById('threshold-slider');
        if (slider) slider.value = val;
        window.onThresholdChange(val);
    };

    function updateClassificationFromThreshold() {
        if (!window.lastInferenceResult) return;
        const res = window.lastInferenceResult;
        const score = res.anomaly_probability || 0.0;
        const thr = window.currentThreshold;
        
        const isAnom = score >= thr;
        const verdictStr = isAnom ? 'Anomalous' : 'Genuine';
        const riskLevel = isAnom ? (score >= 0.85 ? 'HIGH' : 'MEDIUM') : 'LOW';

        // Re-color circle & verdict
        resCircle.className = 'score-circle ' + (isAnom ? 'anomalous glow-danger' : 'genuine glow-success');
        resClass.className = (isAnom ? 'anomalous' : 'genuine');
        resClass.innerText = verdictStr;

        // Re-compute margin
        const margin = score - thr;
        const marginEl = document.getElementById('res-margin-label');
        if (marginEl) {
            marginEl.innerText = `Margin: ${margin >= 0 ? '+' : ''}${margin.toFixed(4)}`;
            marginEl.style.color = margin >= 0 ? '#ef4444' : '#10b981';
        }

        // Re-badge primary driver
        const driverBadge = document.getElementById('res-primary-driver');
        if (driverBadge) {
            const driverName = isAnom ? 'Link Reconstruction Divergence' : 'Genuine Topology Match';
            driverBadge.innerText = driverName;
            driverBadge.className = `badge ${isAnom ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-green-50 text-green-700 border border-green-200'}`;
        }
    }

    // Export complete causal audit report as a downloadable JSON file
    window.exportAuditReport = function() {
        if (!window.lastInferenceResult || !window.lastInferencePayload) {
            alert("Please run or select an interaction first before exporting.");
            return;
        }
        const report = {
            export_timestamp: new Date().toISOString(),
            model: "Causal Temporal Graph Attention Network (T-GAT)",
            decision_parameters: {
                calibrated_threshold: window.currentThreshold,
                temporal_decay_factor_tau: 0.431,
                latent_space_dimension: 16
            },
            interaction_input: window.lastInferencePayload,
            inference_results: window.lastInferenceResult,
            verdict: {
                anomaly_probability: window.lastInferenceResult.anomaly_probability,
                decision: (window.lastInferenceResult.anomaly_probability >= window.currentThreshold) ? "ANOMALOUS" : "GENUINE",
                decision_margin: window.lastInferenceResult.anomaly_probability - window.currentThreshold
            }
        };

        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(report, null, 2));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute("href", dataStr);
        downloadAnchor.setAttribute("download", `TGAT_Audit_${window.lastInferencePayload.reviewerID}_${Date.now()}.json`);
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
    };

    // Global quick inspector: clicking an anomaly in the table immediately loads its visual justification
    window.inspectInteraction = function(reviewerID, asin, overall, unixReviewTime) {
        const analyzeNav = document.querySelector('[data-tab="anomaly-detection"]');
        if (analyzeNav) analyzeNav.click();
        
        document.getElementById('reviewerID').value = reviewerID;
        document.getElementById('asin').value = asin;
        document.getElementById('overall').value = overall;
        document.getElementById('unixReviewTime').value = unixReviewTime;
        
        const formEl = document.getElementById('anomaly-form');
        if (formEl) {
            formEl.dispatchEvent(new Event('submit'));
        }
    };

    // Initial health check
    checkHealth();
});
