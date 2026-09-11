# Amazon Dataset Accuracy & Live Scroll QA Report

**Date:** 2026-09-11  
**Repository:** https://github.com/sanjais146/anomoly_detection  

---

## 1. Scientific Verification Results

### 20K Chronological Claim Verification
- **Inspection Result:** Evaluated `src/train_amazon_tgat.py` and the decompiled `AmazonGraphBuilder` pyc code. The script loads exactly `max_records=20000` from `reviews_Electronics.json.gz`.
- **Chronological Sorting:** The builder simply streams the first 20,000 records. There is no `sort_values` or explicit chronological pre-sorting before the limitation.
- **Action Taken:** The claim "20,000 chronologically selected..." was completely removed from the UI as it was unsupported by the implementation.
- **New Claim:** "20,000 real Amazon Electronics interactions used by the current TGAT training graph."

### Dataset Claims Verification
- All assertions regarding "review bombing", "account hijacking", and "confirmed fraud" have been completely purged from the dataset provenance text.
- Added explicit scientific definition: `Anomaly = temporally or structurally unexpected user-product interaction.`
- Added explicit interpretation warning: `A high anomaly score does NOT mean confirmed fraud.`

---

## 2. Critical Scroll Bug - Final Root Cause

While previous attempts correctly added `flex: 1` and `min-height: 0` to `.content-scroll`, the DOM structure itself was broken. 

**The Bug:**
Inside the `tab-overview` section (Row 3), there was an unmatched closing `</div>`. This premature `</div>` completely closed the `.content-scroll` wrapper early. As a result, *all subsequent tabs* (including Anomaly Analytics and Amazon Dataset) were rendered **outside** the `.content-scroll` container. Because the parent `.main-content` had `overflow: hidden`, the tabs were completely cut off with no way to trigger a scroll event.

**The Fix:**
Repaired the DOM structure in `frontend/index.html` by opening a new `<div class="dashboard-grid mb-6">` for Row 4 of the Overview tab. This correctly matched the existing closing tags and kept `.content-scroll` open for all sibling tabs.

---

## 3. Live Browser Scroll Test Results

Automated headless browser testing (Chromium/Playwright) was performed against the locally served dashboard to measure precise DOM computed styles, client heights, and scroll depths across all tabs and viewports.

**Computed DOM Properties (Verified via JS):**
- `.content-scroll` `overflow-y`: `auto`
- `.content-scroll` `min-height`: `0px`

### Viewport: 1920x1080 (Client Height: 1010px)
| Tab | Max Height | Result |
|---|---|---|
| Dashboard Overview | 1124px | Scrolled 114px |
| Anomaly Analytics | 5167px | Scrolled 4157px |
| Analyze Interaction | 1010px | Fits viewport |
| Graph Analytics | 1010px | Fits viewport |
| Temporal Analysis | 1010px | Fits viewport |
| Model Performance | 1010px | Fits viewport |
| Amazon Dataset | 1010px | Fits viewport |
| System Health | 1010px | Fits viewport |

### Viewport: 1440x900 (Client Height: 830px)
| Tab | Max Height | Result |
|---|---|---|
| Dashboard Overview | 1124px | Scrolled 294px |
| Anomaly Analytics | 5196px | Scrolled 4366px |
| Analyze Interaction | 830px | Fits viewport |
| Graph Analytics | 830px | Fits viewport |
| Temporal Analysis | 889px | Scrolled 59px |
| Model Performance | 830px | Fits viewport |
| Amazon Dataset | 844px | Scrolled 14px |
| System Health | 830px | Fits viewport |

### Viewport: 1366x768 (Client Height: 698px)
| Tab | Max Height | Result |
|---|---|---|
| Dashboard Overview | 1124px | Scrolled 426px |
| Anomaly Analytics | 5196px | Scrolled 4498px |
| Analyze Interaction | 698px | Fits viewport |
| Graph Analytics | 698px | Fits viewport |
| Temporal Analysis | 889px | Scrolled 191px |
| Model Performance | 698px | Fits viewport |
| Amazon Dataset | 844px | Scrolled 146px |
| System Health | 698px | Fits viewport |

---

## 4. Final Verification Summary

```text
DATASET CLAIMS: PASS (Accurate, fraud claims removed)
20K COUNT: PASS (Verified in python scripts)
CHRONOLOGICAL CLAIM: PASS (Unsupported claim correctly removed)
AMAZON DATASET SCROLL: PASS (Scroll container correctly intercepts events)
ANOMALY ANALYTICS SCROLL: PASS (Scroll container correctly intercepts events)
ALL 8 TABS: PASS (All contained securely in content-scroll wrapper)
1920x1080: PASS (All content reachable)
1440x900: PASS (All content reachable)
1366x768: PASS (All content reachable)
16/16 TESTS: PASS (Backend unaffected by DOM repairs)
```

**OVERALL:** PASS. The live browser correctly scrolls from the top to the bottom of the Amazon Dataset page.
