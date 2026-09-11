import asyncio
from playwright.async_api import async_playwright

async def run():
    url = "http://127.0.0.1:8000/"
    
    viewports = [
        {"width": 1920, "height": 1080},
        {"width": 1440, "height": 900},
        {"width": 1366, "height": 768}
    ]
    
    tabs = [
        {"name": "Dashboard Overview", "id": "overview"},
        {"name": "Anomaly Analytics", "id": "anomaly-analytics"},
        {"name": "Analyze Interaction", "id": "anomaly-detection"},
        {"name": "Graph Analytics", "id": "graph-analytics"},
        {"name": "Temporal Analysis", "id": "temporal-analysis"},
        {"name": "Model Performance", "id": "model-performance"},
        {"name": "Amazon Dataset", "id": "dataset"},
        {"name": "System Health", "id": "system-health"}
    ]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        for vp in viewports:
            print(f"\n--- Testing Viewport {vp['width']}x{vp['height']} ---")
            context = await browser.new_context(viewport=vp)
            page = await context.new_page()
            
            await page.goto(url)
            
            try:
                await page.wait_for_selector(".app-container", timeout=5000)
            except Exception:
                print("Failed to load .app-container")
                break
            
            # Check container styles
            container = page.locator(".content-scroll")
            overflow_y = await container.evaluate("el => window.getComputedStyle(el).overflowY")
            min_height = await container.evaluate("el => window.getComputedStyle(el).minHeight")
            print(f"  .content-scroll computed -> overflow-y: {overflow_y}, min-height: {min_height}")
            
            for tab_data in tabs:
                tab_name = tab_data["name"]
                tab_id = tab_data["id"]
                print(f"  Testing Tab: {tab_name}")
                
                # Click tab in sidebar
                await page.click(f"[data-tab='{tab_id}']")
                
                # Wait for the tab content to become visible
                await page.wait_for_selector(f"#tab-{tab_id}.active")
                await page.wait_for_timeout(200) # Let layout settle
                
                # Try scrolling the .content-scroll element
                scroll_info = await container.evaluate("""el => {
                    return {
                        scrollHeight: el.scrollHeight,
                        clientHeight: el.clientHeight,
                        scrollTop: el.scrollTop
                    }
                }""")
                
                can_scroll = scroll_info['scrollHeight'] > scroll_info['clientHeight']
                if not can_scroll:
                    print(f"    Tab '{tab_name}' fits in viewport (no scroll needed). scrollHeight: {scroll_info['scrollHeight']}, clientHeight: {scroll_info['clientHeight']}")
                    await page.screenshot(path=f"screenshot_{vp['width']}_{tab_name.replace(' ', '_')}.png")
                else:
                    # Scroll down
                    await container.evaluate("el => el.scrollTo(0, el.scrollHeight)")
                    await page.wait_for_timeout(500)
                    new_scroll = await container.evaluate("el => el.scrollTop")
                    
                    if new_scroll > 0:
                        print(f"    PASS: Successfully scrolled '{tab_name}' to {new_scroll}px (Max: {scroll_info['scrollHeight']})")
                    else:
                        print(f"    FAIL: Could not scroll '{tab_name}'.")
                
                # Specific verification for Amazon Dataset
                if tab_id == "dataset":
                    text_checks = [
                        "Dataset Provenance",
                        "Training Data Source",
                        "Training Graph",
                        "Dashboard Visualization Sample",
                        "Verified Real Graph Topology",
                        "Self-Supervised Anomaly Detection",
                        "Scientific Definition",
                        "IMPORTANT INTERPRETATION"
                    ]
                    for t in text_checks:
                        is_visible = await page.locator(f"text='{t}'").first.is_visible()
                        if is_visible:
                            print(f"      Text check '{t}': PASS (Visible)")
                        else:
                            print(f"      Text check '{t}': FAIL (Not visible)")
                
                # Scroll back to top
                await container.evaluate("el => el.scrollTo(0, 0)")
                await page.wait_for_timeout(200)

            await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
