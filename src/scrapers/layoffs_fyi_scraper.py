import asyncio
from playwright.async_api import async_playwright
import pandas as pd

async def scrape_airtable():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # set True to hide browser
        page = await browser.new_page()
        await page.goto("https://airtable.com/app1PaujS9zxVGUZ4/shroKsHx3SdYYOzeh/tblleV7Pnb6AcPCYL?viewControls=on")

        try:
            # wait briefly for the cookie banner's close button, adjust selector as needed
            close_btn = await page.wait_for_selector("button[aria-label='Close'], button[title='Close'], button:has-text('×'), button:has-text('x')", timeout=5000)
            async with page.expect_navigation(wait_until='load', timeout=10000):
                await close_btn.click()
            print("[INFO] Closed cookie notification banner.")
        except Exception:
            print("[INFO] No cookie notification banner close button found.")

        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(await page.content())

        print("[INFO] Waiting for Airtable table to appear...")
        await page.wait_for_selector("div.gridView.baymax", timeout=120_000)

       # select the scrollable container
        scrollable = await page.query_selector("div.antiscroll-inner")
        if scrollable is None:
            print("[ERROR] Scrollable container not found!")
        else:
            previous_scroll_top = -1
            scroll_attempts = 0
            max_attempts = 30

            while scroll_attempts < max_attempts:
                scroll_top = await scrollable.evaluate("el => el.scrollTop")
                scroll_height = await scrollable.evaluate("el => el.scrollHeight")
                client_height = await scrollable.evaluate("el => el.clientHeight")

                print(f"[DEBUG] scrollTop={scroll_top}, scrollHeight={scroll_height}, clientHeight={client_height}")

                if scroll_top == previous_scroll_top:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0

                if scroll_top + client_height >= scroll_height:
                    print("[INFO] Reached bottom of scrollable content.")
                    break

                # scroll down by clientHeight pixels
                await scrollable.evaluate(f"el => el.scrollTop = {scroll_top + client_height}")
                previous_scroll_top = scroll_top

                # wait to let content load/render
                await page.wait_for_timeout(1500)


        print("[INFO] Extracting headers and rows...")

        # get column headers
        headers = await page.eval_on_selector_all(
            "div.readonly.gridHeaderCellPhosphorIcons", # error here
            # <div class="headerRightPane pane">
            "nodes => nodes.map(n => n.innerText.trim())"
        )

        # get all rows
        rows = await page.eval_on_selector_all(
            "div.headerAndDataRowContainer",
            """
            rows => rows.map(row => {
                // Find all cell elements inside the row container
                const cells = Array.from(row.querySelectorAll('div.cursorCellBadgeContainer, div.fillHandleWrapper'))
                    .map(cell => cell.innerText.trim());
                return cells;
            })
            """
        )
        print(f"[DEBUG] Extracted {len(rows)} rows.")

        # save to DataFrame
        df = pd.DataFrame(rows, columns=headers)
        print(f"[SUCCESS] Extracted {len(df)} rows.")
        await browser.close()
        return df

if __name__ == "__main__":
    df = asyncio.run(scrape_airtable())
    df.to_csv("airtable_scrape.csv", index=False)
    print("Data saved to airtable_scrape.csv")

# scrape data as scrolling next