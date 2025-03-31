import re
from playwright.sync_api import Page, expect

def _test_has_title(page: Page):
    page.goto("https://playwright.dev/")
    expect(page).to_have_title(re.compile("Playwright"))


def _test_get_started_link(page: Page):
    page.goto("https://playwright.dev/")
    page.get_by_role("link", name="Get started").click()

    expect(page.get_by_role("heading", name="Installation")).to_be_visible()
