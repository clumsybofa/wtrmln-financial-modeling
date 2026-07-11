"""Playwright-backed virtual display implementing the computer_20251124 action set.

The Claude computer-use tool emits actions like {"action": "left_click",
"coordinate": [x, y]}; VirtualBrowser.execute() maps each one onto a real
Chromium page. The same page object also receives forwarded user input during
the human-login handoff, so the user and the agent share one browser session.
"""

import asyncio
import base64
import io
import os

from playwright.async_api import async_playwright

VIEWPORT = {"width": 1280, "height": 800}

# xdotool-style key names (what the model emits) -> Playwright key names
_KEY_MAP = {
    "return": "Enter", "enter": "Enter", "kp_enter": "Enter",
    "tab": "Tab", "space": "Space", "backspace": "Backspace",
    "delete": "Delete", "escape": "Escape", "esc": "Escape",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "page_up": "PageUp", "page_down": "PageDown", "home": "Home", "end": "End",
    "ctrl": "Control", "control": "Control", "alt": "Alt", "shift": "Shift",
    "super": "Meta", "meta": "Meta", "cmd": "Meta",
    "minus": "-", "plus": "+", "equal": "=",
}


def _map_key(combo: str) -> str:
    parts = combo.replace(" ", "").split("+")
    mapped = []
    for p in parts:
        low = p.lower()
        if low in _KEY_MAP:
            mapped.append(_KEY_MAP[low])
        elif len(p) == 1:
            mapped.append(p)
        else:
            mapped.append(p[0].upper() + p[1:].lower())  # F5, etc. pass through
    return "+".join(mapped)


class VirtualBrowser:
    def __init__(self):
        self._pw = None
        self._browser = None
        self.page = None
        self._lock = asyncio.Lock()

    async def start(self, start_url: str = "about:blank"):
        self._pw = await async_playwright().start()
        launch_kwargs = {"headless": True}
        exe = os.environ.get("WTRMLN_CHROMIUM_PATH")
        if exe:
            launch_kwargs["executable_path"] = exe
        context_kwargs = {"viewport": VIEWPORT}
        # Route through an outbound proxy when the environment mandates one
        # (e.g. sandboxed runners). The proxy re-signs TLS with its own CA,
        # which Chromium doesn't trust, so accept its certs in that case.
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy}
            context_kwargs["ignore_https_errors"] = True
            # Some TLS-inspecting egress proxies reset Chromium's TLS 1.3
            # ClientHello; the proxy re-terminates TLS anyway, so cap at 1.2.
            launch_kwargs["args"] = ["--ssl-version-max=tls1.2"]
        self._browser = await self._pw.chromium.launch(**launch_kwargs)
        context = await self._browser.new_context(**context_kwargs)
        self.page = await context.new_page()
        if start_url != "about:blank":
            try:
                await self.page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
            except Exception:
                pass  # agent sees the blank/error page and navigates itself

    async def stop(self):
        try:
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass

    async def screenshot_b64(self) -> str:
        async with self._lock:
            png = await self.page.screenshot(type="png")
        return base64.b64encode(png).decode()

    async def navigate(self, url: str):
        async with self._lock:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)

    async def execute(self, tool_input: dict) -> dict:
        """Run one computer-use action. Returns a dict with either
        {"image_b64": ...} (screenshot-producing actions) or {"text": ...}."""
        action = tool_input.get("action")
        coord = tool_input.get("coordinate")
        modifier = tool_input.get("text") if action in (
            "left_click", "right_click", "middle_click", "double_click",
            "triple_click", "scroll",
        ) else None

        async with self._lock:
            page = self.page
            if action == "screenshot":
                pass  # screenshot taken below
            elif action == "left_click":
                await self._click(coord, "left", modifier)
            elif action == "right_click":
                await self._click(coord, "right", modifier)
            elif action == "middle_click":
                await self._click(coord, "middle", modifier)
            elif action == "double_click":
                await self._click(coord, "left", modifier, count=2)
            elif action == "triple_click":
                await self._click(coord, "left", modifier, count=3)
            elif action == "type":
                await page.keyboard.type(tool_input.get("text", ""), delay=12)
            elif action == "key":
                await page.keyboard.press(_map_key(tool_input.get("text", "")))
            elif action == "hold_key":
                key = _map_key(tool_input.get("text", ""))
                await page.keyboard.down(key)
                await asyncio.sleep(min(float(tool_input.get("duration", 1)), 5))
                await page.keyboard.up(key)
            elif action == "mouse_move":
                await page.mouse.move(*coord)
            elif action == "left_mouse_down":
                if coord:
                    await page.mouse.move(*coord)
                await page.mouse.down()
            elif action == "left_mouse_up":
                if coord:
                    await page.mouse.move(*coord)
                await page.mouse.up()
            elif action == "left_click_drag":
                start = tool_input.get("start_coordinate") or coord
                end = coord
                await page.mouse.move(*start)
                await page.mouse.down()
                await page.mouse.move(*end, steps=12)
                await page.mouse.up()
            elif action == "scroll":
                x, y = coord or (VIEWPORT["width"] // 2, VIEWPORT["height"] // 2)
                amount = int(tool_input.get("scroll_amount", 3)) * 120
                direction = tool_input.get("scroll_direction", "down")
                dx, dy = 0, 0
                if direction == "down":
                    dy = amount
                elif direction == "up":
                    dy = -amount
                elif direction == "right":
                    dx = amount
                elif direction == "left":
                    dx = -amount
                await page.mouse.move(x, y)
                if modifier:
                    await page.keyboard.down(_map_key(modifier))
                await page.mouse.wheel(dx, dy)
                if modifier:
                    await page.keyboard.up(_map_key(modifier))
            elif action == "wait":
                await asyncio.sleep(min(float(tool_input.get("duration", 1)), 10))
            elif action == "zoom":
                region = tool_input.get("region")
                png = await page.screenshot(
                    type="png",
                    clip={
                        "x": region[0], "y": region[1],
                        "width": max(region[2] - region[0], 1),
                        "height": max(region[3] - region[1], 1),
                    },
                )
                return {"image_b64": base64.b64encode(png).decode()}
            else:
                return {"text": f"Unsupported action: {action}"}

            # give the page a beat to settle after interaction, then screenshot
            if action != "screenshot":
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
                await asyncio.sleep(0.4)
            png = await page.screenshot(type="png")
            return {"image_b64": base64.b64encode(png).decode()}

    async def _click(self, coord, button, modifier, count: int = 1):
        if modifier:
            await self.page.keyboard.down(_map_key(modifier))
        await self.page.mouse.click(coord[0], coord[1], button=button, click_count=count)
        if modifier:
            await self.page.keyboard.up(_map_key(modifier))

    # --- direct user input during human-login handoff -----------------------

    async def user_click(self, x: float, y: float):
        async with self._lock:
            await self.page.mouse.click(x, y)

    async def user_type(self, text: str):
        async with self._lock:
            await self.page.keyboard.type(text, delay=10)

    async def user_key(self, key: str):
        async with self._lock:
            await self.page.keyboard.press(_map_key(key))
