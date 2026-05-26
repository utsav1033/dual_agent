"""Capture the Gradio app fullpage from Edge."""
import subprocess, time
import win32gui, win32con, win32api, win32con
from PIL import ImageGrab

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
URL  = "http://127.0.0.1:7860"
OUT  = r"e:\dual_agent\screenshot_ui.png"

def find_hwnd(fragment):
    found = []
    def cb(h, _):
        if win32gui.IsWindowVisible(h) and fragment.lower() in win32gui.GetWindowText(h).lower():
            found.append(h)
    win32gui.EnumWindows(cb, None)
    return found

subprocess.Popen([EDGE, "--new-window", "--start-maximized", URL])
print("Waiting for page to load…")
time.sleep(8)

windows = find_hwnd("Dual AI")
hwnd = windows[0] if windows else None
if not hwnd:
    windows = find_hwnd("127.0.0.1:7860")
    hwnd = windows[0] if windows else None

if hwnd:
    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)

    # Ctrl+0 = reset zoom, then Ctrl+Home = scroll to top
    win32api.keybd_event(0x11, 0, 0, 0)          # CTRL down
    win32api.keybd_event(0x30, 0, 0, 0)          # 0
    win32api.keybd_event(0x30, 0, 2, 0)          # 0 up
    win32api.keybd_event(0x11, 0, 2, 0)          # CTRL up
    time.sleep(0.3)
    win32api.keybd_event(0x11, 0, 0, 0)          # CTRL down
    win32api.keybd_event(0x24, 0, 0, 0)          # Home
    win32api.keybd_event(0x24, 0, 2, 0)          # Home up
    win32api.keybd_event(0x11, 0, 2, 0)          # CTRL up
    time.sleep(1)

    rect = win32gui.GetWindowRect(hwnd)
    img = ImageGrab.grab(bbox=(rect[0]+2, rect[1], rect[2]-2, rect[3]-2))
    img.save(OUT)
    print(f"Saved {img.width}x{img.height}")
else:
    ImageGrab.grab().save(OUT)
    print("Fallback saved")
