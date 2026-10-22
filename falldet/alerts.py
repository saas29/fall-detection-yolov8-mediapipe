"""
Alert generation (Sec. III-G "Sending Alert").

Always: console message + snapshot image in alerts/.
Optional:
  - Telegram photo message  (env TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
  - PC beep                 (--beep, Windows only)
  - GPIO buzzer             (--buzzer-pin N, Raspberry Pi with gpiozero)

Network / buzzer work runs on a background thread so the video loop never
blocks on an alert.
"""

import os
import threading
import time
from datetime import datetime

import cv2


class AlertManager:
    def __init__(self, cooldown_s=5.0, snapshot_dir="alerts", beep=False,
                 buzzer_pin=None, telegram=True, enabled=True):
        self.cooldown_s = cooldown_s
        self.snapshot_dir = snapshot_dir
        self.beep = beep
        self.enabled = enabled
        self._last_alert_t = -1e9
        self.count = 0

        self._tg_token = os.environ.get("TELEGRAM_BOT_TOKEN") if telegram else None
        self._tg_chat = os.environ.get("TELEGRAM_CHAT_ID") if telegram else None

        self._buzzer = None
        if buzzer_pin is not None:
            try:
                from gpiozero import Buzzer
                self._buzzer = Buzzer(buzzer_pin)
            except Exception as e:  # not on a Pi / gpiozero missing
                print(f"[alert] buzzer disabled: {e}")

    def reset(self):
        self._last_alert_t = -1e9
        self.count = 0

    def trigger(self, frame, message="Fall detected"):
        """Returns True if an alert was actually sent (cooldown passed)."""
        if not self.enabled:
            return False
        now = time.time()
        if now - self._last_alert_t < self.cooldown_s:
            return False
        self._last_alert_t = now
        self.count += 1

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ALERT] {stamp} {message}")

        snap_path = None
        if self.snapshot_dir and frame is not None:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            snap_path = os.path.join(
                self.snapshot_dir, datetime.now().strftime("fall_%Y%m%d_%H%M%S.jpg"))
            cv2.imwrite(snap_path, frame)

        threading.Thread(target=self._dispatch, args=(f"{message} at {stamp}", snap_path),
                         daemon=True).start()
        return True

    def _dispatch(self, text, snap_path):
        if self.beep:
            try:
                import winsound
                winsound.Beep(2000, 700)
            except Exception:
                pass

        if self._buzzer is not None:
            try:
                self._buzzer.beep(on_time=0.3, off_time=0.2, n=5, background=True)
            except Exception as e:
                print(f"[alert] buzzer error: {e}")

        if self._tg_token and self._tg_chat:
            self._send_telegram(text, snap_path)

    def _send_telegram(self, text, snap_path):
        try:
            import requests
            base = f"https://api.telegram.org/bot{self._tg_token}"
            if snap_path and os.path.exists(snap_path):
                with open(snap_path, "rb") as f:
                    requests.post(f"{base}/sendPhoto",
                                  data={"chat_id": self._tg_chat, "caption": text},
                                  files={"photo": f}, timeout=10)
            else:
                requests.post(f"{base}/sendMessage",
                              data={"chat_id": self._tg_chat, "text": text}, timeout=10)
        except Exception as e:
            print(f"[alert] telegram error: {e}")
