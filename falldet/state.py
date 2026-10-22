"""
Fall persistence and latch (Sec. III-G).

- A fall is confirmed only if the per-frame fall condition holds for N
  consecutive frames (persist).
- Once confirmed, the FALL status is latched for M frames (latch) so the
  output doesn't flicker.
"""


class FallStateMachine:
    def __init__(self, persist_frames=10, latch_frames=30):
        self.persist_frames = max(1, int(persist_frames))
        self.latch_frames = max(0, int(latch_frames))
        self.reset()

    def reset(self):
        self.consec = 0
        self.latch_remaining = 0
        self.confirmed_ever = False
        self.confirmations = 0

    def update(self, flagged):
        """
        Feed one frame's raw fall flag: True / False, or None when the frame
        gives no evidence (person not detected / no landmarks) - then the
        counter is held instead of reset, so short detection gaps don't
        break the N-frame persistence.
        Returns (in_fall, newly_confirmed).
        """
        if flagged is None:
            pass
        elif flagged:
            self.consec += 1
        else:
            self.consec = 0

        newly_confirmed = False
        if self.consec >= self.persist_frames:
            # rising edge = a new fall event
            if self.latch_remaining == 0:
                newly_confirmed = True
                self.confirmations += 1
            self.latch_remaining = max(self.latch_frames, 1)
            self.confirmed_ever = True

        in_fall = self.latch_remaining > 0
        if in_fall:
            self.latch_remaining -= 1
        return in_fall, newly_confirmed
