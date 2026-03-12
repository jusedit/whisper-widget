"""Apple Dynamic Island-style floating overlay using PyQt6."""

import math
from collections import deque
from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRectF, pyqtProperty,
)
from PyQt6.QtGui import (
    QPainter, QColor, QBrush, QPen, QRadialGradient, QLinearGradient,
    QPainterPath, QFont,
)
from PyQt6.QtWidgets import QWidget, QApplication


class NotchOverlay(QWidget):
    """Floating Dynamic Island-style overlay at top-center of screen."""

    HIDDEN = 0
    RECORDING = 1
    TRANSCRIBING = 2
    SUCCESS = 3
    LOADING = 4

    # Dimensions
    MAX_W = 260
    MAX_H = 52
    MARGIN_TOP = 12

    # Target pill sizes per state
    REC_W = 240
    REC_H = 44
    TRANS_W = 180
    TRANS_H = 40
    SUCCESS_W = 160
    SUCCESS_H = 38
    LOAD_W = 200
    LOAD_H = 40

    def __init__(self):
        super().__init__()
        self._state = self.HIDDEN
        self._level = 0.0
        self._wave_history: deque[float] = deque([0.0] * 48, maxlen=48)
        self._anim_phase = 0.0

        # Animated properties
        self._pill_w = 0.0
        self._pill_h = 0.0
        self._pill_opacity = 0.0

        # Window setup
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(self.MAX_W, self.MAX_H)

        self._reposition()

        # Render timer (~60fps)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        # Property animations
        self._w_anim = QPropertyAnimation(self, b"pill_w")
        self._w_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._h_anim = QPropertyAnimation(self, b"pill_h")
        self._h_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._opacity_anim = QPropertyAnimation(self, b"pill_opacity")
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def _reposition(self):
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.geometry()
            x = geom.x() + (geom.width() - self.MAX_W) // 2
            y = geom.y() + self.MARGIN_TOP
            self.move(x, y)

    # --- Animated properties ---

    @pyqtProperty(float)
    def pill_w(self):
        return self._pill_w

    @pill_w.setter
    def pill_w(self, val):
        self._pill_w = val
        self.update()

    @pyqtProperty(float)
    def pill_h(self):
        return self._pill_h

    @pill_h.setter
    def pill_h(self, val):
        self._pill_h = val
        self.update()

    @pyqtProperty(float)
    def pill_opacity(self):
        return self._pill_opacity

    @pill_opacity.setter
    def pill_opacity(self, val):
        self._pill_opacity = val
        self.update()

    def _animate_to(self, w, h, opacity, duration=300):
        for anim, target in [
            (self._w_anim, w), (self._h_anim, h), (self._opacity_anim, opacity),
        ]:
            anim.stop()
            anim.setDuration(duration)
            anim.setStartValue(anim.targetObject().property(anim.propertyName().data().decode()))
            anim.setEndValue(target)
            anim.start()

    # --- State transitions ---

    def show_loading(self):
        """Show boot/splash animation while model loads."""
        self._state = self.LOADING
        self._anim_phase = 0.0
        self._timer.start(16)
        self.show()
        self._reposition()
        self._animate_to(self.LOAD_W, self.LOAD_H, 1.0, 500)

    def hide_loading(self):
        """Transition from loading to ready — brief success flash then hide."""
        if self._state != self.LOADING:
            return
        self._state = self.SUCCESS
        self._anim_phase = 0.0
        self._animate_to(self.SUCCESS_W, self.SUCCESS_H, 1.0, 200)
        QTimer.singleShot(800, self._fade_out)

    def show_recording(self):
        self._state = self.RECORDING
        self._wave_history = deque([0.0] * 48, maxlen=48)
        self._timer.start(16)  # ~60fps
        self.show()
        self._reposition()
        self._animate_to(self.REC_W, self.REC_H, 1.0, 350)

    def show_transcribing(self):
        self._state = self.TRANSCRIBING
        self._anim_phase = 0.0
        self._timer.start(16)
        self._animate_to(self.TRANS_W, self.TRANS_H, 1.0, 300)

    def show_success(self):
        self._state = self.SUCCESS
        self._anim_phase = 0.0
        self._animate_to(self.SUCCESS_W, self.SUCCESS_H, 1.0, 200)
        QTimer.singleShot(600, self._fade_out)

    def _fade_out(self):
        self._animate_to(0, 0, 0.0, 350)
        QTimer.singleShot(400, self.hide_overlay)

    def hide_overlay(self):
        self._state = self.HIDDEN
        self._timer.stop()
        self._pill_w = 0
        self._pill_h = 0
        self._pill_opacity = 0
        self.hide()

    def set_level(self, level: float):
        self._level = level
        self._wave_history.append(level)

    def _tick(self):
        self._anim_phase += 0.08
        self.update()

    # --- Painting ---

    def paintEvent(self, event):
        if self._pill_w < 2 or self._pill_opacity < 0.01:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(self._pill_opacity)

        # Center the pill in the widget
        pw, ph = self._pill_w, self._pill_h
        px = (self.MAX_W - pw) / 2
        py = (self.MAX_H - ph) / 2
        radius = ph / 2

        pill_rect = QRectF(px, py, pw, ph)

        # Draw shadow
        self._draw_shadow(p, pill_rect, radius)

        # Draw pill background
        self._draw_pill_bg(p, pill_rect, radius)

        # Draw content
        if self._state == self.RECORDING:
            self._paint_recording(p, pill_rect)
        elif self._state == self.TRANSCRIBING:
            self._paint_transcribing(p, pill_rect)
        elif self._state == self.SUCCESS:
            self._paint_success(p, pill_rect, radius)
        elif self._state == self.LOADING:
            self._paint_loading(p, pill_rect, radius)

        p.end()

    def _draw_shadow(self, p: QPainter, rect: QRectF, radius: float):
        """Subtle drop shadow beneath the pill."""
        shadow_rect = QRectF(rect.x() + 2, rect.y() + 3, rect.width() - 4, rect.height())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 50))
        p.drawRoundedRect(shadow_rect, radius, radius)

    def _draw_pill_bg(self, p: QPainter, rect: QRectF, radius: float):
        """Dark glass pill background with subtle border."""
        # Fill
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(22, 22, 26, 245))
        p.drawRoundedRect(rect, radius, radius)

        # Subtle glass border
        border_pen = QPen(QColor(255, 255, 255, 18))
        border_pen.setWidthF(1.0)
        p.setPen(border_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        inset = QRectF(rect.x() + 0.5, rect.y() + 0.5,
                       rect.width() - 1, rect.height() - 1)
        p.drawRoundedRect(inset, radius - 0.5, radius - 0.5)

        # Top highlight (glass reflection)
        highlight = QLinearGradient(rect.x(), rect.y(), rect.x(), rect.y() + 8)
        highlight.setColorAt(0, QColor(255, 255, 255, 10))
        highlight.setColorAt(1, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(highlight))
        p.drawRoundedRect(
            QRectF(rect.x() + 1, rect.y() + 1, rect.width() - 2, 8),
            radius - 1, radius - 1,
        )

    def _paint_recording(self, p: QPainter, rect: QRectF):
        cx, cy = rect.x(), rect.y() + rect.height() / 2

        # --- Pulsing red dot ---
        pulse = 0.5 + 0.5 * math.sin(self._anim_phase * 3)
        dot_x = cx + 22
        dot_y = cy
        dot_r = 5.0 + pulse * 1.5

        # Outer glow
        glow = QRadialGradient(dot_x, dot_y, dot_r * 3)
        glow.setColorAt(0, QColor(239, 68, 68, int(60 * pulse)))
        glow.setColorAt(1, QColor(239, 68, 68, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QRectF(dot_x - dot_r * 3, dot_y - dot_r * 3,
                             dot_r * 6, dot_r * 6))

        # Inner dot
        inner_glow = QRadialGradient(dot_x - 1, dot_y - 1, dot_r)
        inner_glow.setColorAt(0, QColor(255, 120, 120))
        inner_glow.setColorAt(0.5, QColor(239, 68, 68))
        inner_glow.setColorAt(1, QColor(200, 50, 50))
        p.setBrush(QBrush(inner_glow))
        p.drawEllipse(QRectF(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2))

        # --- Smooth waveform ---
        wave_x = cx + 40
        wave_end = rect.right() - 16
        wave_w = wave_end - wave_x

        if wave_w > 20 and len(self._wave_history) >= 4:
            self._draw_waveform(p, wave_x, cy, wave_w, rect.height() * 0.65)

    def _draw_waveform(self, p: QPainter, x: float, cy: float,
                       w: float, max_h: float):
        """Draw smooth mirrored waveform using cubic bezier curves."""
        n = len(self._wave_history)
        step = w / (n - 1)

        # Build smooth path for upper half
        upper = QPainterPath()
        lower = QPainterPath()
        upper.moveTo(x, cy)
        lower.moveTo(x, cy)

        points = []
        for i, lev in enumerate(self._wave_history):
            px = x + i * step
            amp = lev * max_h / 2
            # Smooth the amplitude
            amp = max(1.0, amp)
            points.append((px, amp))

        # Draw with cubic bezier for smoothness
        for i in range(len(points)):
            px, amp = points[i]
            if i == 0:
                upper.moveTo(px, cy - amp)
                lower.moveTo(px, cy + amp)
            elif i == 1:
                upper.lineTo(px, cy - amp)
                lower.lineTo(px, cy + amp)
            else:
                prev_px, prev_amp = points[i - 1]
                cp_x = (prev_px + px) / 2
                upper.cubicTo(cp_x, cy - prev_amp, cp_x, cy - amp, px, cy - amp)
                lower.cubicTo(cp_x, cy + prev_amp, cp_x, cy + amp, px, cy + amp)

        # Create filled area
        fill_path = QPainterPath()
        fill_path.addPath(upper)
        # Connect upper end to lower end
        last_x = points[-1][0] if points else x
        fill_path.lineTo(last_x, cy)
        # Walk lower path in reverse
        for i in range(len(points) - 1, -1, -1):
            px, amp = points[i]
            fill_path.lineTo(px, cy + amp)
        fill_path.closeSubpath()

        # Gradient fill
        grad = QLinearGradient(x, cy - max_h / 2, x, cy + max_h / 2)
        grad.setColorAt(0, QColor(239, 68, 68, 40))
        grad.setColorAt(0.3, QColor(239, 68, 68, 140))
        grad.setColorAt(0.5, QColor(239, 68, 68, 180))
        grad.setColorAt(0.7, QColor(239, 68, 68, 140))
        grad.setColorAt(1, QColor(239, 68, 68, 40))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawPath(fill_path)

        # Stroke the upper and lower edges
        edge_pen = QPen(QColor(239, 100, 100, 200))
        edge_pen.setWidthF(1.2)
        p.setPen(edge_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(upper)
        p.drawPath(lower)

    def _paint_transcribing(self, p: QPainter, rect: QRectF):
        cx = rect.x() + rect.width() / 2
        cy = rect.y() + rect.height() / 2

        # --- Shimmer effect ---
        shimmer_phase = (self._anim_phase * 0.4) % 1.0
        shimmer_x = rect.x() + shimmer_phase * rect.width()
        shimmer = QLinearGradient(shimmer_x - 40, 0, shimmer_x + 40, 0)
        shimmer.setColorAt(0, QColor(255, 255, 255, 0))
        shimmer.setColorAt(0.5, QColor(255, 255, 255, 8))
        shimmer.setColorAt(1, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(shimmer))
        radius = rect.height() / 2
        p.drawRoundedRect(rect, radius, radius)

        # --- Bouncing dots ---
        dot_spacing = 14
        n_dots = 3
        dots_x = cx - (n_dots - 1) * dot_spacing / 2

        for i in range(n_dots):
            dx = dots_x + i * dot_spacing
            phase = self._anim_phase * 3 - i * 0.8
            bounce = max(0, math.sin(phase)) * 6
            scale = 1.0 + max(0, math.sin(phase)) * 0.3
            alpha = 140 + int(100 * max(0, math.sin(phase)))

            # Dot glow
            glow_r = 5 * scale + 4
            glow = QRadialGradient(dx, cy - bounce, glow_r)
            glow.setColorAt(0, QColor(99, 132, 255, int(alpha * 0.3)))
            glow.setColorAt(1, QColor(99, 132, 255, 0))
            p.setBrush(QBrush(glow))
            p.drawEllipse(QRectF(dx - glow_r, cy - bounce - glow_r,
                                 glow_r * 2, glow_r * 2))

            # Dot
            r = 4 * scale
            dot_grad = QRadialGradient(dx - 0.5, cy - bounce - 0.5, r)
            dot_grad.setColorAt(0, QColor(140, 160, 255, alpha))
            dot_grad.setColorAt(0.6, QColor(99, 102, 241, alpha))
            dot_grad.setColorAt(1, QColor(70, 72, 200, alpha))
            p.setBrush(QBrush(dot_grad))
            p.drawEllipse(QRectF(dx - r, cy - bounce - r, r * 2, r * 2))

    def _paint_success(self, p: QPainter, rect: QRectF, radius: float):
        """Brief green flash overlay."""
        flash = max(0, math.sin(self._anim_phase * 4))
        overlay_color = QColor(52, 199, 89, int(60 * flash))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(overlay_color))
        p.drawRoundedRect(rect, radius, radius)

        # Checkmark
        cx = rect.x() + rect.width() / 2
        cy = rect.y() + rect.height() / 2
        check_pen = QPen(QColor(52, 199, 89, int(200 * min(1.0, self._anim_phase * 2))))
        check_pen.setWidthF(2.5)
        check_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        check_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(check_pen)

        path = QPainterPath()
        path.moveTo(cx - 8, cy)
        path.lineTo(cx - 2, cy + 6)
        path.lineTo(cx + 8, cy - 5)
        p.drawPath(path)

    def _paint_loading(self, p: QPainter, rect: QRectF, radius: float):
        """Boot animation: sweeping arc + subtle text."""
        cx = rect.x() + rect.width() / 2
        cy = rect.y() + rect.height() / 2

        # Sweeping shimmer across pill
        shimmer_phase = (self._anim_phase * 0.3) % 1.0
        shimmer_x = rect.x() + shimmer_phase * rect.width()
        shimmer = QLinearGradient(shimmer_x - 60, 0, shimmer_x + 60, 0)
        shimmer.setColorAt(0, QColor(255, 255, 255, 0))
        shimmer.setColorAt(0.5, QColor(255, 255, 255, 6))
        shimmer.setColorAt(1, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(shimmer))
        p.drawRoundedRect(rect, radius, radius)

        # Spinning arc on the left
        arc_x = rect.x() + 20
        arc_y = cy
        arc_r = 7.0
        arc_rect = QRectF(arc_x - arc_r, arc_y - arc_r, arc_r * 2, arc_r * 2)

        # Arc sweep
        start_angle = int(self._anim_phase * 200) % 5760
        span = 2400  # 240 degrees

        # Glow behind arc
        glow = QRadialGradient(arc_x, arc_y, arc_r * 2.5)
        glow.setColorAt(0, QColor(99, 132, 255, 30))
        glow.setColorAt(1, QColor(99, 132, 255, 0))
        p.setBrush(QBrush(glow))
        p.drawEllipse(QRectF(arc_x - arc_r * 2.5, arc_y - arc_r * 2.5,
                             arc_r * 5, arc_r * 5))

        # Draw arc
        arc_pen = QPen(QColor(99, 132, 255, 200))
        arc_pen.setWidthF(2.0)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(arc_rect, start_angle, span)

        # Fainter track ring
        track_pen = QPen(QColor(99, 132, 255, 30))
        track_pen.setWidthF(2.0)
        p.setPen(track_pen)
        p.drawEllipse(arc_rect)

        # "Loading..." text
        fade = min(1.0, self._anim_phase * 0.5)  # fade in
        text_alpha = int(180 * fade)
        p.setPen(QColor(200, 200, 210, text_alpha))
        font = QFont("Segoe UI", 9)
        font.setWeight(QFont.Weight.Medium)
        p.setFont(font)
        text_rect = QRectF(arc_x + arc_r + 8, rect.y(), rect.right() - arc_x - arc_r - 24, rect.height())
        p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "Loading model...")
