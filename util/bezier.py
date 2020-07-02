import numpy as np


# From: https://stackoverflow.com/questions/11696736/recreating-css3-transitions-cubic-bezier-curve
def create_bezier(name):
    if name is "Linear":
        return UnitBezier(0.0, 0.0, 1.0, 1.0)
    elif name is "Ease":
        return UnitBezier(0.25, 0.1, 0.25, 1.0)
    elif name is "EaseInSine":
        return UnitBezier(0.47, 0, 0.745, 0.715);
    elif name is "EaseOutSine":
        return UnitBezier(0.39, 0.575, 0.565, 1);
    elif name is "EaseInOutSine":
        return UnitBezier(0.445, 0.05, 0.55, 0.95)
    elif name is "EaseInOutExpo":
        return UnitBezier(1, 0, 0, 1)


class UnitBezier:
    # From: https://cubic-bezier.com
    # From: https://easings.net



    def __init__(self, x1, y1, x2, y2):
        # Pre-calculate the polynomial coefficients
        # First and last control points are implied to be (0,0) and (1.0, 1.0)
        self.cx = 3.0 * x1
        self.bx = 3.0 * (x2 - x1) - self.cx
        self.ax = 1.0 - self.cx - self.bx

        self.cy = 3.0 * y1
        self.by = 3.0 * (y2 - y1) - self.cy
        self.ay = 1.0 - self.cy - self.by

        self.epsilon = 1e-6;  # Precision    

    # Find new T as a function of Y along curve X
    def solve(self, x):
        return self.sampleCurveY(self.solveCurveX(x))

    def sampleCurveX(self, t):
        return ((self.ax * t + self.bx) * t + self.cx) * t

    def sampleCurveY(self, t):
        return ((self.ay * t + self.by) * t + self.cy) * t

    def sampleCurveDerivativeX(self, t):
        return (3.0 * self.ax * t + 2.0 * self.bx) * t + self.cx

    def solveCurveX(self, x):

        # First try a few iterations of Newton's method -- normally very fast.
        t2 = x
        for i in range(8):
            x2 = self.sampleCurveX(t2) - x;
            if np.abs(x2) < self.epsilon:
                return t2
            d2 = self.sampleCurveDerivativeX(t2);
            if np.abs(d2) < self.epsilon:
                break;
            t2 = t2 - x2 / d2;

        # no solution found - use bi-section
        t0 = 0.0;
        t1 = 1.0;
        t2 = x;

        if (t2 < t0): return t0;
        if (t2 > t1): return t1;

        while (t0 < t1):
            x2 = self.sampleCurveX(t2);
            if (np.abs(x2 - x) < self.epsilon):
                return t2;
            if (x > x2):
                t0 = t2;
            else:
                t1 = t2;

            t2 = (t1 - t0) * .5 + t0;

        # Give up
        return t2;
