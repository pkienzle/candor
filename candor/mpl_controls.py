class SliderSet:
    def __init__(self, n, update, query=None):
        self._handles = {}
        self._connectors = {}
        self.k = 0
        self.n = n
        self.update = update
        self.query = query

    def add(self, axis, limits=None, value=None, label=None):
        # Delayed import, in case we are not running with matplotlib available
        from matplotlib.widgets import Slider
        from matplotlib import pyplot
        if self.k >= self.n:
            raise RuntimeError("too many sliders")
        if label is None:
            label = axis
        if value is None:
            value = limits[0]
        self.k += 1
        ax = pyplot.subplot(self.n, 2, 2*self.k)
        slider = Slider(ax, label, limits[0], limits[1], valinit=value)
        slider.on_changed(lambda v: self.update(axis, v))
        self._handles[axis] = (ax, slider)

    @staticmethod
    def _no_update(axis, value):
        pass

    def reset(self):
        # suppress updates during reset
        # Note: should be able to do this by setting slider.active to False or
        # disconnecting events during reset, then reactivating when the loop
        # is complete but neither method was working.
        cached_update = self.update
        self.update = self._no_update
        for axis, (ax, slider) in self._handles.items():
            # Only update sliders that have changed.
            new_val = self.query(axis)
            if slider.val != new_val:
                #print(axis, "new value", new_val, slider.active)
                slider.set_val(new_val)
        self.update = cached_update

def get_zoom(ax):
    if ax.get_autoscale_on():
        return ()
    view = ax.viewLim.get_points()
    data = ax.dataLim.get_points()
    if (view[0] <= data[0]).all() and (view[1] >= data[1]).all():
        return ()
    return ax.axis()

def set_zoom(ax, limits):
    if limits:
        ax.figure.canvas.manager.toolbar.push_current()
        ax.axis(limits)
