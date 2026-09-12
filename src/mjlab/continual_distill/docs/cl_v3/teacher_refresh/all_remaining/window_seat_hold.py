"""Keep seating compensation while closing on the window handle."""
from mjlab.continual_distill.classical.slide_window import SlideWindowClassicalPolicy as Original
class SlideWindowClassicalPolicy(Original):
  def _go(self,i,phase):
    integral=self._integral[i].copy()
    old=int(self._phase[i])
    super()._go(i,phase)
    if old==1 and phase==2:self._integral[i]=integral
