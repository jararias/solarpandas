
from .helpers import QCTest

from . import ppl

ghi_ppl = QCTest(name="ghi_ppl", _test_func=ppl.test_ghi, _plot_func=ppl.plot_test_ghi)
dif_ppl = QCTest(name="dif_ppl", _test_func=ppl.test_dif, _plot_func=ppl.plot_test_dif)
dni_ppl = QCTest(name="dni_ppl", _test_func=ppl.test_dni, _plot_func=ppl.plot_test_dni)
