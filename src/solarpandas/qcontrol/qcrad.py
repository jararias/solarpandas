
from .helpers import QCTest
from . import ppl, erl, Kspace

ghi_ppl = QCTest(name="ghi_ppl", _test_func=ppl.test_ghi, _plot_func=ppl.plot_test_ghi)
dif_ppl = QCTest(name="dif_ppl", _test_func=ppl.test_dif, _plot_func=ppl.plot_test_dif)
dni_ppl = QCTest(name="dni_ppl", _test_func=ppl.test_dni, _plot_func=ppl.plot_test_dni)

ghi_erl = QCTest(name="ghi_erl", _test_func=erl.test_ghi, _plot_func=erl.plot_test_ghi)
dif_erl = QCTest(name="dif_erl", _test_func=erl.test_dif, _plot_func=erl.plot_test_dif)
dni_erl = QCTest(name="dni_erl", _test_func=erl.test_dni, _plot_func=erl.plot_test_dni)

Kn_ppl = QCTest(name="Kn_ppl", _test_func=Kspace.test_Kn_ppl, _plot_func=Kspace.plot_test_Kn_ppl)
Kn_erl = QCTest(name="Kn_erl", _test_func=Kspace.test_Kn_erl, _plot_func=Kspace.plot_test_Kn_erl)
KT_erl = QCTest(name="KT_erl", _test_func=Kspace.test_KT_erl, _plot_func=Kspace.plot_test_KT_erl)
K_erl = QCTest(name="K_erl", _test_func=Kspace.test_K_erl, _plot_func=Kspace.plot_test_K_erl)
K_erl_clear = QCTest(name="K_erl_clear", _test_func=Kspace.test_K_erl_clear, _plot_func=Kspace.plot_test_K_erl_clear)
