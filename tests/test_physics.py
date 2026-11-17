"""Run with:  python -m pytest -q tests   (from the project folder)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import analysis as an, physics_engine as pe, potentials as pot, experiments as ex


def test_norm_and_energy_conserved():
    r = pe.simulate(pe.SimConfig.make("rect_barrier", dict(V0=2.5, width=1.5)))
    assert np.abs(r.step_norm - 1).max() < 1e-10
    assert np.ptp(r.energy) < 1e-10
    assert abs(r.R[-1] + r.T[-1] + r.B[-1] - 1) < 1e-10


def test_minimum_uncertainty_initial_packet():
    r = pe.simulate(pe.SimConfig.make("free", {}))
    assert abs(r.delta_x[0] * r.delta_p[0] - 0.5) < 1e-6


def test_transfer_matrix_matches_closed_form():
    x = pe.make_grid(-20, 20, 0.005)
    E = np.linspace(0.1, 6, 60)
    T = an.transmission_vs_energy(x, pot.rectangular_barrier(x, 2.5, 1.5), E)
    assert np.abs(T - an.rect_barrier_transmission_analytic(E, 2.5, 1.5)).max() < 1e-3


def test_second_order_convergence():
    df = an.convergence_study("dx")
    assert df["observed_order"].iloc[-1] > 1.8


def test_packet_T_close_to_plane_wave_for_wide_packet():
    out = ex.run_scattering("rect_barrier", dict(V0=2.5, width=1.5, center=0.0), 2.0, 6.0)
    assert abs(out["T"] - out["T_plane_wave"]) < 0.02


def test_resonances_found_for_double_barrier():
    x = pe.make_grid(-15, 15, 0.02)
    V = pot.double_barrier(x, 3, 0.8, 3, 0.8, 4.0)
    E = np.linspace(0.02, 3.5, 1500)
    peaks = an.find_resonances(x, V, E, an.transmission_vs_energy(x, V, E))
    assert len(peaks) >= 3 and (peaks["T_peak"] > 0.99).all()
