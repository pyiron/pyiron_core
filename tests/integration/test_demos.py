import contextlib
import dataclasses
import pathlib
import unittest
import urllib

import numpy as np
import pandas as pd
import pandas.testing as pdt
from scipy import stats

try:
    import pyace
except ImportError:
    pyace = None

import pyiron_core as pc
from pyiron_core import nodes

DEMOS_DIR = pathlib.Path(__file__).parent.parent.parent / "stored_workflows" / "demos"
ALL_DEMOS = list(f.stem for f in DEMOS_DIR.glob("*.json"))


class TestDemoWorkflows(unittest.TestCase):
    @contextlib.contextmanager
    def _download_then_delete(self, url: str) -> None:
        fname = url.split("/")[-1]
        try:
            with urllib.request.urlopen(url) as response:
                with open(fname, "wb") as f:
                    f.write(response.read())
            yield None
        finally:
            pathlib.Path(fname).unlink()

    def _mock_run(self, wf_name: str, node_name: str, port_name: str):
        # Instantiate the gui like a user
        pf = pc.PyironFlow([wf_name], workflow_path=DEMOS_DIR, load_from_compact=True)
        # Mock a "run" click at the PyironFlowWidget level
        pf.wf_widgets[0].on_value_change(
            {"new": f"run: {node_name}", "old": None, "name": "mock"}
        )
        # The result gets passed directly to a display widget, and is hard to parse
        # As a fallback, inspect the underlying graph
        node_instance = pf.wf_widgets[0].graph.nodes[node_name].node
        output_value = getattr(node_instance.outputs, port_name).value
        # Return the target data
        return output_value

    def test_loading(self):
        pc.PyironFlow(
            ALL_DEMOS,
            workflow_path=DEMOS_DIR,
            load_from_compact=True,
        )

    def test_energy(self):
        output = self._mock_run("energy", "GetEnergyLast", "energy_last")
        reference_energy = -0.0015020475862299598
        self.assertAlmostEqual(output, reference_energy)

    @unittest.skipIf(
        pyace is None, "pyace not available -- skipping grace_minimize test"
    )
    def test_grace_minimize(self):
        output = self._mock_run("grace_minimize", "Minimize", "out")
        reference_energies = [-398.1614752680032, -398.1970445514807]
        reference_itersteps = 8

        self.assertTrue(output.is_converged)
        self.assertEqual(output.iter_steps, reference_itersteps)
        self.assertTrue(
            np.allclose(reference_energies, output.energies_pot),
            msg=f"expected {reference_energies}\ngot {output.energies_pot}"
        )


    def test_landau(self):
        with self._download_then_delete(
                "https://github.com/pyiron-workshop/DPG-tutorial-2025/raw/351eeca736cce45f9bc3bfca84ab05de049e38c2/data/MgCaFreeEnergies.pckl.gz"
        ) as _:
            output = self._mock_run("landau", "CalcPhaseDiagram", "df")
        tail = output.tail(3)

        # And compare it to results grabbed from a manual run where the plot is ok
        expected = pd.DataFrame({
            'T': [528.781028, 528.781028, 528.781028],
            'phase': ['fcc Ca', 'liquid', 'Mg$_2$Ca'],  # Note the newline in the third entry
            'phi': [-1.78757, -1.78757, -1.78757],
            'mu': [-0.316953, -0.316953, -0.316953],
            'c': [1.000000, 0.721217, 0.333333],
            'stable': [True, True, True],
            'border': [True, True, True],
            'refined': ['delaunay-triple', 'delaunay-triple', 'delaunay-triple'],
            'f': [-2.104523, -2.016162, -1.893221],
            'f_excess': [0.000000, 0.029454, 0.070434]
        })


        expected_sorted = expected.sort_values(by=list(expected.columns)).reset_index(drop=True)
        actual_sorted = tail.sort_values(by=list(tail.columns)).reset_index(drop=True)

        pdt.assert_frame_equal(
            expected_sorted,
            actual_sorted,
            rtol=1e-3,
            atol=1e-4,
            check_dtype=False
        )

    @unittest.skipIf(
        pyace is None, "pyace not available -- skipping linearfit test"
    )
    def test_linearfit(self):
        with self._download_then_delete(
                "https://github.com/pyiron-workshop/DPG-tutorial-2025/raw/351eeca736cce45f9bc3bfca84ab05de049e38c2/data/mgca.pckl.tgz"
        ) as _:
            output = self._mock_run("linearfit", "PredictEnergiesAndForces", "data_dict")

        reference = {  # Last items of each entry from a manual run
            "reference_training_epa": np.array([-1.6219105069933333, -1.6426530701, -1.4760515700066668, -1.449002318375, -1.204447103705]),
            "reference_training_fpa": np.array([-0.00052099, -3.075e-05, 5.55e-06, 0.00052099, 3.075e-05]),
            "predicted_training_epa": np.array([-1.782124353671963, -1.709263676346597, -1.6974051358740896, -1.6446346861135153, -1.1999858821779013]),
            "predicted_training_fpa": np.array([-0.0006600464048357077, -5.380634712150788e-05, -4.127741166488192e-06, 0.0006600464048357077, 5.38063471215287e-05]),
            "reference_testing_epa": np.array([-0.48387109904111114, -0.3011008274525, -0.8320895942100001, -0.257647389155, -0.27296485680125]),
            "reference_testing_fpa": np.array([0.06379655, -1.23912384, -0.05524943, -0.03189827, -1.23912384]),
            "predicted_testing_epa": np.array([-0.2911790725598916, -0.23546299226525025, -0.7422808416362213, -0.16393019733556505, -0.09089519915920707]),
            "predicted_testing_fpa": np.array([-0.026306930659412288, -1.0806129186000701, 0.02278246992192133, 0.013153465125682134, -1.0806129185952347]),
        }

        for key, data in output.items():
            expected = reference[key]
            actual = data[-len(expected):]
            with self.subTest(key):
                self.assertTrue(
                    np.allclose(expected, actual),
                    msg=f"expected {expected}\ngot {actual}"
                )

    def test_phonons(self):
        output = self._mock_run("phonons", "GetThermalProperties", "thermal_properties")

        reference = nodes.atomistic.property.phonons.ThermalProperties().dataclass(
            temperatures=np.array([0., 10., 20., 30., 40., 50., 60., 70., 80.,
                                                  90., 100., 110., 120., 130., 140., 150., 160., 170.,
                                                  180., 190., 200., 210., 220., 230., 240., 250., 260.,
                                                  270., 280., 290., 300., 310., 320., 330., 340., 350.,
                                                  360., 370., 380., 390., 400., 410., 420., 430., 440.,
                                                  450., 460., 470., 480., 490., 500., 510., 520., 530.,
                                                  540., 550., 560., 570., 580., 590., 600., 610., 620.,
                                                  630., 640., 650., 660., 670., 680., 690., 700., 710.,
                                                  720., 730., 740., 750., 760., 770., 780., 790., 800.,
                                                  810., 820., 830., 840., 850., 860., 870., 880., 890.,
                                                  900., 910., 920., 930., 940., 950., 960., 970., 980.,
                                                  990., 1000.]),
                              free_energy=np.array([3.44449556, 3.44444836, 3.44370675, 3.44022045,
                                                 3.43050864, 3.41078358, 3.37792925, 3.32978767,
                                                 3.26506814, 3.18314467, 3.08385798, 2.96735809,
                                                 2.83399016, 2.68421554, 2.51855892, 2.33757361,
                                                 2.14181921, 1.93184752, 1.70819401, 1.47137294,
                                                 1.22187474, 0.96016505, 0.68668468, 0.4018502,
                                                 0.10605487, -0.20033019, -0.51695479, -0.84348831,
                                                 -1.17961856, -1.52505044, -1.87950488, -2.24271768,
                                                 -2.61443853, -2.99443005, -3.38246688, -3.77833492,
                                                 -4.18183053, -4.59275983, -5.01093811, -5.43618918,
                                                 -5.86834485, -6.30724444, -6.75273428, -7.20466729,
                                                 -7.66290261, -8.12730519, -8.59774548, -9.07409911,
                                                 -9.55624656, -10.04407296, -10.53746774, -11.0363245,
                                                 -11.5405407, -12.05001751, -12.56465959, -13.08437494,
                                                 -13.60907471, -14.13867303, -14.67308691, -15.21223607,
                                                 -15.75604281, -16.3044319, -16.85733047, -17.41466789,
                                                 -17.97637567, -18.54238738, -19.11263856, -19.6870666,
                                                 -20.26561071, -20.84821182, -21.43481251, -22.02535693,
                                                 -22.61979077, -23.21806117, -23.82011668, -24.42590717,
                                                 -25.03538383, -25.64849911, -26.26520663, -26.88546118,
                                                 -27.50921866, -28.13643605, -28.76707135, -29.40108359,
                                                 -30.03843272, -30.67907965, -31.32298618, -31.97011498,
                                                 -32.62042956, -33.27389424, -33.93047411, -34.59013505,
                                                 -35.25284365, -35.91856721, -36.58727374, -37.25893191,
                                                 -37.93351103, -38.61098103, -39.29131247, -39.97447649,
                                                 -40.66044481]),
                              entropy=np.array([0.00000000e+00, 1.83206561e+00, 1.57239945e+01, 5.73898303e+01,
                                             1.36272712e+02, 2.49409659e+02, 3.88032246e+02, 5.42999706e+02,
                                             7.06871528e+02, 8.74241998e+02, 1.04144048e+03, 1.20609800e+03,
                                             1.36676163e+03, 1.52260002e+03, 1.67319350e+03, 1.81838988e+03,
                                             1.95820716e+03, 2.09276871e+03, 2.22226045e+03, 2.34690272e+03,
                                             2.46693188e+03, 2.58258862e+03, 2.69411045e+03, 2.80172719e+03,
                                             2.90565834e+03, 3.00611167e+03, 3.10328270e+03, 3.19735463e+03,
                                             3.28849871e+03, 3.37687469e+03, 3.46263146e+03, 3.54590772e+03,
                                             3.62683265e+03, 3.70552661e+03, 3.78210177e+03, 3.85666272e+03,
                                             3.92930709e+03, 4.00012603e+03, 4.06920476e+03, 4.13662301e+03,
                                             4.20245544e+03, 4.26677204e+03, 4.32963847e+03, 4.39111641e+03,
                                             4.45126385e+03, 4.51013534e+03, 4.56778229e+03, 4.62425314e+03,
                                             4.67959363e+03, 4.73384694e+03, 4.78705392e+03, 4.83925321e+03,
                                             4.89048139e+03, 4.94077315e+03, 4.99016140e+03, 5.03867738e+03,
                                             5.08635077e+03, 5.13320980e+03, 5.17928132e+03, 5.22459090e+03,
                                             5.26916293e+03, 5.31302064e+03, 5.35618621e+03, 5.39868082e+03,
                                             5.44052471e+03, 5.48173723e+03, 5.52233690e+03, 5.56234142e+03,
                                             5.60176778e+03, 5.64063224e+03, 5.67895040e+03, 5.71673723e+03,
                                             5.75400710e+03, 5.79077381e+03, 5.82705060e+03, 5.86285025e+03,
                                             5.89818500e+03, 5.93306666e+03, 5.96750659e+03, 6.00151573e+03,
                                             6.03510463e+03, 6.06828346e+03, 6.10106203e+03, 6.13344979e+03,
                                             6.16545587e+03, 6.19708911e+03, 6.22835801e+03, 6.25927080e+03,
                                             6.28983546e+03, 6.32005967e+03, 6.34995089e+03, 6.37951631e+03,
                                             6.40876292e+03, 6.43769748e+03, 6.46632652e+03, 6.49465639e+03,
                                             6.52269325e+03, 6.55044305e+03, 6.57791157e+03, 6.60510444e+03,
                                             6.63202709e+03]),
                              heat_capacity=np.array([0., 5.5584809, 50.23800156, 179.64171083,
                                                   388.95227005, 637.42780152, 889.23745386, 1123.39866139,
                                                   1330.90586861, 1509.96168943, 1662.3510954, 1791.2691535,
                                                   1900.19406985, 1992.37592759, 2070.64906398, 2137.39966629,
                                                   2194.59989434, 2243.86485232, 2286.51237222, 2323.61737441,
                                                   2356.05818506, 2384.55469173, 2409.69920712, 2431.9811955,
                                                   2451.80699349, 2469.51551907, 2485.39079476, 2499.67195305,
                                                   2512.56125446, 2524.23053626, 2534.82641991, 2544.47453354,
                                                   2553.28295069, 2561.34500285, 2568.74158987, 2575.54308588,
                                                   2581.81091828, 2587.59888114, 2592.95423206, 2597.91861186,
                                                   2602.5288185, 2606.81746075, 2610.81351231, 2614.54278308,
                                                   2618.02832132, 2621.29075793, 2624.34860209, 2627.21849592,
                                                   2629.91543439, 2632.45295588, 2634.84330761, 2637.09758975,
                                                   2639.22588122, 2641.23734974, 2643.14034843, 2644.94250065,
                                                   2646.65077485, 2648.27155061, 2649.81067707, 2651.27352476,
                                                   2652.66503166, 2653.98974412, 2655.25185344, 2656.45522851,
                                                   2657.60344495, 2658.69981136, 2659.74739277, 2660.74903178,
                                                   2661.7073676, 2662.6248532, 2663.50377082, 2664.34624596,
                                                   2665.15426015, 2665.92966239, 2666.67417967, 2667.38942645,
                                                   2668.07691332, 2668.73805493, 2669.37417716, 2669.98652371,
                                                   2670.57626213, 2671.14448936, 2671.6922367, 2672.22047457,
                                                   2672.73011667, 2673.22202395, 2673.6970082, 2674.15583538,
                                                   2674.59922869, 2675.02787138, 2675.44240939, 2675.84345371,
                                                   2676.23158271, 2676.60734412, 2676.97125699, 2677.32381349,
                                                   2677.6654805, 2677.99670123, 2678.31789658, 2678.6294665,
                                                   2678.93179121])
        )
        for field in dataclasses.fields(reference):
            with self.subTest(field):
                expected = getattr(reference, field.name)
                got = getattr(output, field.name)
                self.assertTrue(np.allclose(expected, got), msg=f"expected {expected}\ngot {got}")

    def test_water(self):
        output = self._mock_run("water", "Equilibrate", "out")

        reference_cell = np.array([[1.71826948e+01, 1.05213661e-15, 1.05213661e-15],
           [0.00000000e+00, 1.98408669e+01, 1.21490271e-15],
           [0.00000000e+00, 0.00000000e+00, 4.16913430e+01]])
        reference_initial_energy = 31.46838306
        n_equilibration_steps = 10
        reference_energy = np.array([ 31.46838306,  -4.26942597,  -8.0805482 , -12.90076907,
           -16.59626162, -18.47873051, -18.93215519, -18.67742173,
           -15.40624307, -16.3339607 , -15.76489409, -16.9078756 ,
           -17.66525076, -16.80113781, -15.2462573 , -14.0982591 ,
           -14.18508625, -13.6224965 , -14.5188483 , -12.92224425,
           -12.56978264, -13.1520431 , -12.56133904, -14.46415631,
           -13.17798482, -11.97324439, -11.19910964, -11.90843353,
           -12.09170554, -12.8062513 , -12.85819809, -10.6408585 ,
           -11.49414487, -10.38585455, -10.26564317, -13.25688849,
           -11.77704207, -12.00399434, -11.03770623, -10.99755484,
           -11.22154936, -11.34856006, -12.161935  , -11.74241786,
           -11.00740265, -11.26740768, -12.61250859, -10.43336029,
           -12.64427893, -13.44524596, -12.58108501, -11.86740602,
           -10.17643539, -10.55642178,  -8.13795718,  -8.92421519,
           -11.08234688,  -9.19346724,  -9.45115114,  -9.0158142 ,
            -8.14580283,  -8.88854698, -10.55075374,  -8.21890427,
            -9.11128619,  -8.21918011,  -7.46283417,  -8.04671325,
            -9.83082643, -11.59542109,  -9.38027939,  -8.74295339,
            -8.76248409,  -9.03951998,  -8.97056175,  -9.59879561,
           -11.30991291,  -9.14053122,  -8.60409889, -10.40048267,
            -9.87544651,  -9.00918605,  -9.58545053, -10.30768069,
            -8.91142289, -10.37895725,  -9.60720873,  -9.40381222,
           -11.48253315,  -8.87587232,  -8.6542851 , -10.44444786,
           -11.08185074, -10.05379793, -10.43616253,  -8.2677656 ,
            -9.8360291 ,  -9.38505847, -10.70454689, -11.89606576,
           -11.70837562
        ])

        self.assertTrue(
            np.allclose(reference_cell, output.cells[0]),
            msg=f"expected\n{reference_cell}\ngot\n{output.cells[0]}"
        )
        self.assertAlmostEqual(reference_initial_energy, output.energies_pot[0])

        # Finally, a bit of a physics comparison
        # What is the likelihood that the potential energies we obtained were drawn
        # from the same distribution as the reference data?
        p_same_distribution = stats.mannwhitneyu(
            reference_energy[n_equilibration_steps:],
            output.energies_pot[n_equilibration_steps:],
            alternative="two-sided",
        ).pvalue
        six_sigma_p = 2.0e-09  # I didn't verify this, I just got ChatGPT to give me
        # a table relating two-sided alternative Mann-Whitney U p-values to standard
        # deviations of likelihood.
        # A couple by-hand tests indicate that the odds of two samples of the same
        # water simulation appear very unlikely to be from the same distribution
        # (p = 1e-05 to 1e-06 seems to be the ballpark), but different physics (e.g.
        # e.g. comparing total energy to potential energy) or complete nonsense (e.g.
        # all 1's) gives p-values of 1e-20 and beyond.
        # so let's be pretty generous and say we demand a 6-sigma result to claim the
        # null hypothesis has been broken and the test has failed.
        self.assertGreater(p_same_distribution, six_sigma_p)
