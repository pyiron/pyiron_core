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
ALL_DEMOS = [f.stem for f in DEMOS_DIR.glob("*.json")]


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
        pf = pc.PyironFlow([wf_name], workflow_path=DEMOS_DIR)
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

    def test_calphy(self):
        pf = pc.PyironFlow(["calphy"], workflow_path=DEMOS_DIR)

        # If we run the calphy as-is, it melts too often
        # (The default stopping temperature in the calphy InputData dataclass is 600k,
        # and the example runs aluminum)
        # So we manually lower the temperature first
        pfw = pf.wf_widgets[0]
        pfw.graph.nodes["InputClass"].node.to_inputs(temperature_stop=450)

        # And then run it as usual
        node_name = "SolidFreeEnergyWithTemperature"
        port_name = "f"
        pf.wf_widgets[0].on_value_change(
            {"new": f"run: {node_name}", "old": None, "name": "mock"}
        )
        node_instance = pf.wf_widgets[0].graph.nodes[node_name].node
        output_value = getattr(node_instance.outputs, port_name).value

        reference_f = np.load(
            pathlib.Path(__file__).parent.parent
            / "static"
            / "calphy_reference_f_450K.npy"
        )
        # This is still stochastic, so don't be too strict on the comparison
        self.assertTrue(np.allclose(reference_f, output_value, atol=0.02, rtol=0.008))

    @unittest.skipIf(
        True,
        "The elastic demo is not running -- https://github.com/pyiron/pyiron_core/issues/119",
    )
    def test_elastic(self):
        raise NotImplementedError()

    def test_energy(self):
        output = self._mock_run("energy", "GetEnergyLast", "energy_last")
        reference_energy = -0.0015020475862299598
        self.assertAlmostEqual(output, reference_energy)

    def test_grace_minimize(self):
        output = self._mock_run("grace_minimize", "Minimize", "out")
        reference_energies = [-398.1614752680032, -398.1970445514807]
        reference_itersteps = 8

        self.assertTrue(output.is_converged)
        self.assertEqual(output.iter_steps, reference_itersteps)
        self.assertTrue(
            np.allclose(reference_energies, output.energies_pot),
            msg=f"expected {reference_energies}\ngot {output.energies_pot}",
        )

    def test_landau(self):
        with self._download_then_delete(
            "https://github.com/pyiron-workshop/DPG-tutorial-2025/raw/351eeca736cce45f9bc3bfca84ab05de049e38c2/data/MgCaFreeEnergies.pckl.gz"
        ) as _:
            output = self._mock_run("landau", "CalcPhaseDiagram", "df")
        tail = output.tail(3)

        # And compare it to results grabbed from a manual run where the plot is ok
        expected = pd.DataFrame(
            {
                "T": [528.781028, 528.781028, 528.781028],
                "phase": [
                    "fcc Ca",
                    "liquid",
                    "Mg$_2$Ca",
                ],  # Note the newline in the third entry
                "phi": [-1.78757, -1.78757, -1.78757],
                "mu": [-0.316953, -0.316953, -0.316953],
                "c": [1.000000, 0.721217, 0.333333],
                "stable": [True, True, True],
                "border": [True, True, True],
                "refined": ["delaunay-triple", "delaunay-triple", "delaunay-triple"],
                "f": [-2.104523, -2.016162, -1.893221],
                "f_excess": [0.000000, 0.029454, 0.070434],
            }
        )

        expected_sorted = expected.sort_values(by=list(expected.columns)).reset_index(
            drop=True
        )
        actual_sorted = tail.sort_values(by=list(tail.columns)).reset_index(drop=True)

        pdt.assert_frame_equal(
            expected_sorted, actual_sorted, rtol=1e-3, atol=1e-4, check_dtype=False
        )

    def test_linearfit(self):
        with self._download_then_delete(
            "https://github.com/pyiron-workshop/DPG-tutorial-2025/raw/351eeca736cce45f9bc3bfca84ab05de049e38c2/data/mgca.pckl.tgz"
        ) as _:
            output = self._mock_run(
                "linearfit", "PredictEnergiesAndForces", "data_dict"
            )

        reference = {  # Last items of each entry from a manual run
            "reference_training_epa": np.array(
                [
                    -1.6219105069933333,
                    -1.6426530701,
                    -1.4760515700066668,
                    -1.449002318375,
                    -1.204447103705,
                ]
            ),
            "reference_training_fpa": np.array(
                [-0.00052099, -3.075e-05, 5.55e-06, 0.00052099, 3.075e-05]
            ),
            "predicted_training_epa": np.array(
                [
                    -1.782124353671963,
                    -1.709263676346597,
                    -1.6974051358740896,
                    -1.6446346861135153,
                    -1.1999858821779013,
                ]
            ),
            "predicted_training_fpa": np.array(
                [
                    -0.0006600464048357077,
                    -5.380634712150788e-05,
                    -4.127741166488192e-06,
                    0.0006600464048357077,
                    5.38063471215287e-05,
                ]
            ),
            "reference_testing_epa": np.array(
                [
                    -0.48387109904111114,
                    -0.3011008274525,
                    -0.8320895942100001,
                    -0.257647389155,
                    -0.27296485680125,
                ]
            ),
            "reference_testing_fpa": np.array(
                [0.06379655, -1.23912384, -0.05524943, -0.03189827, -1.23912384]
            ),
            "predicted_testing_epa": np.array(
                [
                    -0.2911790725598916,
                    -0.23546299226525025,
                    -0.7422808416362213,
                    -0.16393019733556505,
                    -0.09089519915920707,
                ]
            ),
            "predicted_testing_fpa": np.array(
                [
                    -0.026306930659412288,
                    -1.0806129186000701,
                    0.02278246992192133,
                    0.013153465125682134,
                    -1.0806129185952347,
                ]
            ),
        }

        for key, data in output.items():
            expected = reference[key]
            actual = data[-len(expected) :]
            with self.subTest(key):
                self.assertTrue(
                    np.allclose(expected, actual),
                    msg=f"expected {expected}\ngot {actual}",
                )

    def test_phonons(self):
        output = self._mock_run("phonons", "GetThermalProperties", "thermal_properties")

        reference = nodes.atomistic.property.phonons.ThermalProperties().dataclass(
            temperatures=np.array(
                [
                    0.0,
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                    70.0,
                    80.0,
                    90.0,
                    100.0,
                    110.0,
                    120.0,
                    130.0,
                    140.0,
                    150.0,
                    160.0,
                    170.0,
                    180.0,
                    190.0,
                    200.0,
                    210.0,
                    220.0,
                    230.0,
                    240.0,
                    250.0,
                    260.0,
                    270.0,
                    280.0,
                    290.0,
                    300.0,
                    310.0,
                    320.0,
                    330.0,
                    340.0,
                    350.0,
                    360.0,
                    370.0,
                    380.0,
                    390.0,
                    400.0,
                    410.0,
                    420.0,
                    430.0,
                    440.0,
                    450.0,
                    460.0,
                    470.0,
                    480.0,
                    490.0,
                    500.0,
                    510.0,
                    520.0,
                    530.0,
                    540.0,
                    550.0,
                    560.0,
                    570.0,
                    580.0,
                    590.0,
                    600.0,
                    610.0,
                    620.0,
                    630.0,
                    640.0,
                    650.0,
                    660.0,
                    670.0,
                    680.0,
                    690.0,
                    700.0,
                    710.0,
                    720.0,
                    730.0,
                    740.0,
                    750.0,
                    760.0,
                    770.0,
                    780.0,
                    790.0,
                    800.0,
                    810.0,
                    820.0,
                    830.0,
                    840.0,
                    850.0,
                    860.0,
                    870.0,
                    880.0,
                    890.0,
                    900.0,
                    910.0,
                    920.0,
                    930.0,
                    940.0,
                    950.0,
                    960.0,
                    970.0,
                    980.0,
                    990.0,
                    1000.0,
                ]
            ),
            free_energy=np.array(
                [
                    1.01957963,
                    1.01955688,
                    1.0192582,
                    1.01803825,
                    1.01489062,
                    1.008729,
                    0.99865344,
                    0.98403665,
                    0.9645021,
                    0.93986716,
                    0.91008604,
                    0.87520389,
                    0.83532335,
                    0.79058131,
                    0.74113335,
                    0.68714347,
                    0.62877751,
                    0.56619898,
                    0.49956657,
                    0.42903264,
                    0.35474252,
                    0.27683418,
                    0.19543823,
                    0.11067809,
                    0.0226703,
                    -0.06847519,
                    -0.16265462,
                    -0.25977003,
                    -0.35972891,
                    -0.46244383,
                    -0.5678321,
                    -0.67581543,
                    -0.78631968,
                    -0.89927449,
                    -1.01461309,
                    -1.13227206,
                    -1.25219104,
                    -1.3743126,
                    -1.498582,
                    -1.62494705,
                    -1.7533579,
                    -1.88376695,
                    -2.01612867,
                    -2.15039947,
                    -2.28653763,
                    -2.42450312,
                    -2.56425756,
                    -2.7057641,
                    -2.84898734,
                    -2.99389322,
                    -3.14044899,
                    -3.28862313,
                    -3.43838525,
                    -3.58970606,
                    -3.74255733,
                    -3.89691179,
                    -4.05274312,
                    -4.21002586,
                    -4.36873544,
                    -4.52884807,
                    -4.69034072,
                    -4.8531911,
                    -5.01737763,
                    -5.18287937,
                    -5.34967603,
                    -5.51774793,
                    -5.68707596,
                    -5.85764159,
                    -6.02942679,
                    -6.20241407,
                    -6.37658641,
                    -6.55192726,
                    -6.72842053,
                    -6.90605056,
                    -7.0848021,
                    -7.26466031,
                    -7.44561071,
                    -7.62763923,
                    -7.81073211,
                    -7.99487596,
                    -8.18005771,
                    -8.36626463,
                    -8.55348426,
                    -8.74170446,
                    -8.93091338,
                    -9.12109942,
                    -9.31225126,
                    -9.50435786,
                    -9.69740838,
                    -9.89139227,
                    -10.08629918,
                    -10.282119,
                    -10.47884183,
                    -10.67645798,
                    -10.87495798,
                    -11.07433253,
                    -11.27457255,
                    -11.47566913,
                    -11.67761354,
                    -11.88039724,
                    -12.08401184,
                ]
            ),
            entropy=np.array(
                [
                    0.00000000e00,
                    8.43070418e-01,
                    5.94634385e00,
                    1.92730463e01,
                    4.32590090e01,
                    7.71007183e01,
                    1.18331577e02,
                    1.64325223e02,
                    2.12918568e02,
                    2.62529751e02,
                    3.12080607e02,
                    3.60873783e02,
                    4.08481050e02,
                    4.54657235e02,
                    4.99278607e02,
                    5.42300397e02,
                    5.83728120e02,
                    6.23598440e02,
                    6.61966505e02,
                    6.98897630e02,
                    7.34461875e02,
                    7.68730565e02,
                    8.01774085e02,
                    8.33660534e02,
                    8.64454950e02,
                    8.94218897e02,
                    9.23010308e02,
                    9.50883469e02,
                    9.77889117e02,
                    1.00407459e03,
                    1.02948400e03,
                    1.05415844e03,
                    1.07813619e03,
                    1.10145292e03,
                    1.12414185e03,
                    1.14623398e03,
                    1.16775824e03,
                    1.18874162e03,
                    1.20920939e03,
                    1.22918517e03,
                    1.24869107e03,
                    1.26774784e03,
                    1.28637493e03,
                    1.30459061e03,
                    1.32241208e03,
                    1.33985548e03,
                    1.35693606e03,
                    1.37366816e03,
                    1.39006534e03,
                    1.40614039e03,
                    1.42190543e03,
                    1.43737188e03,
                    1.45255060e03,
                    1.46745186e03,
                    1.48208542e03,
                    1.49646052e03,
                    1.51058597e03,
                    1.52447013e03,
                    1.53812095e03,
                    1.55154601e03,
                    1.56475253e03,
                    1.57774741e03,
                    1.59053721e03,
                    1.60312820e03,
                    1.61552640e03,
                    1.62773751e03,
                    1.63976704e03,
                    1.65162023e03,
                    1.66330212e03,
                    1.67481751e03,
                    1.68617104e03,
                    1.69736714e03,
                    1.70841007e03,
                    1.71930390e03,
                    1.73005258e03,
                    1.74065989e03,
                    1.75112944e03,
                    1.76146475e03,
                    1.77166917e03,
                    1.78174595e03,
                    1.79169822e03,
                    1.80152899e03,
                    1.81124115e03,
                    1.82083753e03,
                    1.83032081e03,
                    1.83969362e03,
                    1.84895848e03,
                    1.85811783e03,
                    1.86717402e03,
                    1.87612935e03,
                    1.88498600e03,
                    1.89374613e03,
                    1.90241179e03,
                    1.91098499e03,
                    1.91946767e03,
                    1.92786171e03,
                    1.93616892e03,
                    1.94439109e03,
                    1.95252991e03,
                    1.96058705e03,
                    1.96856414e03,
                ]
            ),
            heat_capacity=np.array(
                [
                    0.0,
                    2.36694328,
                    17.09079368,
                    55.64016204,
                    117.01244941,
                    189.9697677,
                    264.1249891,
                    333.23293926,
                    394.55967104,
                    447.52387866,
                    492.62497987,
                    530.79300622,
                    563.04929665,
                    590.35170041,
                    613.53712949,
                    633.31094797,
                    650.25653557,
                    664.85186942,
                    677.4870723,
                    688.48042631,
                    698.09204317,
                    706.53513842,
                    713.98515774,
                    720.58709214,
                    726.46131429,
                    731.70822875,
                    736.4119799,
                    740.64341472,
                    744.46245742,
                    747.92001945,
                    751.05954194,
                    753.91824646,
                    756.52815373,
                    758.9169168,
                    761.10850547,
                    763.12377103,
                    764.98091393,
                    766.69587303,
                    768.2826505,
                    769.75358429,
                    771.11957749,
                    772.39029194,
                    773.57431245,
                    774.67928642,
                    775.71204297,
                    776.67869499,
                    777.5847267,
                    778.43506912,
                    779.23416522,
                    779.98602641,
                    780.6942815,
                    781.36221946,
                    781.99282665,
                    782.58881944,
                    783.15267286,
                    783.68664568,
                    784.19280266,
                    784.67303404,
                    785.12907291,
                    785.56251056,
                    785.9748101,
                    786.3673186,
                    786.74127794,
                    787.09783445,
                    787.4380476,
                    787.76289772,
                    788.07329295,
                    788.37007558,
                    788.65402757,
                    788.92587575,
                    789.18629633,
                    789.43591911,
                    789.67533119,
                    789.90508044,
                    790.12567856,
                    790.33760391,
                    790.54130408,
                    790.73719822,
                    790.92567918,
                    791.10711548,
                    791.28185306,
                    791.45021693,
                    791.61251267,
                    791.7690278,
                    791.92003307,
                    792.06578356,
                    792.20651981,
                    792.34246876,
                    792.47384471,
                    792.60085009,
                    792.72367629,
                    792.84250437,
                    792.95750566,
                    793.06884248,
                    793.17666862,
                    793.28112989,
                    793.38236465,
                    793.48050421,
                    793.57567327,
                    793.66799035,
                    793.75756811,
                ]
            ),
        )
        for field in dataclasses.fields(reference):
            with self.subTest(field):
                expected = getattr(reference, field.name)
                got = getattr(output, field.name)
                self.assertTrue(
                    np.allclose(expected, got),
                    msg=f"{field} expected {expected}\ngot {got}",
                )

    def test_water(self):
        output = self._mock_run("water", "Equilibrate", "out")

        reference_cell = np.array(
            [
                [1.71826948e01, 1.05213661e-15, 1.05213661e-15],
                [0.00000000e00, 1.98408669e01, 1.21490271e-15],
                [0.00000000e00, 0.00000000e00, 4.16913430e01],
            ]
        )
        reference_initial_energy = 31.46838306
        n_equilibration_steps = 10
        reference_energy = np.array(
            [
                31.46838306,
                -4.26942597,
                -8.0805482,
                -12.90076907,
                -16.59626162,
                -18.47873051,
                -18.93215519,
                -18.67742173,
                -15.40624307,
                -16.3339607,
                -15.76489409,
                -16.9078756,
                -17.66525076,
                -16.80113781,
                -15.2462573,
                -14.0982591,
                -14.18508625,
                -13.6224965,
                -14.5188483,
                -12.92224425,
                -12.56978264,
                -13.1520431,
                -12.56133904,
                -14.46415631,
                -13.17798482,
                -11.97324439,
                -11.19910964,
                -11.90843353,
                -12.09170554,
                -12.8062513,
                -12.85819809,
                -10.6408585,
                -11.49414487,
                -10.38585455,
                -10.26564317,
                -13.25688849,
                -11.77704207,
                -12.00399434,
                -11.03770623,
                -10.99755484,
                -11.22154936,
                -11.34856006,
                -12.161935,
                -11.74241786,
                -11.00740265,
                -11.26740768,
                -12.61250859,
                -10.43336029,
                -12.64427893,
                -13.44524596,
                -12.58108501,
                -11.86740602,
                -10.17643539,
                -10.55642178,
                -8.13795718,
                -8.92421519,
                -11.08234688,
                -9.19346724,
                -9.45115114,
                -9.0158142,
                -8.14580283,
                -8.88854698,
                -10.55075374,
                -8.21890427,
                -9.11128619,
                -8.21918011,
                -7.46283417,
                -8.04671325,
                -9.83082643,
                -11.59542109,
                -9.38027939,
                -8.74295339,
                -8.76248409,
                -9.03951998,
                -8.97056175,
                -9.59879561,
                -11.30991291,
                -9.14053122,
                -8.60409889,
                -10.40048267,
                -9.87544651,
                -9.00918605,
                -9.58545053,
                -10.30768069,
                -8.91142289,
                -10.37895725,
                -9.60720873,
                -9.40381222,
                -11.48253315,
                -8.87587232,
                -8.6542851,
                -10.44444786,
                -11.08185074,
                -10.05379793,
                -10.43616253,
                -8.2677656,
                -9.8360291,
                -9.38505847,
                -10.70454689,
                -11.89606576,
                -11.70837562,
            ]
        )

        self.assertTrue(
            np.allclose(reference_cell, output.cells[0]),
            msg=f"expected\n{reference_cell}\ngot\n{output.cells[0]}",
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
