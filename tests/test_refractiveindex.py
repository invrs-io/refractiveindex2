"""Tests for the `refractiveindex` module.

Copyright (c) 2025 Martin F. Schubert
"""

import unittest

import numpy as np
import yaml
from parameterized import parameterized

import refractiveindex2 as ri


def custom_name_func(testcase_func, param_num, param):
    # Generate the custom test name from the test arguments.
    del param_num
    shelf, book, page = param[0]
    return f"{testcase_func.__name__}_{shelf}_{book}_{page}"


class RefractiveIndexMaterialTest(unittest.TestCase):
    @parameterized.expand(
        list(ri.refractiveindex._CATALOG.keys()),
        name_func=custom_name_func,
    )
    def test_can_load_material(self, shelf, book, page):
        ri.RefractiveIndexMaterial(shelf, book, page)

    @parameterized.expand(
        list(ri.refractiveindex._CATALOG.keys()),
        name_func=custom_name_func,
    )
    def test_can_compute_refractive_index(self, shelf, book, page):
        mat = ri.RefractiveIndexMaterial(shelf, book, page)
        self.assertFalse((mat.n_fn is None) and (mat.k_fn is None))
        wavelength_um = (
            mat.wavelength_um_lower_bound + mat.wavelength_um_upper_bound
        ) / 2
        if mat.n_fn is not None:
            refractive_index = mat.get_refractive_index(np.asarray(wavelength_um))
        if mat.k_fn is not None:
            extinction_coefficient = mat.get_extinction_coefficient(
                np.asarray(wavelength_um)
            )
        if (mat.n_fn is not None) and (mat.k_fn is not None):
            permittivity = mat.get_epsilon(np.asarray(wavelength_um))
            np.testing.assert_allclose(
                permittivity,
                (refractive_index + 1j * extinction_coefficient) ** 2,
            )


class NoDataExceptionTest(unittest.TestCase):
    def test_no_refractive_index(self):
        mat = ri.RefractiveIndexMaterial(shelf="main", book="H2O", page="Wang")
        with self.assertRaisesRegex(
            ri.NoRefractiveIndex, ".* has no refractive index data."
        ):
            mat.get_refractive_index(0.5)

    def test_no_extinction_coefficient(self):
        mat = ri.RefractiveIndexMaterial(shelf="main", book="Si3N4", page="Luke")
        with self.assertRaisesRegex(
            ri.NoExtinctionCoefficient, ".* has no extinction coefficient data."
        ):
            mat.get_extinction_coefficient(0.5)


class RefractiveIndexValueTest(unittest.TestCase):
    def test_values_match_expected_tabulated_nk(self):
        # https://refractiveindex.info/?shelf=main&book=Ag&page=Johnson
        mat = ri.RefractiveIndexMaterial(shelf="main", book="Ag", page="Johnson")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "tabulated nk")
        n = mat.get_refractive_index(wavelength_um=0.984)
        k = mat.get_extinction_coefficient(wavelength_um=0.984)
        np.testing.assert_allclose(n, 0.04, atol=1e-15)
        np.testing.assert_allclose(k, 6.992, atol=1e-15)

    def test_values_match_expected_tabulated_n(self):
        # https://refractiveindex.info/?shelf=main&book=Ar&page=Larsen
        mat = ri.RefractiveIndexMaterial(shelf="main", book="Ar", page="Larsen")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "tabulated n")
        n = mat.get_refractive_index(wavelength_um=0.41092)
        np.testing.assert_allclose(n, 1.000286541, atol=1e-15)

    def test_values_match_expected_tabulated_k(self):
        # https://refractiveindex.info/?shelf=main&book=H2O&page=Wang
        mat = ri.RefractiveIndexMaterial(shelf="main", book="H2O", page="Wang")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "tabulated k")
        k = mat.get_extinction_coefficient(wavelength_um=1.6)
        np.testing.assert_allclose(k, 0.0000980207, atol=1e-15)

    def test_values_match_expected_formula_1(self):
        # https://refractiveindex.info/?shelf=main&book=Ar&page=Grace-liquid-90K
        mat = ri.RefractiveIndexMaterial(
            shelf="main", book="Ar", page="Grace-liquid-90K"
        )
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 1")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([0.12240, 0.19850, 0.39940]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([1.48738387294016, 1.258212767722068, 1.227635767292153]),
            atol=1e-15,
        )

    def test_values_match_expected_formula_2(self):
        # https://refractiveindex.info/?shelf=main&book=Ar&page=Borzsonyi
        mat = ri.RefractiveIndexMaterial(shelf="main", book="Ar", page="Borzsonyi")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 2")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([0.406, 0.664, 0.994]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([1.0002829048665627, 1.000277172874615, 1.0002753554410945]),
            atol=1e-15,
        )

    def test_values_match_expected_formula_3(self):
        # https://refractiveindex.info/?shelf=main&book=BeAl6O10&page=Pestryakov-α
        mat = ri.RefractiveIndexMaterial(
            shelf="main", book="BeAl6O10", page="Pestryakov-α"
        )
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 3")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([0.4434, 0.6913, 1.067]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([1.7540636750136969, 1.737215705939756, 1.72805219823705]),
            atol=1e-15,
        )

    def test_values_match_expected_formula_4(self):
        # https://refractiveindex.info/?shelf=main&book=KNbO3&page=Zysset-α
        mat = ri.RefractiveIndexMaterial(shelf="main", book="KNbO3", page="Zysset-α")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 4")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([0.4, 0.85, 3.19]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([2.31717108479427, 2.13473569847233, 2.05709122611713]),
            atol=1e-15,
        )

    def test_values_match_expected_formula_5(self):
        # https://refractiveindex.info/?shelf=main&book=SiC&page=Shaffer
        mat = ri.RefractiveIndexMaterial(shelf="main", book="SiC", page="Shaffer")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 5")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([0.467, 0.5678, 0.682]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([2.71061671244309, 2.65988044421279, 2.62732877942226]),
            atol=1e-15,
        )

    def test_values_match_expected_formula_6(self):
        # https://refractiveindex.info/?shelf=main&book=Ar&page=Cuthbertson
        mat = ri.RefractiveIndexMaterial(shelf="main", book="Ar", page="Cuthbertson")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 6")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([0.4819, 0.543, 0.6594]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([1.0002837415737948, 1.0002823453630771, 1.0002806965920565]),
            atol=1e-15,
        )

    def test_values_match_expected_formula_7(self):
        # https://refractiveindex.info/?shelf=main&book=Si&page=Edwards
        mat = ri.RefractiveIndexMaterial(shelf="main", book="Si", page="Edwards")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 7")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([2.663, 8.98, 24.32]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([3.44000961990886, 3.42188432316437, 3.42016792554306]),
            atol=1e-15,
        )

    def test_values_match_expected_formula_8(self):
        # https://refractiveindex.info/?shelf=main&book=AgBr&page=Schröter
        mat = ri.RefractiveIndexMaterial(shelf="main", book="AgBr", page="Schröter")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 8")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([0.4985, 0.5703, 0.6648]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([2.3107296174871097, 2.26543016085205, 2.233424297615897]),
            atol=1e-15,
        )

    def test_values_match_expected_formula_9(self):
        # https://refractiveindex.info/?shelf=organic&book=urea&page=Rosker-e
        mat = ri.RefractiveIndexMaterial(shelf="organic", book="urea", page="Rosker-e")
        with open(mat.filename) as f:
            contents = yaml.safe_load(f)
        self.assertEqual(contents["DATA"][0]["type"], "formula 9")
        n = mat.get_refractive_index(
            wavelength_um=np.asarray([0.3076, 0.5964, 1.007]),
        )
        np.testing.assert_allclose(
            n,
            np.asarray([1.6959815715611959, 1.605703093293888, 1.590805190018974]),
            atol=1e-15,
        )
