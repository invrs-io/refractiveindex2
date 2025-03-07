"""refractiveindex2: A python interface to the refractiveindex.info database.

Copyright (c) 2025 Martin F. Schubert
"""

import functools
import os
import pathlib
import shutil
import tempfile
import zipfile
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import requests
import yaml

Array = np.ndarray[Any, Any]


# Commits: https://github.com/polyanskiy/refractiveindex.info-database/commits/master/
_DATABASE_SHA = "2f975ebe617dd7f2b2db86a28a1d76910dd9bd1a"
_ZIP_URL = f"https://github.com/polyanskiy/refractiveindex.info-database/archive/{_DATABASE_SHA}.zip"
_BASE_PATH = f"{pathlib.Path(__file__).resolve().parent}/database/{_DATABASE_SHA}"
_ZIP_PATH = f"{_BASE_PATH}/refractiveindex.info-database-{_DATABASE_SHA}.zip"
_DATABASE_PATH = f"{_BASE_PATH}/database"


# Keys to the database which are used across multiple functions.
_TYPE = "type"  # The type of refractive index data, e.g. tabulated or formula.
_DATA = "DATA"  # Data for refractive index or extinction coefficient.


def _download_database(url: str, output_path: str) -> None:
    """Download the database archive to the specified output path."""
    r = requests.get(url)
    if not os.path.exists("/".join(output_path.split("/")[:-1])):
        os.makedirs("/".join(output_path.split("/")[:-1]))
    with open(output_path, "wb") as f:
        f.write(r.content)


def _extract_database(zip_path: str, output_path: str) -> None:
    """Extracts the zip archive to the specified path."""
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(zip_path, "r") as file:
            file.extractall(tempdir)
        temp_database_path = f"{tempdir}/{zip_path.split('/')[-1][:-4]}/database"
        shutil.move(f"{temp_database_path}/data", f"{output_path}/data")
        shutil.move(f"{temp_database_path}/doc", f"{output_path}/doc")
        shutil.move(
            f"{temp_database_path}/catalog-nk.yml", f"{output_path}/catalog-nk.yml"
        )


def _parse_catalog(
    catalog: Sequence[Any],
    path_prefix: str,
) -> Dict[Tuple[str, str, str], str]:
    """Parses the catalog loaded from the database `catalog-nk.yml` file."""
    _shelf = "SHELF"
    _content = "content"
    _book = "BOOK"
    _page = "PAGE"
    _data = "data"

    parsed: Dict[Tuple[str, str, str], str] = {}
    for shelf in catalog:
        if _shelf not in shelf:
            continue
        shelf_name = shelf[_shelf]
        for content in shelf[_content]:
            if _book not in content:
                continue
            book_name = content[_book]
            book = content[_content]
            for page in book:
                if _page not in page:
                    continue
                page_name = page[_page]
                key = (shelf_name, book_name, page_name)
                path = page[_data]
                full_path = f"{path_prefix}/{path}"
                # Duplicate keys are acceptable, but should be consistent.
                if key in parsed:
                    assert parsed[key] == full_path
                parsed[key] = full_path

    return parsed


# On module import, the database is automatically downloaded and parsed.

if not os.path.exists(_ZIP_PATH):
    _download_database(url=_ZIP_URL, output_path=_ZIP_PATH)
if not os.path.exists(_DATABASE_PATH):
    _extract_database(zip_path=_ZIP_PATH, output_path=_DATABASE_PATH)

with open(f"{_DATABASE_PATH}/catalog-nk.yml", encoding="utf-8") as f:
    _CATALOG = _parse_catalog(
        catalog=yaml.safe_load(f),
        path_prefix=f"{_DATABASE_PATH}/data",
    )


class RefractiveIndexMaterial:
    """Defines a material whose optical properties can be queried."""

    def __init__(self, shelf: str, book: str, page: str) -> None:
        """Initializes the `RefractiveIndexMaterial`."""
        self.key = (shelf, book, page)
        if self.key not in _CATALOG:
            raise ValueError(f"Material {self.key} not found in catalog.")
        self.filename = _CATALOG[self.key]
        (
            self.n_fn,
            self.k_fn,
            (
                self.wavelength_um_lower_bound,
                self.wavelength_um_upper_bound,
            ),
        ) = _load_nk_fns(self.filename)

    def get_refractive_index(self, wavelength_um: Array) -> Array:
        """Return the refractive index at the given `wavelength_um`."""
        if self.n_fn is None:
            raise NoRefractiveIndex(
                f"Material {self.key} has no refractive index data."
            )
        _validate_wavelength_in_bounds(
            wavelength_um=wavelength_um,
            lower_bound=self.wavelength_um_lower_bound,
            upper_bound=self.wavelength_um_upper_bound,
        )
        return self.n_fn(wavelength_um)

    def get_extinction_coefficient(self, wavelength_um: Array) -> Array:
        """Return the extinction coefficient at the given `wavelength_um`."""
        if self.k_fn is None:
            raise NoExtinctionCoefficient(
                f"Material {self.key} has no extinction coefficient data."
            )
        _validate_wavelength_in_bounds(
            wavelength_um=wavelength_um,
            lower_bound=self.wavelength_um_lower_bound,
            upper_bound=self.wavelength_um_upper_bound,
        )
        return self.k_fn(wavelength_um)

    def get_epsilon(
        self, wavelength_um: Array, exp_type: str = "exp_minus_i_omega_t"
    ) -> Array:
        """Return the permittivity at the given `wavelength_um`."""
        n = self.get_refractive_index(wavelength_um)
        k = self.get_extinction_coefficient(wavelength_um)
        if exp_type == "exp_minus_i_omega_t":
            return np.asarray((n + 1j * k) ** 2)
        else:
            return np.asarray((n - 1j * k) ** 2)


class NoRefractiveIndex(Exception):
    pass


class NoExtinctionCoefficient(Exception):
    pass


def _validate_wavelength_in_bounds(
    wavelength_um: Array, lower_bound: float, upper_bound: float
) -> None:
    """Checks that values of `wavelength` are in bounds."""
    if np.any(wavelength_um < lower_bound) or np.any(wavelength_um > upper_bound):
        raise ValueError(
            f"Values of `wavelength_um` were out of bounds. Minimum and maximum "
            f"values were {np.amin(wavelength_um)} and {np.amax(wavelength_um)}, "
            f"respectively, but must lie in the range `{(lower_bound, upper_bound)}`."
        )


def _load_nk_fns(
    path: str,
) -> Tuple[
    Callable[[Array], Array] | None,
    Callable[[Array], Array] | None,
    Tuple[float, float],
]:
    """Returns functions that compute refractive index and extinction coefficient."""
    with open(path, encoding="utf-8") as f:
        entry = yaml.safe_load(f)

    # A material may have optical properties stored in two different sections, e.g.
    # refractive index via the coefficients of some analytical expression, and the
    # extinction coefficient by tabular data. Each section defines a wavelength bound,
    # and these need not agree. We always use the wavelength bound defined by the
    # first data section. Also, in some cases both a formula for refractive index
    # and tabular data is given. Here, we use whichever is defined first.
    n_fn = None
    k_fn = None
    bounds = None
    for data in entry[_DATA]:
        _n_fn = None
        _k_fn = None
        _bounds = None

        if data[_TYPE] == "tabulated nk":
            _n_fn, _k_fn, _bounds = _load_tabulated_nk_fns(data)
        elif data[_TYPE] == "tabulated n":
            _n_fn, _bounds = _load_tabulated_fn(data)
        elif data[_TYPE] == "tabulated k":
            _k_fn, _bounds = _load_tabulated_fn(data)
        elif data[_TYPE].startswith("formula"):
            _n_fn, _bounds = _load_formula_fn(data)
        else:
            raise ValueError(f"Unsupported data type: {data[0][_TYPE]}")

        if n_fn is None:
            n_fn = _n_fn
        if k_fn is None:
            k_fn = _k_fn
        if bounds is None:
            bounds = _bounds

    wvl_lo: float
    wvl_hi: float
    assert len(bounds) == 2  # type: ignore[arg-type]
    wvl_lo, wvl_hi = bounds  # type: ignore[misc]
    return n_fn, k_fn, (wvl_lo, wvl_hi)


# -----------------------------------------------------------------------------
# Functions for loading nk functions from tabulated data.
# -----------------------------------------------------------------------------


def _load_tabulated_nk_fns(
    data: Dict[str, str],
) -> Tuple[Callable[[Array], Array], Callable[[Array], Array], Tuple[float, float]]:
    """Return functions for tabulated data including n and k."""
    data_wavelength_um, data_n, data_k = _parse_tabulated(data["data"])

    def n_fn(wavelength_um: Array) -> Array:
        return np.interp(x=wavelength_um, xp=data_wavelength_um, fp=data_n)

    def k_fn(wavelength_um: Array) -> Array:
        return np.interp(x=wavelength_um, xp=data_wavelength_um, fp=data_k)

    wvl_lo = float(np.amin(data_wavelength_um))
    wvl_hi = float(np.amax(data_wavelength_um))
    return n_fn, k_fn, (wvl_lo, wvl_hi)


def _load_tabulated_fn(
    data: Dict[str, str],
) -> Tuple[Callable[[Array], Array], Tuple[float, float]]:
    """Return functions for tabulated data including e.g. n or k only."""
    data_wavelength_um, data_points = _parse_tabulated(data["data"])

    def data_fn(wavelength_um: Array) -> Array:
        return np.interp(x=wavelength_um, xp=data_wavelength_um, fp=data_points)

    wvl_lo = float(np.amin(data_wavelength_um))
    wvl_hi = float(np.amax(data_wavelength_um))
    return data_fn, (wvl_lo, wvl_hi)


def _parse_tabulated(data_str: str) -> Sequence[Array]:
    """Parses tabulated data."""
    data = [[float(f) for f in d.strip().split()] for d in data_str.strip().split("\n")]
    return tuple(np.asarray(f) for f in zip(*data))


# -----------------------------------------------------------------------------
# Functions for loading nk functions from formula coefficients.
# -----------------------------------------------------------------------------


def _load_formula_fn(
    data: Dict[str, str],
) -> Tuple[Callable[[Array], Array], Tuple[float, float]]:
    """Return functions that compute refractive index using formulas."""
    (wvl_lo, wvl_hi), coeffs = _parse_formula(data)
    formula_fn = _FORMULA_REFRACTIVE_INDEX_FNS[data[_TYPE]]
    n_fn = functools.partial(formula_fn, coeffs=coeffs)
    return n_fn, (wvl_lo, wvl_hi)


def _parse_formula(data: Dict[str, str]) -> Tuple[Tuple[float, float], Array]:
    """Extracts wavelength bounds and formula coefficients from `data`."""
    assert set(data.keys()) == {_TYPE, "wavelength_range", "coefficients"}
    wvl_lo, wvl_hi = (float(w) for w in data["wavelength_range"].strip().split())
    coefficients = np.asarray([float(d) for d in data["coefficients"].strip().split()])
    return (wvl_lo, wvl_hi), coefficients


def _refractive_index_formula_1(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using the Sellmeier equation of the first type."""

    def g(wvl: Array, c1: Array, c2: Array) -> Array:
        return np.asarray(c1 * (wvl**2) / (wvl**2 - c2**2))

    n_squared = 1 + coeffs[0]
    for i in range(1, len(coeffs), 2):
        c1, c2 = coeffs[i : i + 2]
        n_squared += g(wavelength_um, c1=c1, c2=c2)
    return np.asarray(np.sqrt(n_squared))


def _refractive_index_formula_2(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using the Sellmeier equation of the second type."""

    def g(wvl: Array, c1: Array, c2: Array) -> Array:
        return np.asarray(c1 * (wvl**2) / (wvl**2 - c2))  # c2 is not squared here

    n_squared = 1 + coeffs[0]
    for i in range(1, len(coeffs), 2):
        c1, c2 = coeffs[i : i + 2]
        n_squared += g(wavelength_um, c1=c1, c2=c2)
    return np.asarray(np.sqrt(n_squared))


def _refractive_index_formula_3(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using a polynomial fit."""

    def g(wvl: Array, c1: Array, c2: Array) -> Array:
        return np.asarray(c1 * wvl**c2)

    n_squared = coeffs[0]
    for i in range(1, len(coeffs), 2):
        c1, c2 = coeffs[i : i + 2]
        n_squared += g(wavelength_um, c1=c1, c2=c2)
    return np.asarray(np.sqrt(n_squared))


def _refractive_index_formula_4(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using formula of type 4 (RefractiveIndex.INFO)."""

    def g1(wvl: Array, c1: Array, c2: Array, c3: Array, c4: Array) -> Array:
        return np.asarray(c1 * wvl**c2 / (wvl**2 - c3**c4))

    def g2(wvl: Array, c1: Array, c2: Array) -> Array:
        return np.asarray(c1 * wvl**c2)

    n_squared = coeffs[0]
    for i in range(1, min(8, len(coeffs)), 4):
        c1, c2, c3, c4 = coeffs[i : i + 4]
        n_squared += g1(wavelength_um, c1=c1, c2=c2, c3=c3, c4=c4)
    if len(coeffs) > 9:
        for i in range(9, len(coeffs), 2):
            c1, c2 = coeffs[i : i + 2]
            n_squared += g2(wavelength_um, c1=c1, c2=c2)
    return np.asarray(np.sqrt(n_squared))


def _refractive_index_formula_5(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using the Cauchy formula."""

    def g(wvl: Array, c1: Array, c2: Array) -> Array:
        return np.asarray(c1 * wvl**c2)

    n = coeffs[0]
    for i in range(1, len(coeffs), 2):
        c1, c2 = coeffs[i : i + 2]
        n += g(wavelength_um, c1=c1, c2=c2)
    return np.asarray(n)


def _refractive_index_formula_6(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using formula of type 6 (gasses)."""

    def g(wvl: Array, c1: Array, c2: Array) -> Array:
        return np.asarray(c1 / (c2 - wvl ** (-2)))

    n = 1 + coeffs[0]
    for i in range(1, len(coeffs), 2):
        c1, c2 = coeffs[i : i + 2]
        n += g(wavelength_um, c1=c1, c2=c2)
    return np.asarray(n)


def _refractive_index_formula_7(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using the Herzberger formula."""

    def g1(wvl: Array, c1: Array, p: Array) -> Array:
        return np.asarray(c1 / (wvl**2 - 0.028) ** p)

    def g2(wvl: Array, c1: Array, p: Array) -> Array:
        return np.asarray(c1 * wvl**p)

    n = coeffs[0]
    n += g1(wavelength_um, coeffs[1], np.asarray(1.0))
    n += g1(wavelength_um, coeffs[2], np.asarray(2.0))
    for i in range(3, len(coeffs)):
        n += g2(wavelength_um, coeffs[i], np.asarray(2.0 * (i - 2.0)))
    return np.asarray(n)


def _refractive_index_formula_8(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using the formula of type 8 (retro)."""
    c1, c2, c3, c4 = coeffs
    rhs = (
        c1
        + (c2 * wavelength_um**2) / (wavelength_um**2 - c3)
        + c4 * wavelength_um**2
    )
    n_squared = (1 + 2 * rhs) / (1 - rhs)
    return np.asarray(np.sqrt(n_squared))


def _refractive_index_formula_9(wavelength_um: Array, coeffs: Array) -> Array:
    """Compute refractive index using the formula of type 9 (exotic)."""
    c1, c2, c3, c4, c5, c6 = coeffs
    n_squared = (
        c1
        + c2 / (wavelength_um**2 - c3)
        + c4 * (wavelength_um - c5) / ((wavelength_um - c5) ** 2 + c6)
    )
    return np.asarray(np.sqrt(n_squared))


# See the `database/doc/Dispersion formulas.pdf` file for details.
_FORMULA_REFRACTIVE_INDEX_FNS = {
    "formula 1": _refractive_index_formula_1,  # Sellmeier 1
    "formula 2": _refractive_index_formula_2,  # Sellmeier 1
    "formula 3": _refractive_index_formula_3,  # polynomial
    "formula 4": _refractive_index_formula_4,  # RefractiveIndex.INFO
    "formula 5": _refractive_index_formula_5,  # Cauchy
    "formula 6": _refractive_index_formula_6,  # Gasses
    "formula 7": _refractive_index_formula_7,  # Herzberger
    "formula 8": _refractive_index_formula_8,  # Retro
    "formula 9": _refractive_index_formula_9,  # Exotic
}
