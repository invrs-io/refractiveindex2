# A python interface to refractiveindex.info
![Continuous integration](https://github.com/mfschubert/refractiveindex2/actions/workflows/build-ci.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/refractiveindex2)

This repo is based on the [refractiveindex repo](https://github.com/toftul/refractiveindex) by [Ivan Toftul](https://github.com/toftul), which itself is based on the `refractiveindex.py` module from the [PyTMM project](https://github.com/kitchenknif/PyTMM) by [Pavel Dmitriev](https://github.com/kitchenknif). It provides a python interface to the [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database) by [Mikhail Polyanskiy](https://github.com/polyanskiy).

Compared to other interfaces to the database, `refractiveindex2` implements additional formulas required by some materials in the database and is rewritten to have a simpler class structure. It also includes extensive tests for all materials in the database.


## Installation

```
pip install refractiveindex2
```

## Usage


```python
import refractiveindex2 as ri

SiO = ri.RefractiveIndexMaterial(shelf="main", book="SiO", page="Hass")

wavelength_um = 0.6  # [microns]

SiO.get_epsilon(wavelength_um=wavelength_um)
# (3.8633404437869827 + 0.003931076923076923j)

SiO.get_refractive_index(wavelength_um=wavelength_um)
# (1.96553846)

SiO.get_extinction_coefficient(wavelength_um=wavelength_um)
# (0.001)
```

Notes:
- here the time dependence is assumed to be $\mathrm{e}^{-\mathrm{i} \omega t}$, so $\mathrm{Im}(\varepsilon) > 0$ is responsible for the losses.
- if there is a space in the name, one should write underscore instead of it, i.e. not `page='Rodriguez-de Marcos'` but `page='Rodriguez-de_Marcos'`.


## How to get material page names

You can find the proper “page” name by hovering your cursor on the link in the Data section

![How to get page name](docs/img/material_page_names.png)

Or you can look up folders in this repository<br>
https://github.com/polyanskiy/refractiveindex.info-database

## Known issues
For unknown reasons, download of the database can be slow on some systems. You can avoid this by manually downloading the zip file with the `SHA` hard-coded in the `refractiveindex.py` module, e.g.
```
https://github.com/polyanskiy/refractiveindex.info-database/archive/{SHA}.zip
```
which will be a file named
```
refractiveindex.info-database-{SHA}.zip
```
Then, manually add this file to the `src/refractiveindex2/database/{SHA}` directory, creating the directory if needed. This ensures that the automatic download of the database on first module import is avoided.


## Related projects

- https://github.com/stillyslalom/RefractiveIndex.jl
- https://github.com/kitchenknif/PyTMM
- https://github.com/toftul/refractiveindex
- https://github.com/polyanskiy/refractiveindex.info-database
