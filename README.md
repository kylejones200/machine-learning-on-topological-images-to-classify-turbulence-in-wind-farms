# Machine Learning on Topological Images to Classify Turbulence in Wind Farms

Published: 2025-11-10
Medium: [https://medium.com/@kyle-t-jones/machine-learning-on-topological-images-to-classify-turbulence-in-wind-farms-796a70ae1f1b](https://medium.com/@kyle-t-jones/machine-learning-on-topological-images-to-classify-turbulence-in-wind-farms-796a70ae1f1b)

## Business context

Turbulence intensity defines how much wind speed fluctuates around its mean. In wind energy, turbulence intensity is not a curiosity --- it directly determines structural loads and fatigue life. A turbine operating in ten-percent turbulence intensity experiences baseline structural loads. At 20% turbulence, those loads increase by thirty to fifty percent. Over twenty years, high turbulence can reduce component life by decades or force premature replacement of blades, drivetrains, and towers.

Measuring turbulence properly requires three-dimensional sonic anemometers that capture wind velocity in all directions at fifty hertz or higher sampling rates. These instruments cost ten thousand to fifty thousand dollars per installation and require careful calibration and maintenance. Most wind turbines do not have them. Standard SCADA systems record only mean wind speed averaged over ten-minute intervals, with no direct measure of turbulence.

This creates a problem for operational monitoring. Operators know turbulence matters but cannot measure it with existing sensors. They infer turbulence indirectly through power variability or rotor speed fluctuations, but these proxies are imperfect --- power variability could reflect turbulence or control actions or grid disturbances. Load monitoring systems that depend on knowing turbulence intensity cannot function without turbulence measurements, forcing operators to use conservative assumptions that overestimate loads and trigger unnecessary maintenance.

## About

Place the code for this article in this repository.
The original article export is saved as `article.md`.

## Files

Add your `.ipynb`, `.py`, `.yaml`, `.js`, `.ts`, or other project files here.

## Setup

1. Copy `.env.example` to `.env` and set `NREL_API_KEY` (free at [developer.nrel.gov/signup](https://developer.nrel.gov/signup/)). Optionally set `NREL_EMAIL` for large downloads.
2. Adjust non-secret NREL settings in `config.yaml` (`nrel.lat`, `nrel.lon`, `nrel.years`, etc.).
3. Install dependencies: `uv sync` (or `pip install -e .`).

Runnable scripts load `config.yaml` and read secrets from `.env` via `python-dotenv` (see `nrel_wtk.py`).

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).