import configparser
from pathlib import Path

from packaging.requirements import Requirement


def test_extras_require_are_individual_pep508_requirements():
    setup_cfg = Path(__file__).resolve().parents[1] / "setup.cfg"
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(setup_cfg)
    extras = parser["options.extras_require"]

    expected = {
        "api": {"fastapi", "uvicorn"},
        "dev": {"pytest", "pytest-cov"},
    }
    for extra, expected_names in expected.items():
        requirements = [line.strip() for line in extras[extra].splitlines() if line.strip()]
        parsed_names = {Requirement(requirement).name for requirement in requirements}
        assert parsed_names == expected_names
