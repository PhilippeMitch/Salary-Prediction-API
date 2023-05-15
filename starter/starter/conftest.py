"""
Implement our parametrization scheme
author: Philippe Jean Mith
Date: April 13th 2023
"""
def pytest_addoption(parser):
    parser.addoption(
        "--data_path",
        action="append",
        default=[],
        help="The path of the data",
    )

def pytest_generate_tests(metafunc):
    if "data_path" in metafunc.fixturenames:
        metafunc.parametrize("data_path", metafunc.config.getoption("data_path"))