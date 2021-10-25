#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
"""
Deprecation path for renamed Tray model.
"""
from pyomo.common.deprecation import RenamedClass
from idaes.generic_models.unit_models.column_models.tray import \
    Tray as NewTray


class Tray(metaclass=RenamedClass):
    __renamed__new_class__ = NewTray
    __renamed__version__ = '1.12'
