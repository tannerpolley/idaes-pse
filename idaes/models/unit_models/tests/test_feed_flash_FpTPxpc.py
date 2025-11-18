#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Tests for feed with flash.
Authors: Andrew Lee
"""

import pytest

from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    value,
    units as pyunits,
)

from core.util.model_statistics import degrees_of_freedom
from idaes.core import FlowsheetBlock, MaterialBalanceType, Component
from idaes.core import LiquidPhase, VaporPhase
from idaes.models.unit_models.feed_flash import FeedFlash, FlashType
from idaes.models.properties import iapws95
from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)
from idaes.core.util.model_statistics import (
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core.util.testing import PhysicalParameterTestBlock, initialization_tester
from idaes.core.solvers import get_solver

from idaes.core.initialization import (
    SingleControlVolumeUnitInitializer,
    BlockTriangularizationInitializer,
    InitializationStatus,
)
from idaes.core.util import DiagnosticsToolbox

from idaes.models.properties.modular_properties.state_definitions import FpTPxpc
from idaes.models.properties.modular_properties.eos.ideal import Ideal
from idaes.models.properties.modular_properties.phase_equil import SmoothVLE
from idaes.models.properties.modular_properties.phase_equil.bubble_dew import (
    IdealBubbleDew,
)
from idaes.models.properties.modular_properties.phase_equil.forms import fugacity

import idaes.models.properties.modular_properties.pure.Perrys as Perrys
import idaes.models.properties.modular_properties.pure.RPP4 as RPP4
import idaes.models.properties.modular_properties.pure.NIST as NIST

from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)


# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver("ipopt_v2")

config_dict = {
    "components": {
        "benzene": {
            "type": Component,
            "dens_mol_liq_comp": Perrys,
            "enth_mol_liq_comp": Perrys,
            "enth_mol_ig_comp": RPP4,
            "pressure_sat_comp": NIST,
            "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
            "parameter_data": {
                "mw": 78.1136e-3,  # [1]
                "pressure_crit": 48.9e5,  # [1]
                "temperature_crit": 562.2,  # [1]
                "dens_mol_liq_comp_coeff": {
                    "eqn_type": 1,
                    "1": 1.0162,  # [2] pg. 2-98
                    "2": 0.2655,
                    "3": 562.16,
                    "4": 0.28212,
                },
                "cp_mol_ig_comp_coeff": {
                    "A": -3.392e1,  # [1]
                    "B": 4.739e-1,
                    "C": -3.017e-4,
                    "D": 7.130e-8,
                },
                "cp_mol_liq_comp_coeff": {
                    "1": 1.29e2,  # [2]
                    "2": -1.7e-1,
                    "3": 6.48e-4,
                    "4": 0,
                    "5": 0,
                },
                "enth_mol_form_liq_comp_ref": 49.0e3,  # [3]
                "enth_mol_form_vap_comp_ref": 82.9e3,  # [3]
                "pressure_sat_comp_coeff": {
                    "A": 4.72583,  # [NIST]
                    "B": 1660.652,
                    "C": -1.461,
                },
            },
        },
        "toluene": {
            "type": Component,
            "dens_mol_liq_comp": Perrys,
            "enth_mol_liq_comp": Perrys,
            "enth_mol_ig_comp": RPP4,
            "pressure_sat_comp": NIST,
            "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
            "parameter_data": {
                "mw": 92.1405e-3,  # [1]
                "pressure_crit": 41e5,  # [1]
                "temperature_crit": 591.8,  # [1]
                "dens_mol_liq_comp_coeff": {
                    "eqn_type": 1,
                    "1": 0.8488,  # [2] pg. 2-98
                    "2": 0.26655,
                    "3": 591.8,
                    "4": 0.2878,
                },
                "cp_mol_ig_comp_coeff": {
                    "A": -2.435e1,
                    "B": 5.125e-1,
                    "C": -2.765e-4,
                    "D": 4.911e-8,
                },
                "cp_mol_liq_comp_coeff": {
                    "1": 1.40e2,  # [2]
                    "2": -1.52e-1,
                    "3": 6.95e-4,
                    "4": 0,
                    "5": 0,
                },
                "enth_mol_form_liq_comp_ref": 12.0e3,  # [3]
                "enth_mol_form_vap_comp_ref": 50.1e3,  # [3]
                "pressure_sat_comp_coeff": {
                    "A": 4.07827,  # [NIST]
                    "B": 1343.943,
                    "C": -53.773,
                },
            },
        },
    },
    "phases": {
        "Liq": {"type": LiquidPhase, "equation_of_state": Ideal},
        "Vap": {"type": VaporPhase, "equation_of_state": Ideal},
    },
    "base_units": {
        "time": pyunits.s,
        "length": pyunits.m,
        "mass": pyunits.kg,
        "amount": pyunits.mol,
        "temperature": pyunits.K,
    },
    "state_definition": FpTPxpc,
    "state_bounds": {
        "flow_mol_phase": (0, 100, 1000, pyunits.mol / pyunits.s),
        "temperature": (273.15, 300, 450, pyunits.K),
        "pressure": (5e4, 1e5, 1e6, pyunits.Pa),
    },
    "pressure_ref": 1e5,
    "temperature_ref": 300,
    "phases_in_equilibrium": [("Vap", "Liq")],
    "phase_equilibrium_state": {("Vap", "Liq"): SmoothVLE},
    "bubble_dew_method": IdealBubbleDew,
}


# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_config():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = PhysicalParameterTestBlock()

    m.fs.unit = FeedFlash(property_package=m.fs.properties)

    # Check unit config arguments
    assert len(m.fs.unit.config) == 6

    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup
    assert m.fs.unit.config.material_balance_type == MaterialBalanceType.useDefault
    assert m.fs.unit.config.flash_type == FlashType.isothermal
    assert m.fs.unit.config.property_package is m.fs.properties

    assert m.fs.unit.default_initializer is SingleControlVolumeUnitInitializer


# -----------------------------------------------------------------------------
class TestBTXIdeal(object):
    @pytest.fixture(scope="class")
    def btx(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.properties = GenericParameterBlock(**config_dict)

        m.fs.unit = FeedFlash(property_package=m.fs.properties)

        # Fix state
        phase_frac = 0.5
        x_liq_benz = 0.5
        x_vap_benz = 0.5

        m.fs.unit.flow_mol_phase[0, "Liq"].fix(1.0 - phase_frac)
        m.fs.unit.flow_mol_phase[0, "Vap"].fix(phase_frac)
        m.fs.unit.mole_frac_phase_comp[0, "Liq", "benzene"].fix(x_liq_benz)
        m.fs.unit.mole_frac_phase_comp[0, "Liq", "toluene"].fix(1 - x_liq_benz)
        m.fs.unit.mole_frac_phase_comp[0, "Vap", "benzene"].fix(x_vap_benz)
        m.fs.unit.mole_frac_phase_comp[0, "Vap", "toluene"].fix(1 - x_vap_benz)
        m.fs.unit.temperature.fix(368)
        m.fs.unit.pressure.fix(101325)

        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, btx):
        assert hasattr(btx.fs.unit, "flow_mol_phase")
        assert hasattr(btx.fs.unit, "mole_frac_phase_comp")
        assert hasattr(btx.fs.unit, "temperature")
        assert hasattr(btx.fs.unit, "pressure")

        assert hasattr(btx.fs.unit, "outlet")
        assert len(btx.fs.unit.outlet.vars) == 4
        assert hasattr(btx.fs.unit.outlet, "flow_mol_phase")
        assert hasattr(btx.fs.unit.outlet, "mole_frac_phase_comp")
        assert hasattr(btx.fs.unit.outlet, "temperature")
        assert hasattr(btx.fs.unit.outlet, "pressure")

        assert hasattr(btx.fs.unit, "isothermal")

        print()
        print(f'number_variables: {number_variables(btx)}')
        print(f'number_total_constraints: {number_total_constraints(btx)}')
        print(f'number_unused_variables: {number_unused_variables(btx)}')

        assert number_variables(btx) == 34
        assert number_total_constraints(btx) == 29
        assert number_unused_variables(btx) == 0

    @pytest.mark.component
    def test_structural_issues(self, btx):
        dt = DiagnosticsToolbox(btx)
        dt.report_structural_issues()
        dt.display_potential_evaluation_errors()
        dt.assert_no_structural_warnings()

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_performance_contents(self, btx):
        perf_dict = btx.fs.unit._get_performance_contents()

        assert perf_dict is None

    # @pytest.mark.ui
    # @pytest.mark.unit
    # def test_get_stream_table_contents(self, btx):
    #     stable = btx.fs.unit._get_stream_table_contents()
    #
    #     print(stable.to_dict())
    #
    #     expected = {
    #         "Units": {
    #             "flow_mol_phase": getattr(pyunits.pint_registry, "mole/second"),
    #             "mole_frac_phase_comp benzene": getattr(
    #                 pyunits.pint_registry, "dimensionless"
    #             ),
    #             "mole_frac_phase_comp toluene": getattr(
    #                 pyunits.pint_registry, "dimensionless"
    #             ),
    #             "temperature": getattr(pyunits.pint_registry, "kelvin"),
    #             "pressure": getattr(pyunits.pint_registry, "Pa"),
    #         },
    #         "Outlet": {
    #             "flow_mol_phase": pytest.approx(1.0, rel=1e-4),
    #             "mole_frac_phase_comp benzene": pytest.approx(0.5, rel=1e-4),
    #             "mole_frac_phase_comp toluene": pytest.approx(0.5, rel=1e-4),
    #             "temperature": pytest.approx(298.15, rel=1e-4),
    #             "pressure": pytest.approx(101325.0, rel=1e-4),
    #         },
    #     }
    #
    #     assert stable.to_dict() == expected

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, btx):
        initialization_tester(btx)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, btx):
        results = solver.solve(btx)

        # Check for optimal solution
        assert check_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, btx):
        # assert pytest.approx(101325.0, abs=1e3) == value(btx.fs.unit.outlet.pressure[0])
        # assert pytest.approx(368.00, abs=1e-0) == value(
        #     btx.fs.unit.outlet.temperature[0]
        # )
        # assert pytest.approx(1.0, abs=1e-2) == value(btx.fs.unit.outlet.flow_mol[0])
        print()
        print('x_vap_ben', value(btx.fs.unit.outlet.mole_frac_phase_comp[0, "Vap", 'benzene']))
        print('x_vap_tol', value(btx.fs.unit.outlet.mole_frac_phase_comp[0, "Vap", 'toluene']))
        print('x_liq_ben', value(btx.fs.unit.outlet.mole_frac_phase_comp[0, "Liq", 'benzene']))
        print('x_liq_tol', value(btx.fs.unit.outlet.mole_frac_phase_comp[0, "Liq", 'toluene']))
        # assert pytest.approx(0.396, abs=1e-3) == value(
        #     btx.fs.unit.control_volume.properties_out[0].flow_mol_phase["Vap"]
        # )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_numerical_issues(self, btx):
        dt = DiagnosticsToolbox(btx)
        dt.assert_no_numerical_warnings()


# -----------------------------------------------------------------------------
@pytest.mark.iapws
@pytest.mark.skipif(not iapws95.iapws95_available(), reason="IAPWS not available")
class TestIAPWS(object):
    @pytest.fixture(scope="class")
    def iapws(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.properties = iapws95.Iapws95ParameterBlock(
            phase_presentation=iapws95.PhaseType.LG
        )

        m.fs.unit = FeedFlash(
            property_package=m.fs.properties, flash_type=FlashType.isenthalpic
        )

        m.fs.unit.flow_mol.fix(100)
        m.fs.unit.enth_mol.fix(24000)
        m.fs.unit.pressure.fix(101325)

        return m

    @pytest.mark.unit
    def test_build(self, iapws):
        assert hasattr(iapws.fs.unit, "flow_mol")
        assert hasattr(iapws.fs.unit, "enth_mol")
        assert hasattr(iapws.fs.unit, "pressure")

        assert hasattr(iapws.fs.unit, "outlet")
        assert len(iapws.fs.unit.outlet.vars) == 3
        assert hasattr(iapws.fs.unit.outlet, "flow_mol")
        assert hasattr(iapws.fs.unit.outlet, "enth_mol")
        assert hasattr(iapws.fs.unit.outlet, "pressure")

        assert hasattr(iapws.fs.unit, "isenthalpic")

        assert number_variables(iapws) == 6
        assert number_total_constraints(iapws) == 3
        assert number_unused_variables(iapws) == 0

    @pytest.mark.component
    def test_structural_issues(self, iapws):
        dt = DiagnosticsToolbox(iapws)
        dt.assert_no_structural_warnings()

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_performance_contents(self, iapws):
        perf_dict = iapws.fs.unit._get_performance_contents()

        assert perf_dict is None

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_stream_table_contents(self, iapws):
        stable = iapws.fs.unit._get_stream_table_contents()

        expected = {
            "Units": {
                "Molar Flow": getattr(pyunits.pint_registry, "mole/second"),
                "Mass Flow": getattr(pyunits.pint_registry, "kg/second"),
                "T": getattr(pyunits.pint_registry, "K"),
                "P": getattr(pyunits.pint_registry, "Pa"),
                "Vapor Fraction": getattr(pyunits.pint_registry, "dimensionless"),
                "Molar Enthalpy": getattr(pyunits.pint_registry, "J/mole"),
            },
            "Outlet": {
                "Molar Flow": pytest.approx(1, rel=1e-4),
                "Mass Flow": pytest.approx(1.8015e-2, rel=1e-4),
                "T": pytest.approx(270.4877112932641, rel=1e-4),
                "P": pytest.approx(11032305.8275, rel=1e-4),
                "Vapor Fraction": pytest.approx(0, abs=1e-4),
                "Molar Enthalpy": pytest.approx(0.01102138712926277, rel=1e-4),
            },
        }

        assert stable.to_dict() == expected

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, iapws):
        initialization_tester(iapws)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, iapws):
        results = solver.solve(iapws)

        # Check for optimal solution
        assert check_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, iapws):
        assert pytest.approx(101325.0, abs=1e3) == value(
            iapws.fs.unit.outlet.pressure[0]
        )
        assert pytest.approx(24000, abs=1e3) == value(iapws.fs.unit.outlet.enth_mol[0])
        assert pytest.approx(100.0, abs=1e-2) == value(iapws.fs.unit.outlet.flow_mol[0])

        assert pytest.approx(373.12, abs=1e-2) == value(
            iapws.fs.unit.control_volume.properties_out[0].temperature
        )
        assert pytest.approx(0.5953, abs=1e-4) == value(
            iapws.fs.unit.control_volume.properties_out[0].phase_frac["Liq"]
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_numerical_issues(self, iapws):
        dt = DiagnosticsToolbox(iapws)
        dt.assert_no_numerical_warnings()


class TestInitializersIAWPS:
    @pytest.fixture
    def model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.properties = iapws95.Iapws95ParameterBlock(
            phase_presentation=iapws95.PhaseType.LG
        )

        m.fs.unit = FeedFlash(
            property_package=m.fs.properties, flash_type=FlashType.isenthalpic
        )

        m.fs.unit.flow_mol.fix(100)
        m.fs.unit.enth_mol.fix(24000)
        m.fs.unit.pressure.fix(101325)

        return m

    @pytest.mark.integration
    def test_general_hierarchical(self, model):
        initializer = SingleControlVolumeUnitInitializer()
        initializer.initialize(model.fs.unit)

        assert initializer.summary[model.fs.unit]["status"] == InitializationStatus.Ok

        assert pytest.approx(101325.0, abs=1e3) == value(
            model.fs.unit.outlet.pressure[0]
        )
        assert pytest.approx(24000, abs=1e3) == value(model.fs.unit.outlet.enth_mol[0])
        assert pytest.approx(100.0, abs=1e-2) == value(model.fs.unit.outlet.flow_mol[0])

        assert pytest.approx(373.12, abs=1e-2) == value(
            model.fs.unit.control_volume.properties_out[0].temperature
        )
        assert pytest.approx(0.5953, abs=1e-4) == value(
            model.fs.unit.control_volume.properties_out[0].phase_frac["Liq"]
        )

    @pytest.mark.integration
    def test_block_triangularization(self, model):
        initializer = BlockTriangularizationInitializer(constraint_tolerance=2e-5)
        initializer.initialize(model.fs.unit)

        assert initializer.summary[model.fs.unit]["status"] == InitializationStatus.Ok

        assert pytest.approx(101325.0, abs=1e3) == value(
            model.fs.unit.outlet.pressure[0]
        )
        assert pytest.approx(24000, abs=1e3) == value(model.fs.unit.outlet.enth_mol[0])
        assert pytest.approx(100.0, abs=1e-2) == value(model.fs.unit.outlet.flow_mol[0])

        assert pytest.approx(373.12, abs=1e-2) == value(
            model.fs.unit.control_volume.properties_out[0].temperature
        )
        assert pytest.approx(0.5953, abs=1e-4) == value(
            model.fs.unit.control_volume.properties_out[0].phase_frac["Liq"]
        )


class TestInitializersBT:
    @pytest.fixture
    def model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.properties = BTXParameterBlock(
            valid_phase=("Liq", "Vap"), activity_coeff_model="Ideal"
        )

        m.fs.unit = FeedFlash(property_package=m.fs.properties)

        m.fs.unit.flow_mol[0].set_value(1)
        m.fs.unit.temperature[0].set_value(368)
        m.fs.unit.pressure[0].set_value(101325)
        m.fs.unit.mole_frac_comp[0, "benzene"].set_value(0.5)
        m.fs.unit.mole_frac_comp[0, "toluene"].set_value(0.5)

        return m

    @pytest.mark.component
    def test_general_hierarchical(self, model):
        initializer = SingleControlVolumeUnitInitializer()
        initializer.initialize(model.fs.unit)

        assert initializer.summary[model.fs.unit]["status"] == InitializationStatus.Ok

        assert pytest.approx(101325.0, abs=1e3) == value(
            model.fs.unit.outlet.pressure[0]
        )
        assert pytest.approx(368.00, abs=1e-0) == value(
            model.fs.unit.outlet.temperature[0]
        )
        assert pytest.approx(1.0, abs=1e-2) == value(model.fs.unit.outlet.flow_mol[0])
        assert pytest.approx(0.396, abs=1e-3) == value(
            model.fs.unit.control_volume.properties_out[0].flow_mol_phase["Vap"]
        )

        assert not model.fs.unit.flow_mol[0].fixed
        assert not model.fs.unit.temperature[0].fixed
        assert not model.fs.unit.pressure[0].fixed
        assert not model.fs.unit.mole_frac_comp[0, "benzene"].fixed
        assert not model.fs.unit.mole_frac_comp[0, "toluene"].fixed

    @pytest.mark.component
    def test_block_triangularization(self, model):
        initializer = BlockTriangularizationInitializer(constraint_tolerance=2e-5)
        initializer.initialize(model.fs.unit)

        assert initializer.summary[model.fs.unit]["status"] == InitializationStatus.Ok

        assert pytest.approx(101325.0, abs=1e3) == value(
            model.fs.unit.outlet.pressure[0]
        )
        assert pytest.approx(368.00, abs=1e-0) == value(
            model.fs.unit.outlet.temperature[0]
        )
        assert pytest.approx(1.0, abs=1e-2) == value(model.fs.unit.outlet.flow_mol[0])
        assert pytest.approx(0.396, abs=1e-3) == value(
            model.fs.unit.control_volume.properties_out[0].flow_mol_phase["Vap"]
        )

        assert not model.fs.unit.flow_mol[0].fixed
        assert not model.fs.unit.temperature[0].fixed
        assert not model.fs.unit.pressure[0].fixed
        assert not model.fs.unit.mole_frac_comp[0, "benzene"].fixed
        assert not model.fs.unit.mole_frac_comp[0, "toluene"].fixed
