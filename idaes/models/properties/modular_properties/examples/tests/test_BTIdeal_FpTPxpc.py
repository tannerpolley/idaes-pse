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
Author: Andrew Lee
"""
import pytest
from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    Set,
    value,
    Var,
    units as pyunits,
)
from pyomo.util.check_units import assert_units_consistent
from pyomo.common.unittest import assertStructuredAlmostEqual

from idaes.core import Component
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    fixed_variables_set,
    activated_constraints_set,
)
from idaes.core.solvers import get_solver

from idaes.core import LiquidPhase, VaporPhase
from idaes.core import FlowsheetBlock, MaterialBalanceType, Component

from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.unit_models.feed_flash import FeedFlash, FlashType

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

from idaes.models.properties.tests.test_harness import PropertyTestHarness

from idaes.models.unit_models.feed_flash import FeedFlash


# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver("ipopt_v2")


def _as_quantity(x):
    unit = pyunits.get_units(x)
    if unit is None:
        unit = pyunits.dimensionless
    return value(x) * unit._get_pint_unit()

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


class TestBTIdeal_FpTPxpc(PropertyTestHarness):
    def configure(self):
        self.prop_pack = GenericParameterBlock
        self.param_args = config_dict
        self.prop_args = {}
        self.has_density_terms = True


class TestParamBlock(object):
    @pytest.mark.unit
    def test_build(self):
        model = ConcreteModel()
        model.params = GenericParameterBlock(**config_dict)

        assert isinstance(model.params.phase_list, Set)
        assert len(model.params.phase_list) == 2
        for i in model.params.phase_list:
            assert i in ["Liq", "Vap"]
        assert model.params.Liq.is_liquid_phase()
        assert model.params.Vap.is_vapor_phase()

        assert isinstance(model.params.component_list, Set)
        assert len(model.params.component_list) == 2
        for i in model.params.component_list:
            assert i in ["benzene", "toluene"]
            assert isinstance(model.params.get_component(i), Component)

        assert isinstance(model.params._phase_component_set, Set)
        assert len(model.params._phase_component_set) == 4
        for i in model.params._phase_component_set:
            assert i in [
                ("Liq", "benzene"),
                ("Liq", "toluene"),
                ("Vap", "benzene"),
                ("Vap", "toluene"),
            ]

        assert model.params.config.state_definition == FpTPxpc

        assertStructuredAlmostEqual(
            model.params.config.state_bounds,
            {
                "flow_mol_phase": (0, 100, 1000, pyunits.mol / pyunits.s),
                "temperature": (273.15, 300, 450, pyunits.K),
                "pressure": (5e4, 1e5, 1e6, pyunits.Pa),
            },
            item_callback=_as_quantity,
        )

        assert model.params.config.phase_equilibrium_state == {
            ("Vap", "Liq"): SmoothVLE
        }

        assert isinstance(model.params.phase_equilibrium_idx, Set)
        assert len(model.params.phase_equilibrium_idx) == 2
        for i in model.params.phase_equilibrium_idx:
            assert i in ["PE1", "PE2"]

        assert model.params.phase_equilibrium_list == {
            "PE1": ["benzene", ("Vap", "Liq")],
            "PE2": ["toluene", ("Vap", "Liq")],
        }

        assert model.params.pressure_ref.value == 1e5
        assert model.params.temperature_ref.value == 300

        assert model.params.benzene.mw.value == 78.1136e-3
        assert model.params.benzene.pressure_crit.value == 48.9e5
        assert model.params.benzene.temperature_crit.value == 562.2

        assert model.params.toluene.mw.value == 92.1405e-3
        assert model.params.toluene.pressure_crit.value == 41e5
        assert model.params.toluene.temperature_crit.value == 591.8

        assert_units_consistent(model)


class TestStateBlock(object):
    @pytest.fixture(scope="class")
    def model(self):
        model = ConcreteModel()
        model.fs = FlowsheetBlock(dynamic=False)
        model.fs.params = GenericParameterBlock(**config_dict)

        model.fs.unit = FeedFlash(property_package=model.fs.params)

        # Fix state
        print()
        print(degrees_of_freedom(model.fs.unit))
        phase_frac = 0.5
        x_liq_benz = 0.5
        x_vap_benz = 0.5

        model.fs.unit.temperature.fix(368)
        model.fs.unit.pressure.fix(101325)

        model.fs.unit.flow_mol_phase[0, "Liq"].fix(1.0 - phase_frac)
        model.fs.unit.flow_mol_phase[0, "Vap"].fix(phase_frac)
        model.fs.unit.mole_frac_phase_comp[0, "Liq", "benzene"].set_value(x_liq_benz)
        model.fs.unit.mole_frac_phase_comp[0, "Liq", "toluene"].set_value(1 - x_liq_benz)
        model.fs.unit.mole_frac_phase_comp[0, "Vap", "benzene"].set_value(x_vap_benz)
        model.fs.unit.mole_frac_phase_comp[0, "Vap", "toluene"].set_value(1 - x_vap_benz)

        print()
        print(degrees_of_freedom(model.fs.unit))

        return model

    @pytest.mark.unit
    def test_build(self, model):
        # Check state variable values and bounds
        assert isinstance(model.fs.unit.flow_mol_phase, Var)
        # for p in model.props[1].flow_mol_phase:
        #     if p == 'Liq':
        #         assert value(model.props[1].flow_mol_phase[p]) == 1.0 - 0.3961
        #         assert model.props[1].flow_mol_phase[p].ub == 1000
        #         assert model.props[1].flow_mol_phase[p].lb == 0

        assert isinstance(model.fs.unit.pressure, Var)
        assert value(model.fs.unit.pressure[0]) == 101325
        assert model.fs.unit.pressure[0].ub == 1e6
        assert model.fs.unit.pressure[0].lb == 5e4

        assert isinstance(model.fs.unit.temperature, Var)
        assert value(model.fs.unit.temperature[0]) == 368
        assert model.fs.unit.temperature[0].ub == 450
        assert model.fs.unit.temperature[0].lb == 273.15

        assert isinstance(model.fs.unit.mole_frac_phase_comp, Var)
        assert len(model.fs.unit.mole_frac_phase_comp) == 4
        # for p, i in model.props[1].mole_frac_phase_comp:
        #     if p == 'Liq' and i == 'benzene':
        #         assert value(model.props[1].mole_frac_phase_comp[p, i]) == 0.4121

        assert_units_consistent(model)

    # @pytest.mark.unit
    # def test_define_state_vars(self, model):
    #     sv = model.props[1].define_state_vars()
    #
    #     assert len(sv) == 4
    #     for i in sv:
    #         print(i)
    #         assert i in ["flow_mol_phase", "mole_frac_phase_comp", "temperature", "pressure"]
    #
    # @pytest.mark.unit
    # def test_define_port_members(self, model):
    #     sv = model.props[1].define_state_vars()
    #
    #     assert len(sv) == 4
    #     for i in sv:
    #         assert i in ["flow_mol_phase", "mole_frac_phase_comp", "temperature", "pressure"]
    #
    # @pytest.mark.unit
    # def test_define_display_vars(self, model):
    #     sv = model.props[1].define_display_vars()
    #
    #     assert len(sv) == 4
    #     for i in sv:
    #         assert i in [
    #             "Total Molar Flowrate",
    #             "Total Mole Fraction",
    #             "Temperature",
    #             "Pressure",
    #         ]

    @pytest.mark.unit
    def test_dof(self, model):
        assert degrees_of_freedom(model.fs.unit) == 0

    @pytest.mark.unit
    def test_basic_scaling(self, model):
        model.fs.unit.scaling_factor.display()

        assert len(model.fs.unit.scaling_factor) == 15
        # assert (
        #     model.props[1].scaling_factor[
        #         model.props[1]._mole_frac_tbub["Vap", "Liq", "benzene"]
        #     ]
        #     == 1000
        # )
        # assert (
        #     model.props[1].scaling_factor[
        #         model.props[1]._mole_frac_tbub["Vap", "Liq", "toluene"]
        #     ]
        #     == 1000
        # )
        # assert (
        #     model.props[1].scaling_factor[
        #         model.props[1]._mole_frac_tdew["Vap", "Liq", "benzene"]
        #     ]
        #     == 1000
        # )
        # assert (
        #     model.props[1].scaling_factor[
        #         model.props[1]._mole_frac_tdew["Vap", "Liq", "toluene"]
        #     ]
        #     == 1000
        # )
        # assert model.props[1].scaling_factor[model.props[1]._t1_Vap_Liq] == 1e-2
        # assert model.props[1].scaling_factor[model.props[1]._teq["Vap", "Liq"]] == 1e-2
        assert model.fs.unit.scaling_factor[model.fs.unit.flow_mol] == 1e-2
        assert (
            model.fs.unit.scaling_factor[model.fs.unit.flow_mol_phase["Liq"]] == 1e-2
        )
        assert (
            model.fs.unit.scaling_factor[model.fs.unit.flow_mol_phase["Vap"]] == 1e-2
        )
        assert (
            model.fs.unit.scaling_factor[
                model.fs.unit.flow_mol_phase_comp["Liq", "benzene"]
            ]
            == 1e-2
        )
        assert (
            model.fs.unit.scaling_factor[
                model.fs.unit.flow_mol_phase_comp["Liq", "toluene"]
            ]
            == 1e-2
        )
        assert (
            model.fs.unit.scaling_factor[
                model.fs.unit.flow_mol_phase_comp["Vap", "benzene"]
            ]
            == 1e-2
        )
        assert (
            model.fs.unit.scaling_factor[
                model.fs.unit.flow_mol_phase_comp["Vap", "toluene"]
            ]
            == 1e-2
        )
        assert (
            model.fs.unit.scaling_factor[model.fs.unit.mole_frac_comp["benzene"]]
            == 1000
        )
        assert (
            model.fs.unit.scaling_factor[model.fs.unit.mole_frac_comp["toluene"]]
            == 1000
        )
        assert (
            model.fs.unit.scaling_factor[
                model.fs.unit.mole_frac_phase_comp["Liq", "benzene"]
            ]
            == 1000
        )
        assert (
            model.fs.unit.scaling_factor[
                model.fs.unit.mole_frac_phase_comp["Liq", "toluene"]
            ]
            == 1000
        )
        assert (
            model.fs.unit.scaling_factor[
                model.fs.unit.mole_frac_phase_comp["Vap", "benzene"]
            ]
            == 1000
        )
        assert (
            model.fs.unit.scaling_factor[
                model.fs.unit.mole_frac_phase_comp["Vap", "toluene"]
            ]
            == 1000
        )
        assert model.fs.unit.scaling_factor[model.fs.unit.pressure] == 1e-5
        assert model.fs.unit.scaling_factor[model.fs.unit.temperature] == 1e-2
        # assert (
        #     model.props[1].scaling_factor[
        #         model.props[1].temperature_bubble["Vap", "Liq"]
        #     ]
        #     == 1e-2
        # )
        # assert (
        #     model.props[1].scaling_factor[model.props[1].temperature_dew["Vap", "Liq"]]
        #     == 1e-2
        # )

    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, model):
        orig_fixed_vars = fixed_variables_set(model)
        orig_act_consts = activated_constraints_set(model)

        model.fs.unit.initialize(optarg={"tol": 1e-6})

        assert degrees_of_freedom(model) == 0

        fin_fixed_vars = fixed_variables_set(model)
        fin_act_consts = activated_constraints_set(model)

        assert len(fin_act_consts) == len(orig_act_consts)
        assert len(fin_fixed_vars) == len(orig_fixed_vars)

        for c in fin_act_consts:
            assert c in orig_act_consts
        for v in fin_fixed_vars:
            assert v in orig_fixed_vars

    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, model):
        results = solver.solve(model)

        # Check for optimal solution
        assert check_optimal_termination(results)

    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, model):
        # Check phase equilibrium results
        print(model.fs.unit.mole_frac_phase_comp[
            0, "Liq", "benzene"
        ].value)
        print(model.fs.unit.mole_frac_phase_comp[
            0, "Vap", "benzene"
        ].value)
        assert model.fs.unit.mole_frac_phase_comp[
            0, "Liq", "benzene"
        ].value == pytest.approx(0.4121, abs=1e-4)
        assert model.fs.unit.mole_frac_phase_comp[
            0, "Vap", "benzene"
        ].value == pytest.approx(0.6339, abs=1e-4)
        assert model.fs.unit.phase_frac[0, "Vap"].value == pytest.approx(0.3961, abs=1e-4)

        assert value(
            model.fs.unit.conc_mol_phase_comp[0, "Vap", "benzene"]
        ) == pytest.approx(20.9946, abs=1e-4)

    @pytest.mark.ui
    @pytest.mark.unit
    def test_report(self, model):
        model.fs.unit.report()
