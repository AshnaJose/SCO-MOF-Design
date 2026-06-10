from aiida.engine import run
from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group, Float
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
import copy
from aiida import engine
from aiida.engine import WorkChain
from aiida import orm
import aiida_quantumespresso

@engine.calcfunction
def get_energy(output_params):
    return orm.Float(output_params['energy'])

@engine.calcfunction
def get_tot_magnetization(output_params):
    return orm.Float(output_params['total_magnetization'])

class MOFSCO_Workchain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("params_scf", valid_type=orm.Dict)
        spec.input("code", valid_type=orm.Code)
        spec.input("pseudo_family", valid_type=orm.Str)
        spec.input("group", valid_type=orm.Str)
        spec.input("mof", valid_type=orm.Str)
        spec.input("hs_total_magnetization", valid_type=orm.Float)
        spec.input("num_machines", valid_type=orm.Int)
        spec.input("max_iterations", valid_type=orm.Int)
        spec.input("walltime_scf", valid_type=orm.Int)

        spec.output("low_spin_structure", valid_type=orm.StructureData)
        spec.output("high_spin_structure", valid_type=orm.StructureData)
        spec.output("low_spin_energy_scf", valid_type=orm.Float)
        spec.output("high_spin_energy_scf", valid_type=orm.Float)
        spec.output("low_spin_tot_mag_scf", valid_type=orm.Float)
        spec.output("high_spin_tot_mag_scf", valid_type=orm.Float)

        spec.outline(
            cls.scf_calculation,
            cls.get_outputs,
        )

    def scf_calculation(self):
        scf_params_low = self.inputs.params_scf
        code = self.inputs.code
        pseudo_family = load_group(self.inputs.pseudo_family.value)
        group = load_group(label=self.inputs.group.value)

        calc_low_spin = self.submit(PwBaseWorkChain,
            kpoints_distance=orm.Float(0.5),
            max_iterations=self.inputs.max_iterations,
            pw={
                'code': code,
                'parameters': scf_params_low,
                'pseudos': pseudo_family.get_pseudos(structure=self.inputs.structure),
                'structure': self.inputs.structure,
                'metadata': {'options': {
                    'resources': {'num_machines': self.inputs.num_machines.value, 'num_mpiprocs_per_machine': 48},
                    'max_wallclock_seconds': self.inputs.walltime_scf.value,
                    'withmpi': True,
                    'account': 'engmof',
                    'input_filename': self.inputs.mof.value + '_ls.in',
                    'output_filename': self.inputs.mof.value + '_ls.out',
                }}
            })
        calc_low_spin.base.extras.set('set', 'prediction_set')
        calc_low_spin.base.extras.set('MOF', self.inputs.mof.value)
        calc_low_spin.base.extras.set('calculation', 'scf')
        calc_low_spin.base.extras.set('spin_state', 'LS')
        group.add_nodes(calc_low_spin)

        scf_params_high = copy.deepcopy(scf_params_low)
        scf_params_high['SYSTEM']['tot_magnetization'] = self.inputs.hs_total_magnetization.value

        calc_high_spin = self.submit(PwBaseWorkChain,
            kpoints_distance=orm.Float(0.5),
            max_iterations=self.inputs.max_iterations,
            pw={
                'code': code,
                'parameters': scf_params_high,
                'pseudos': pseudo_family.get_pseudos(structure=self.inputs.structure),
                'structure': self.inputs.structure,
                'metadata': {'options': {
                    'resources': {'num_machines': self.inputs.num_machines.value, 'num_mpiprocs_per_machine': 48},
                    'max_wallclock_seconds': self.inputs.walltime_scf.value,
                    'withmpi': True,
                    'account': 'engmof',
                    'input_filename': self.inputs.mof.value + '_hs.in',
                    'output_filename': self.inputs.mof.value + '_hs.out',
                }}
            })
        calc_high_spin.base.extras.set('set', 'prediction_set')
        calc_high_spin.base.extras.set('MOF', self.inputs.mof.value)
        calc_high_spin.base.extras.set('calculation', 'scf')
        calc_high_spin.base.extras.set('spin_state', 'HS')
        group.add_nodes(calc_high_spin)

        return engine.ToContext(calc_scf_low=calc_low_spin, calc_scf_high=calc_high_spin)

    def get_outputs(self):
        self.out("low_spin_structure", self.inputs.structure)
        self.out("high_spin_structure", self.inputs.structure)
        self.out("low_spin_tot_mag_scf", get_tot_magnetization(self.ctx.calc_scf_low.outputs.output_parameters))
        self.out("high_spin_tot_mag_scf", get_tot_magnetization(self.ctx.calc_scf_high.outputs.output_parameters))
        self.out("low_spin_energy_scf", get_energy(self.ctx.calc_scf_low.outputs.output_parameters))
        self.out("high_spin_energy_scf", get_energy(self.ctx.calc_scf_high.outputs.output_parameters))