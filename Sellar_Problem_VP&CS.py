import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2

class SellarMDAPromoteConnect(om.Group):
    """
    Group containing the Sellar MDA. This version uses the discipline without derivatives.
    """

    def setup(self):
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 1.0)
        indeps.add_output('z', np.array([5.0, 2.0]))

        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('d1', SellarDis1())
        cycle.add_subsystem('d2', SellarDis2())
        cycle.connect('d1.y1', 'd2.y1')
        cycle.connect('d2.y2', 'd1.y2')

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS()

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)', z=np.array([0.0, 0.0]), x=0.0))
        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

        self.connect('x', ['d1.x', 'obj_cmp.x'])
        self.connect('z', ['d1.z', 'd2.z', 'obj_cmp.z'])
        self.connect('d1.y1', ['con_cmp1.y1', 'obj_cmp.y1'])
        self.connect('d2.y2', ['con_cmp2.y2', 'obj_cmp.y2'])

prob = om.Problem()
prob.model = SellarMDAPromoteConnect()

prob.setup()

prob['x'] = 2.
prob['z'] = [-1., -1.]

prob.run_model()

print((prob['d1.y1'][0], prob['d2.y2'][0], prob['obj_cmp.obj'][0], prob['con_cmp1.con1'][0], prob['con_cmp2.con2'][0]))