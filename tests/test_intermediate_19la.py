"""Check the actual runner dispatch without importing GPU/training processes."""
import ast
from pathlib import Path
import unittest


class Intermediate19LATests(unittest.TestCase):
    def test_actual_runner_routes_all_scheduled_evaluations_to_19LA(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'tools/train_coverage_experiment.py').read_text())
        trainer=next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=='AuditedTrainer')
        method=next(node for node in trainer.body if isinstance(node,ast.FunctionDef) and node.name=='evaluation')
        namespace={}
        module=ast.Module(body=[method],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),'actual_runner_method','exec'),namespace)
        calls=[]
        class Probe:
            def test(self,phase,epoch):
                calls.append((phase,epoch));return (7.,0,0,0)
        probe=Probe()
        for epoch in range(5,51,5):
            self.assertEqual(namespace['evaluation'](probe,epoch),(7.,0,0,0))
        self.assertEqual(calls,[('19LA',epoch) for epoch in range(5,51,5)])
        # The hook only chooses the evaluation dataset; it cannot update models
        # or change checkpoint selection, training epochs or optimizer state.
        self.assertEqual(len(method.body),1)
        self.assertIsInstance(method.body[0],ast.Return)


if __name__=='__main__':unittest.main()
