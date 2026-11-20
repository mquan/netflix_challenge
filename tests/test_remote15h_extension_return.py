import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_upstream_remote15h_extension_return as importer

class OriginalExtensionRecipeGuards(unittest.TestCase):
    def setUp(self):
        self.plan=importer.helper.read(importer.u.ROOT/'experiments/upstream-remote15h-ext-return/s3-bknbrx-100w-v1/plan.json')
        self.base=importer.helper.read(importer.helper.PLAN)
        self.registry=importer.helper.read(importer.u.ROOT/'experiments/upstream-all-pipeline-coverage/registry.json')['3']['jobs']

    def test_all_eight_actual_original_assignments_match(self):
        for job in self.plan['jobs']:
            with self.subTest(job=job['name']):importer.validate_plan(self.plan,self.base,self.registry,job['name'])

    def test_reduced_budget_or_neighborhood_rejected(self):
        for key,value in [('EPOCHS','1'),('NBR','1')]:
            altered=copy.deepcopy(self.plan);altered['jobs'][0]['environment'][key]=value
            with self.subTest(parameter=key),self.assertRaises(ValueError):
                importer.validate_plan(altered,self.base,self.registry,altered['jobs'][0]['name'])

    def test_changed_known_dataset_binding_rejected(self):
        altered=copy.deepcopy(self.plan);key=next(iter(altered['data_sha256']));altered['data_sha256'][key]='0'*64
        with self.assertRaises(ValueError):importer.validate_plan(altered,self.base,self.registry,altered['jobs'][0]['name'])

if __name__=='__main__':unittest.main()
