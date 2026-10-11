"""BAR label isolation, persistence, scoring, and projection geometry."""
import array
import json
import math
import pathlib
import re
import subprocess
import tempfile
import unittest

from test_joint import dataset, run


class BarTests(unittest.TestCase):
    def test_probe_labels_never_affect_training(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            labels = dataset(p/'a.bin', correlated_train=True)
            dataset(p/'b.bin', change_fold=0, correlated_train=True)
            for name in ('a', 'b'):
                result = run('bar', '--data', p/(name+'.bin'), '--model', p/(name+'.model'),
                             '--predictions', p/(name+'.f32'), '--attributes', 2, '--movies', 3,
                             '--iterations', 80, '--eval-every', 10, '--threads', 2)
                self.assertIn('count=200', result.stdout)
            self.assertEqual((p/'a.model').read_bytes()[16:], (p/'b.model').read_bytes()[16:])
            self.assertEqual((p/'a.f32').read_bytes(), (p/'b.f32').read_bytes())
            reload = run('eval-bar', '--data', p/'a.bin', '--model', p/'a.model',
                         '--predictions', p/'reload.f32')
            self.assertEqual((p/'a.f32').read_bytes(), (p/'reload.f32').read_bytes())
            pred = array.array('f'); pred.frombytes((p/'a.f32').read_bytes())
            self.assertTrue(all(math.isfinite(x) for x in pred))
            expected = math.sqrt(sum((r-min(5, max(1, x)))**2
                                     for x, (_, _, _, r) in zip(pred, labels))/len(labels))
            reported = float(re.search(r'probe_rmse=(\S+)', reload.stdout)[1])
            self.assertAlmostEqual(expected, reported, places=8)
            independent = subprocess.check_output([
                'python3', 'scripts/score_probe.py', '--data', str(p/'a.bin'),
                '--prediction', str(p/'a.f32')], text=True)
            self.assertAlmostEqual(json.loads(independent)['predictions'][0]['clipped_rmse'],
                                   reported, places=8)
            run('eval-bar', '--data', p/'b.bin', '--model', p/'a.model', ok=False)

    def test_projection_and_weight_solver(self):
        # Compile a small harness against the actual implementation. Known geometry
        # and independent normal-equation residuals catch numeric implementation errors.
        source = pathlib.Path('src/netflix.cpp').resolve()
        code = r'''
#define main netflix_main
#include "SOURCE"
#undef main
#include <cassert>
int main() {
  float w[2] = {1,0}, x[2] = {0,1}, aw[2], ax[2];
  bar_project(w,x,aw,ax,2,true,0.25);
  assert(aw[0]==1 && aw[1]==0 && ax[0]==0 && ax[1]==1);
  bar_project(w,x,aw,ax,2,true,0.25,false);
  assert(std::abs(double(aw[0])*ax[0]+double(aw[1])*ax[1]-.25) < 1e-6);
  bar_project(w,x,aw,ax,2,false,4);
  assert(aw[0]==1 && ax[1]==1);
  for (double target : {-5.,-1.,1.,5.}) {
    bar_project(w,x,aw,ax,2,true,target);
    double dot = double(aw[0])*ax[0]+double(aw[1])*ax[1];
    assert(std::abs(dot-(target-std::copysign(.5,target))) < 1e-5);
    // Symmetry of this input gives the analytic nearest-point form.
    assert(std::abs(aw[0]-ax[1]) < 1e-7 && std::abs(aw[1]-ax[0]) < 1e-7);
  }
  Dataset d; d.header.users=4; d.header.items=2; d.header.mean=3;
  d.train_offsets={0,2,4,6,8}; d.train_items={1,2,1,2,1,2,1,2};
  d.train_ratings={1,2,2,4,3,3,4,5};
  auto m=bar_baseline(d,2); m.bits={0,1,2,3};
  fit_bar_weights(d,m,.001);
  for (uint32_t i=0;i<2;++i) for (uint32_t k=0;k<2;++k) {
    double gradient=-.001*m.weights[i*2+k];
    for (uint32_t u=0;u<4;++u) if (m.bits[u]&(1U<<k))
      gradient+=d.train_ratings[u*2+i]-bar_prediction(m,u,i);
    assert(std::abs(gradient)<1e-8);
  }
}
'''.replace('SOURCE', str(source))
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            (p/'check.cpp').write_text(code)
            subprocess.run(['c++', '-O2', '-std=c++17', str(p/'check.cpp'), '-o', str(p/'check')], check=True)
            subprocess.run([str(p/'check')], check=True)


if __name__ == '__main__':
    unittest.main()
