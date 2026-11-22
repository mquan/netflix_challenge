"""Independent family-domain reservations and identity/orphan-stop fixtures."""
import copy
import unittest
from unittest.mock import patch
import upstream_full_context_catboost_family as module


class FamilyTests(unittest.TestCase):
    def fixture(self):
        g=module.GiB
        workers={'10':dict(bytes=1*g,rss_bytes=2*g),'11':dict(bytes=2*g,rss_bytes=3*g),
                 '12':dict(bytes=3*g,rss_bytes=5*g),'99':dict(bytes=5*g,rss_bytes=6*g)}
        return workers,{10,11,12},dict(compressor_bytes=0,swap_used_bytes=100),dict(swap_used_bytes=100)

    def captured(self,pid,**extra):
        return dict(pid=pid,started='original-start',pgid=10,state='Ss',command='original-worker',**extra)

    def test_full_family_sum_and_independent_domain_padding(self):
        workers,group,current,baseline=self.fixture();g=module.GiB
        self.assertEqual(module.usage(workers,group),dict(bytes=6*g,rss_bytes=10*g))
        padded=module.padded(workers,group,10)
        self.assertEqual(sum(v['bytes'] for v in padded.values()),21*g)
        self.assertEqual(sum(v['rss_bytes'] for v in padded.values()),22*g)
        self.assertEqual(module.usage(padded,group),dict(bytes=16*g,rss_bytes=16*g))
        self.assertEqual(workers['12']['rss_bytes'],5*g)
        self.assertTrue(module.readmission(workers,group,10,current,baseline,module.FLOOR+module.RESERVE))
        workers['99']['bytes']=8*g
        self.assertFalse(module.readmission(workers,group,10,current,baseline,module.FLOOR+module.RESERVE))
        workers['99']['bytes']=5*g;workers['99']['rss_bytes']=8*g
        self.assertFalse(module.readmission(workers,group,10,current,baseline,module.FLOOR+module.RESERVE))

    def test_group_overcap_cannot_hide_behind_each_small_role(self):
        workers,group,current,baseline=self.fixture();g=module.GiB
        for pid in group:workers[str(pid)].update(bytes=6*g,rss_bytes=6*g)
        self.assertFalse(module.safe(workers,group,current,baseline,100*g))
        with self.assertRaises(RuntimeError):module.padded(workers,group,10)
        workers['10']['bytes']=-1
        with self.assertRaises(RuntimeError):module.usage(workers,group)

    def test_original_compressor_swap_and_floor_boundaries(self):
        workers,group,current,baseline=self.fixture();g=module.GiB;free=module.FLOOR+module.RESERVE
        self.assertTrue(module.safe(workers,group,current,baseline,free))
        self.assertFalse(module.safe(workers,group,current,baseline,free-1))
        current['swap_used_bytes']=baseline['swap_used_bytes']+128*2**20+1
        self.assertFalse(module.safe(workers,group,current,baseline,free))
        current['swap_used_bytes']=100;current['compressor_bytes']=int(4.5*g)
        self.assertTrue(module.safe(workers,group,current,baseline,free))
        self.assertFalse(module.readmission(workers,group,10,current,baseline,free))
        current['compressor_bytes']=5*g
        self.assertFalse(module.safe(workers,group,current,baseline,free))

    def test_selected_descendants_and_bootstrap_exec_identity(self):
        root=self.captured(10,worker_command='executed-worker')
        rows={10:dict(ppid=1,started=root['started'],command='executed-worker'),
              11:dict(ppid=10,started='child',command='producer'),12:dict(ppid=11,started='child2',command='helper'),
              13:dict(ppid=10,started='observer',command='ps'),99:dict(ppid=1,started='other',command='other')}
        self.assertEqual(module.members(rows,{10,11,12,99},root),{10,11,12})
        rows[10]['started']='recycled'
        with self.assertRaises(RuntimeError):module.members(rows,{10,11,12,99},root)

    def test_native_family_sample_rejects_peer_ancestry_identity_or_group_races(self):
        workers,group,_,_=self.fixture()
        rows={pid:dict(ppid=(1 if pid in (10,99) else 10),started='original-start',
                      command='python upstream_role_%s.py'%pid) for pid in (10,11,12,99)}
        rows[13]=dict(ppid=10,started='observer',command='ps -axo pid')
        captured=dict(self.captured(10),command=rows[10]['command'])
        role=dict(self.captured(11),command=rows[11]['command'])
        current={10:captured,11:role}
        def sample(after,counts=workers,roles=None):
            with patch.object(module.resources.native,'processes',side_effect=[rows,after]), \
                 patch.object(module.resources,'footprints',return_value=counts), \
                 patch.object(module.identity,'process',side_effect=lambda pid:current.get(pid)):
                return module.measured(captured,roles)
        observed,members=sample(copy.deepcopy(rows),roles={'producer':role})
        self.assertEqual(observed,workers);self.assertEqual(members,group)
        changed=copy.deepcopy(rows);changed[12]['ppid']=1
        with self.assertRaises(RuntimeError):sample(changed)
        changed=copy.deepcopy(rows);changed[12]['started']='recycled'
        with self.assertRaises(RuntimeError):sample(changed)
        changed=copy.deepcopy(rows);changed[14]=dict(ppid=10,started='new',command='python upstream_new.py')
        with self.assertRaises(RuntimeError):sample(changed)
        partial=dict(workers);partial.pop('99')
        with self.assertRaises(RuntimeError):sample(copy.deepcopy(rows),counts=partial)
        current[11]=dict(role,pgid=11)
        with self.assertRaises(RuntimeError):sample(copy.deepcopy(rows),roles={'producer':role})

    def test_root_loss_preserves_surviving_role_and_never_signals_recycled_group(self):
        root=self.captured(10);role=self.captured(11)
        with patch.object(module.identity,'process',side_effect=lambda pid:None if pid==10 else role),patch.object(module.os,'killpg') as stop:
            self.assertTrue(module.stop_family(root,{'consumer':role}))
            stop.assert_called_once_with(10,module.signal.SIGSTOP)
        recycled=dict(role,started='new-start')
        with patch.object(module.identity,'process',return_value=recycled),patch.object(module.os,'killpg') as stop:
            self.assertFalse(module.stop_family(root,{'consumer':role}));stop.assert_not_called()


if __name__=='__main__':unittest.main()
