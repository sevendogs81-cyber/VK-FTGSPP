#!/usr/bin/env python3
"""Single-device B4 gradient/Adam oracle, replay, resume checks."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.reference.gradients import GradientLayoutReference
from ftgspp.training.backend import _shader_dir
from ftgspp._ftgspp_vk import FTGSPPVulkan


def expect_rejected(fn):
    try:
        fn()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('invalid batch transition was accepted')


def state(native):
    return ({k:np.array(v) for k,v in native.get_dynamic_gaussians().items()},
            {k:np.array(v) for k,v in native.get_optimizer_state().items()})


def compare(actual, expected, *, skip_stats=False, atol=2e-7):
    for aa,bb in zip(actual,expected,strict=True):
        for key in bb:
            if skip_stats and key=='relocation_stats': continue
            np.testing.assert_allclose(aa[key],bb[key],rtol=2e-5,atol=atol,err_msg=key)


def zero_optimizer(native, n):
    """Reset moments to the fixture's topology, including padded SH lanes."""
    block = native.packed_gradient_layout()['sh_reorder_size']
    sh_size = 48 * ((n + block - 1) // block) * block
    sizes = dict(g_xyz_ws=6*n, g_rotations=8*n, g_scales_opacs=8*n,
        g_times_durations=4*n, g_velocities_gates=8*n,
        g_sh_coeffs_1=sh_size, g_sh_coeffs_2=sh_size, relocation_stats=2*n)
    native.set_optimizer_state({key: np.zeros(size, np.float32) for key, size in sizes.items()})


def check_real_batch_overflow(native):
    """Overflow the first view on device; later offscreen views must not hide it."""
    n, extent = 256, 512
    rng = np.random.default_rng(9)
    xyz = rng.normal(0, .01, (n, 3)).astype(np.float32)
    xyz[:, 2] += 3
    model = (xyz, np.full((n, 3), .6, np.float32),
        np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        np.full((n, 1), -2, np.float32), np.zeros((n, 16, 3), np.float32),
        np.full((n, 1), .5, np.float32), np.full((n, 1), 4, np.float32),
        np.zeros((n, 3), np.float32), np.full((n, 1), -8, np.float32))
    native.configure_ftgspp(dict(local_batch_size=4, image_objective=1,
        max_steps=1000, max_sh_degree=3, max_duration=1., scene_scale=1.,
        color_correction=False))
    native.clear_dynamic_views()
    ref = rng.integers(0, 256, (extent, extent, 4), dtype=np.uint8)
    ref[..., 3] = 255
    k = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]], np.float32)
    for i in range(2):
        pose = np.eye(4, dtype=np.float32); pose[0, 3] = 1000 * i
        native.add_dynamic_view_meta(extent, extent, pose, k, .5, i, 0, True)

    def reset():
        native.set_dynamic_gaussians(*model)
        zero_optimizer(native, n)

    def attempt(collect):
        native.begin_batch(1, collect)
        for view in (0, 1, 1, 1):
            native.bind_dynamic_view_image(view, True, ref)
            if not native.accumulate_batch_view(view, (.2, .4, .6)):
                return False
        native.commit_batch()
        return True

    reset(); before = state(native)
    assert not attempt(False), 'fixture must exceed bootstrap tile capacity'
    compare(state(native), before, atol=0)
    assert attempt(False), 'grown capacity must complete the same batch'
    replay = state(native)
    reset(); assert attempt(True)
    compare(state(native), replay)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    native=FTGSPPVulkan()
    native.initialize(str(_shader_dir())+'/',args.device)
    try:
        check_real_batch_overflow(native)
    except BaseException:
        native.abort_batch()
        native.cleanup()
        raise
    print('real device overflow rollback/replay matches synchronized batch', flush=True)
    rng=np.random.default_rng(42); n=5
    means=np.array([[-.2,-.1,3.],[.15,.12,3.2],[.1,-.2,2.8],[.3,.1,3.5],[20,20,3]],np.float32)
    # Isotropic splats make orientation unidentifiable: near-zero quaternion
    # gradients can be dominated by FP32 reduction noise then amplified by Adam.
    # Use identifiable anisotropic geometry for the strict state-resume oracle.
    logs=np.tile(np.array([-2.0,-2.3,-2.6],np.float32),(n,1))
    rotation=np.array([.94,.1,-.2,.25],np.float32)
    rotation/=np.linalg.norm(rotation)
    quat=np.tile(rotation,(n,1))
    opacity=np.linspace(-1,1,n,dtype=np.float32)[:,None]
    sh=rng.normal(0,.1,(n,16,3)).astype(np.float32)
    times=np.linspace(.3,.7,n,dtype=np.float32)[:,None]
    duration=np.full((n,1),-2,np.float32)
    velocity=rng.normal(0,.03,(n,3)).astype(np.float32)
    gate=np.full((n,1),-.2,np.float32)
    initial=(means,logs,quat,opacity,sh,times,duration,velocity,gate)
    config=dict(max_steps=2000,max_sh_degree=3,image_objective=1,scene_scale=1.,
        means_lr=.0001,means_lr_final=.0001,velocities_lr=.0001,velocities_lr_final=.0001,
        scales_lr=.001,quats_lr=.001,opacities_lr=.005,features_dc_lr=.001,
        features_rest_lr=.0001,times_lr=.0001,durations_lr=.001,gates_lr=.0001,
        max_duration=1.,marginal_threshold=.05,scale_reg=.01,opacity_reg=.01,gate_reg=.001,
        color_correction=False,local_batch_size=4)
    native.configure_ftgspp(config)
    native.clear_dynamic_views()
    references=[]; backgrounds=[]
    for i,(h,w) in enumerate([(40,48),(48,40),(32,56),(48,48)]):
        k=np.array([[45,0,w/2],[0,45,h/2],[0,0,1]],np.float32)
        pose=np.eye(4,dtype=np.float32); pose[0,3]=(i-1.5)*.1
        native.add_dynamic_view_meta(h,w,pose,k,.2+.2*i,i,i,True)
        im=rng.integers(0,220,(h,w,4),dtype=np.uint8)
        yy,xx=np.mgrid[:h,:w]; im[...,3]=np.where((xx-w/2)**2+(yy-h/2)**2 < 100,220,20)
        references.append(im); backgrounds.append(tuple(float(x) for x in rng.uniform(.1,.9,3)))

    def reset(batch=4):
        native.set_dynamic_gaussians(*initial)
        native.configure_ftgspp(dict(local_batch_size=batch))
        # Model upload intentionally does not reset optimizer/relocation state.
        zero_optimizer(native, n)

    def bind(i):
        native.bind_dynamic_view_image(i,True,references[i])

    def batch(step,indices=(0,1,2,3),inject=-1,collect_loss=True):
        attempts=0
        while True:
            native.begin_batch(step,collect_loss)
            for j,i in enumerate(indices):
                bind(i)
                if j==inject and attempts==0: native.inject_sync_tile_overflow(65536)
                if not native.accumulate_batch_view(i,backgrounds[i]): break
            else:
                native.commit_batch(); return attempts
            attempts+=1
            assert attempts==1

    checks=['real device overflow rollback/replay matches synchronized batch']
    try:
        # Independent per-view backward at identical theta, no optimizer.
        grads=[]; losses=[]; relocation=[]
        for i in range(4):
            reset(); bind(i)
            native.backward_dynamic_view(i,0,True,backgrounds[i])
            layout=native.packed_gradient_layout()
            g=np.empty(native.packed_gradient_size(),np.float32)
            native.gradient_state_packed_into(g)
            grads.append(g); losses.append(dict(native.get_last_loss()))
            relocation.append(np.array(native.get_optimizer_state()['relocation_stats']))
        average=np.zeros_like(grads[0])
        for g in grads: average+=g*np.float32(.25)
        canonical=GradientLayoutReference(layout,average).canonical_gradient_state()
        reset(); batch(0)
        model,opt=state(native)
        groups={
            'g_xyz_ws':canonical['means'],
            'g_scales_opacs':np.c_[canonical['log_scales'],canonical['opacity_logits']],
            'g_times_durations':np.c_[canonical['times'],canonical['duration_raw']],
            'g_velocities_gates':np.c_[canonical['velocities'],canonical['gate_raw']],
            'g_rotations':canonical['quaternions'],
        }
        for key,g in groups.items():
            moment=opt[key].reshape(n,2,-1)
            np.testing.assert_allclose(moment[:,0],.1*g,rtol=2e-5,atol=2e-8,err_msg=key+' m')
            np.testing.assert_allclose(moment[:,1],.001*g*g,rtol=3e-5,atol=2e-10,err_msg=key+' v')
        gsh=canonical['sh_coefficients']
        actual_sh=GradientLayoutReference._unpack_reordered_sh(opt['g_sh_coeffs_1'],n,layout['sh_reorder_size'])
        np.testing.assert_allclose(actual_sh,.1*gsh,rtol=2e-5,atol=2e-8)
        # First-step Adam oracle checks parameters as well as both moments.
        def update(x,g,lr): return x-lr*g/(np.abs(g)+1e-15)
        for key,x,g,lr in [
            ('means',means,canonical['means'],.0001),
            ('log_scales_opacity',np.c_[logs,opacity],np.c_[canonical['log_scales'],canonical['opacity_logits']],np.array([.001,.001,.001,.005])),
            ('times_durations',np.c_[times,duration],np.c_[canonical['times'],canonical['duration_raw']],np.array([.0001,.001])),
            ('velocities_gates',np.c_[velocity,gate],np.c_[canonical['velocities'],canonical['gate_raw']],.0001),
            ('rotations',quat,canonical['quaternions'],.001),
        ]:
            np.testing.assert_allclose(model[key],update(x,g,lr),rtol=2e-5,atol=2e-7,err_msg=key)
        np.testing.assert_allclose(opt['relocation_stats'],sum(relocation),rtol=2e-5,atol=2e-7)
        actual_loss=dict(native.get_last_loss())
        for key in actual_loss:
            np.testing.assert_allclose(actual_loss[key],np.mean([x[key] for x in losses]),rtol=2e-5,atol=2e-7)
        checks.append('four distinct views gradient/Adam/loss/relocation oracle')
        print(checks[-1],flush=True)

        reset(batch=1); bind(0); native.train_dynamic_view(0,0,True,backgrounds[0]); fused=state(native)
        reset(batch=1); bind(0); native.backward_dynamic_view(0,0,True,backgrounds[0]); native.optimizer_step(0); split=state(native)
        compare(split,fused,skip_stats=True)
        reset(); batch(0,(0,0,0,0)); repeated=state(native)
        compare(repeated,fused,skip_stats=True)
        checks.append('fused B1 = split B1 = repeated-view B4')
        print(checks[-1],flush=True)

        reset(); batch(0); control=state(native)
        for slot in range(4):
            reset(); before=state(native)
            native.begin_batch(0,True)
            for i in range(slot+1):
                bind(i)
                if i==slot: native.inject_sync_tile_overflow(65536)
                ok=native.accumulate_batch_view(i,backgrounds[i])
                assert ok == (i!=slot)
            compare(state(native),before,atol=0)
            batch(0); compare(state(native),control)
        checks.append('overflow rollback/replay at each of four microviews')
        print(checks[-1],flush=True)

        reset()
        for step in range(4):
            batch(step)
            if step==1: saved=state(native); saved_raster=native.rasterizer_state()
        continuous=state(native)
        m,o=saved
        native.set_dynamic_gaussians(m['means'],m['log_scales_opacity'][:,:3],m['rotations'],
            m['log_scales_opacity'][:,3:],m['sh_coeffs'],m['times_durations'][:,:1],
            m['times_durations'][:,1:],m['velocities_gates'][:,:3],m['velocities_gates'][:,3:])
        native.set_optimizer_state(o); native.restore_rasterizer_state(saved_raster)
        for step in range(2,4): batch(step)
        compare(state(native),continuous)
        checks.append('model/moments/relocation/rasterizer resume')

        reset(); expect_rejected(native.commit_batch)
        expect_rejected(lambda:native.accumulate_batch_view(0,backgrounds[0]))
        native.begin_batch(0,True)
        for fn in [native.commit_batch,lambda:native.begin_batch(0),lambda:native.optimizer_step(0),
                   native.get_optimizer_state,native.get_dynamic_gaussians,
                   lambda:native.set_dynamic_gaussians(*initial)]: expect_rejected(fn)
        for i in range(4): bind(i); assert native.accumulate_batch_view(i,backgrounds[i])
        expect_rejected(lambda:native.accumulate_batch_view(0,backgrounds[0]))
        native.commit_batch(); expect_rejected(native.commit_batch)
        native.begin_batch(1); native.abort_batch()
        checks.append('illegal transitions, partial checkpoint, duplicate update rejection')

        reset(); batch(250) # 1000 exposures enables SH1, not SH0 at update 250.
        assert native.get_workload_counters()['active_sh_degree']==1
        checks.append('exposure-based SH schedule, mixed image extents and padded SH')

        # Direct accumulation must preserve every parameter group at all SH
        # specializations, including inactive channels and the padded last block.
        # Obtain an independent mean from unchanged raw single-view backward.
        for step in (249, 250, 499, 500, 749, 750):
            gradients = []
            for i in range(4):
                reset(); bind(i)
                native.backward_dynamic_view(i, step, False, backgrounds[i])
                layout = native.packed_gradient_layout()
                raw = np.empty(native.packed_gradient_size(), np.float32)
                native.gradient_state_packed_into(raw)
                gradients.append(raw)
            mean = np.zeros_like(gradients[0])
            for raw in gradients:
                mean += raw * np.float32(.25)
            expected = GradientLayoutReference(layout, mean).canonical_gradient_state()
            reset(); batch(step, collect_loss=False)
            _, opt = state(native)
            for key, gradient in {
                'g_xyz_ws': expected['means'],
                'g_scales_opacs': np.c_[expected['log_scales'], expected['opacity_logits']],
                'g_times_durations': np.c_[expected['times'], expected['duration_raw']],
                'g_velocities_gates': np.c_[expected['velocities'], expected['gate_raw']],
                'g_rotations': expected['quaternions'],
            }.items():
                moments = opt[key].reshape(n, 2, -1)
                np.testing.assert_allclose(moments[:, 0], .1 * gradient, rtol=2e-5, atol=2e-8)
                np.testing.assert_allclose(moments[:, 1], .001 * gradient**2, rtol=3e-5, atol=2e-10)
            for key, value in (('g_sh_coeffs_1', .1 * expected['sh_coefficients']),
                               ('g_sh_coeffs_2', .001 * expected['sh_coefficients']**2)):
                unpacked = GradientLayoutReference._unpack_reordered_sh(
                    opt[key], n, layout['sh_reorder_size'])
                np.testing.assert_allclose(unpacked, value, rtol=3e-5, atol=2e-8)
                # Padded lanes must remain canonical zero, not stale allocation data.
                block = opt[key].reshape(-1, 12, layout['sh_reorder_size'], 4)
                assert not np.any(block[-1, :, n % layout['sh_reorder_size']:, :])
        checks.append('independent mean/moments oracle across all SH activation boundaries')
        print(checks[-1], flush=True)

        # Abort after real accumulation, with nonzero Adam state. The next batch
        # uses different views/background order so stale means cannot cancel out.
        reset(); batch(750); saved = state(native)
        saved_raster = native.rasterizer_state()
        order = (3, 1, 0, 2)
        batch(751, order, collect_loss=False); expected = state(native)
        m, o = saved
        for consumed in (1, 3, 4):
            native.set_dynamic_gaussians(m['means'], m['log_scales_opacity'][:, :3], m['rotations'],
                m['log_scales_opacity'][:, 3:], m['sh_coeffs'], m['times_durations'][:, :1],
                m['times_durations'][:, 1:], m['velocities_gates'][:, :3], m['velocities_gates'][:, 3:])
            native.set_optimizer_state(o); native.restore_rasterizer_state(saved_raster)
            native.begin_batch(751, False)
            for i in range(consumed):
                bind(i); assert native.accumulate_batch_view(i, backgrounds[i])
            # Even four successfully accumulated views must not update theta or
            # moments until commit; abort restores the complete pre-batch state.
            np.testing.assert_array_equal(np.asarray(native.base_xyz_ws), m['means'])
            native.abort_batch()
            compare(state(native), saved, atol=0)
            batch(751, order, collect_loss=False)
            compare(state(native), expected)
        assert 'local_batch_gradient' not in native.get_vram_breakdown()
        checks.append('aborted nonzero batch overwritten; one gradient allocation, no mean copy')
        print(checks[-1], flush=True)

        # Releasing explicit validity buffers while a view is still queued must
        # wait for their final GPU read. Alternate explicit and implicit masks
        # across repeated ring reuse, then compare with per-view loss waits.
        outputs = []
        for collect in (False, True):
            reset()
            for step in range(750, 754):
                native.begin_batch(step, collect)
                for i in range(4):
                    bind(i)
                    if (step + i) % 2:
                        mask = np.ones(references[i].shape[:2], np.uint8)
                        mask[:, :3] = 0
                        native.bind_dynamic_view_validity(i, True, mask)
                    assert native.accumulate_batch_view(i, backgrounds[i])
                    native.release_dynamic_view_image(i, True)
                native.commit_batch()
            outputs.append(state(native))
        compare(*outputs)
        checks.append('queued image/validity release and repeated ring reuse match logged batches')
        print(checks[-1], flush=True)
        result={'passed':True,'device':args.device,'checks':checks,
                'vram_bytes':native.get_vram_usage(),'peak_vram_bytes':native.get_peak_vram_usage()}
    finally:
        native.abort_batch()
        native.cleanup()
        assert native.get_vram_usage() == 0, 'cleanup leaked batch buffers'
    checks.append('cleanup releases all batch allocations')
    print(json.dumps(result,indent=2),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2)+'\n'); tmp.replace(args.output)


if __name__=='__main__': main()
