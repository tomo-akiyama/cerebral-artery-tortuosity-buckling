"""Tagged surface aggregates never fall back to measurement rows or overwrite other tags."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

_HERE=Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0,str(_HERE))
import sin_trend_paths as paths
from sin_trend_measure import measurement_row_schema


def _load(name):
    spec=importlib.util.spec_from_file_location('test_'+name[:-3],_HERE/name)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

agg=_load('32_create_long_format_csv.py')
analysis=_load('33_prepare_core_inplane_dataset.py')


class TestSurfaceTagAggregation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.run='synthetic_tag_test'; self.frame='baseline'
        paths.write_run_config(self.run,{'voxel_size':.5},project_root=self.root)
        paths.atomic_to_csv(pd.DataFrame({'directory_id':['SYNTHETIC_A','SYNTHETIC_B']}),paths.participants_csv_path(self.root),index=False)
        self.base=measurement_row_schema()
        self.base.update(directory_id='SYNTHETIC_A',segment_id=7,segment_index=0,frame_reference='baseline',
                         voxel_size_mm=.5,fit_status='ok',surface_evaluation_status='not_run',
                         interval_eligible=True,fit_valid=True,core_all=False,core_periodic=False,core_inplane=False,
                         surface_evaluable=False,surface_constraint='',periodicity='',lambda_hat_mm=10.)
        for did in ('SYNTHETIC_A','SYNTHETIC_B'):
            row=dict(self.base,directory_id=did)
            paths.atomic_to_csv(pd.DataFrame([row]),paths.per_case_csv_path(self.run,self.frame,did,self.root),index=False)
            paths.atomic_write_json({'status':'complete'},paths.participant_status_path(self.run,self.frame,did,self.root))
        paths.atomic_write_json({'sentinel':'measurement manifest','frames':{}},paths.run_manifest_path(self.run,self.root))
        self.manifest_before=paths.run_manifest_path(self.run,self.root).read_bytes()

    def surface(self,tag,*,ids=('SYNTHETIC_A','SYNTHETIC_B'),lam=20.,status='complete'):
        record=paths.write_surface_config(self.run,{'voxel_size':.5,'R_out_thresh':.25},surface_tag=tag,project_root=self.root)
        for did in ids:
            row=dict(self.base,directory_id=did,lambda_hat_mm=lam,surface_tag=tag,
                     surface_config_hash=record['config_hash'],surface_code_hash=record['surface_source_code_hash_combined'],
                     surface_source_measurement_csv_sha256=paths.sha256_file(paths.per_case_csv_path(self.run,self.frame,did,self.root)),
                     surface_evaluation_status='ok')
            paths.atomic_to_csv(pd.DataFrame([row]),paths.surface_per_case_csv_path(self.run,self.frame,did,tag,self.root),index=False)
            paths.atomic_write_json(dict(status=status,surface_tag=tag,surface_config_hash=record['config_hash'],
                                        per_case_csv_sha256=paths.sha256_file(paths.surface_per_case_csv_path(self.run,self.frame,did,tag,self.root))),
                                    paths.surface_participant_status_path(self.run,self.frame,did,tag,self.root))
        return record

    def refresh_surface_hash(self,tag,did='SYNTHETIC_A'):
        status_path=paths.surface_participant_status_path(self.run,self.frame,did,tag,self.root)
        status=json.loads(status_path.read_text())
        status['per_case_csv_sha256']=paths.sha256_file(paths.surface_per_case_csv_path(self.run,self.frame,did,tag,self.root))
        paths.atomic_write_json(status,status_path)

    def test_surface_csv_changed_after_status_is_rejected(self):
        self.surface('primary')
        path=paths.surface_per_case_csv_path(self.run,self.frame,'SYNTHETIC_A','primary',self.root)
        df=pd.read_csv(path); df['lambda_hat_mm']=999.; df.to_csv(path,index=False)
        result=agg.create_long_format(self.run,self.frame,self.root)
        self.assertEqual(result['rc'],6,result)
        self.assertIn('surface CSV hash',result['reason'])
        self.assertFalse(paths.long_csv_path(self.run,self.frame,self.root).exists())

    def test_changed_measurement_csv_rejects_stale_surface(self):
        self.surface('primary')
        path=paths.per_case_csv_path(self.run,self.frame,'SYNTHETIC_A',self.root)
        df=pd.read_csv(path); df['lambda_hat_mm']=99.; df.to_csv(path,index=False)
        result=agg.create_long_format(self.run,self.frame,self.root)
        self.assertEqual(result['rc'],6,result)
        self.assertIn('source measurement CSV hash mismatch',result['reason'])
        self.assertFalse(paths.long_csv_path(self.run,self.frame,self.root).exists())

    def test_missing_measurement_csv_rejects_surface(self):
        self.surface('primary')
        paths.per_case_csv_path(self.run,self.frame,'SYNTHETIC_A',self.root).unlink()
        result=agg.create_long_format(self.run,self.frame,self.root)
        self.assertEqual(result['rc'],6,result)
        self.assertIn('source measurement CSV is missing',result['reason'])

    def test_missing_or_mixed_source_hash_is_rejected(self):
        for mode in ('missing','nan','mixed'):
            with self.subTest(mode=mode):
                self.surface('primary')
                path=paths.surface_per_case_csv_path(self.run,self.frame,'SYNTHETIC_A','primary',self.root)
                df=pd.read_csv(path)
                if mode=='missing':
                    df=df.drop(columns='surface_source_measurement_csv_sha256')
                elif mode=='nan':
                    df['surface_source_measurement_csv_sha256']=float('nan')
                else:
                    second=df.copy(); second['segment_id']=8
                    second['surface_source_measurement_csv_sha256']='stale'
                    df=pd.concat([df,second],ignore_index=True)
                df.to_csv(path,index=False)
                self.refresh_surface_hash('primary')
                result=agg.create_long_format(self.run,self.frame,self.root)
                self.assertEqual(result['rc'],6,result)
                self.assertIn('source measurement CSV hash mismatch',result['reason'])

    def test_surface_status_without_csv_hash_is_rejected(self):
        self.surface('primary')
        path=paths.surface_participant_status_path(self.run,self.frame,'SYNTHETIC_A','primary',self.root)
        status=json.loads(path.read_text()); status.pop('per_case_csv_sha256')
        paths.atomic_write_json(status,path)
        result=agg.create_long_format(self.run,self.frame,self.root)
        self.assertEqual(result['rc'],6,result)
        self.assertIn('missing the surface CSV hash',result['reason'])

    def test_default_uses_primary_and_preserves_measurement_manifest(self):
        self.surface('primary')
        result=agg.create_long_format(self.run,self.frame,self.root)
        self.assertEqual(result['rc'],0,result)
        self.assertEqual(result['aggregation_stage'],'surface')
        self.assertEqual(result['surface_tag'],'primary')
        self.assertEqual(Path(result['out_csv']),paths.long_csv_path(self.run,self.frame,self.root))
        self.assertEqual(pd.read_csv(result['out_csv']).lambda_hat_mm.tolist(),[20.,20.])
        self.assertEqual(paths.run_manifest_path(self.run,self.root).read_bytes(),self.manifest_before)

    def test_nonprimary_output_is_separate(self):
        self.surface('primary',lam=20.); self.surface('gap8',lam=30.)
        first=agg.create_long_format(self.run,self.frame,self.root)
        saved=Path(first['out_csv']).read_bytes()
        second=agg.create_long_format(self.run,self.frame,self.root,surface_tag='gap8')
        self.assertEqual(second['rc'],0,second)
        self.assertEqual(Path(second['out_csv']),paths.surface_long_csv_path(self.run,self.frame,'gap8',self.root))
        self.assertEqual(pd.read_csv(second['out_csv']).lambda_hat_mm.tolist(),[30.,30.])
        self.assertEqual(Path(first['out_csv']).read_bytes(),saved)
        self.assertEqual(paths.run_manifest_path(self.run,self.root).read_bytes(),self.manifest_before)

    def test_missing_surface_case_never_falls_back_to_measurement(self):
        self.surface('primary',ids=('SYNTHETIC_A',))
        result=agg.create_long_format(self.run,self.frame,self.root)
        self.assertEqual(result['rc'],0,result)
        self.assertTrue(result['incomplete'])
        self.assertEqual(pd.read_csv(result['out_csv']).directory_id.tolist(),['SYNTHETIC_A'])

    def test_measurement_only_and_default_without_surface(self):
        no_surface=agg.create_long_format(self.run,self.frame,self.root)
        self.assertEqual(no_surface['aggregation_stage'],'measurement')
        self.surface('primary')
        result=agg.create_long_format(self.run,self.frame,self.root,measurement_only=True)
        self.assertEqual(result['surface_tag'],None)
        self.assertEqual(pd.read_csv(result['out_csv']).lambda_hat_mm.tolist(),[10.,10.])

    def test_tag_or_hash_mix_is_rejected(self):
        self.surface('primary')
        path=paths.surface_per_case_csv_path(self.run,self.frame,'SYNTHETIC_A','primary',self.root)
        for column,bad in [('surface_tag','another'),('surface_config_hash','different')]:
            self.surface('primary')
            df=pd.read_csv(path); df[column]=bad; df.to_csv(path,index=False)
            self.refresh_surface_hash('primary')
            result=agg.create_long_format(self.run,self.frame,self.root)
            self.assertEqual(result['rc'],6,result)
            self.assertFalse(paths.long_csv_path(self.run,self.frame,self.root).exists())

    def test_surface_status_not_measurement_status_controls_completeness(self):
        self.surface('primary',status='partial')
        result=agg.create_long_format(self.run,self.frame,self.root)
        self.assertEqual(result['rc'],0,result)
        self.assertTrue(result['incomplete'])

    def test_nonprimary_33_uses_selected_input_and_metadata(self):
        self.surface('gap8',lam=30.)
        result=agg.create_long_format(self.run,self.frame,self.root,surface_tag='gap8')
        self.assertEqual(result['rc'],0,result)
        with patch.object(analysis,'project_root_from_here',return_value=self.root):
            rc=analysis._main_unlocked(['--run-id',self.run,'--surface-tag','gap8','--skip-mca'])
        self.assertEqual(rc,0)
        out_dir=paths.analysis_root(self.run,self.frame,self.root)/'surface'/'gap8'
        out_csv=out_dir/paths.analysis_csv_path(self.run,self.frame,self.root).name
        self.assertTrue(out_csv.is_file())
        self.assertFalse(paths.analysis_csv_path(self.run,self.frame,self.root).exists())
        meta=json.loads((out_dir/'analysis_dataset_metadata.json').read_text())
        self.assertEqual(meta['surface_tag'],'gap8')
        self.assertEqual(meta['morphometrics_csv'],result['out_csv'])
        self.assertEqual(meta['long_format_metadata_path'],str(Path(result['out_csv']).parent/'long_format_metadata.json'))
        self.assertFalse(meta['upstream_aggregation_incomplete'])

    def test_explicit_morphometrics_csv_remains_supported(self):
        self.surface('gap8')
        result=agg.create_long_format(self.run,self.frame,self.root,surface_tag='gap8')
        custom=self.root/'custom.csv'; custom.write_bytes(Path(result['out_csv']).read_bytes())
        with patch.object(analysis,'project_root_from_here',return_value=self.root):
            rc=analysis._main_unlocked(['--run-id',self.run,'--surface-tag','gap8','--morphometrics-csv',str(custom),'--skip-mca'])
        self.assertEqual(rc,0)
        out_dir=paths.analysis_root(self.run,self.frame,self.root)/'surface'/'gap8'
        meta=json.loads((out_dir/'analysis_dataset_metadata.json').read_text())
        self.assertEqual(meta['morphometrics_csv'],str(custom))
        self.assertIsNone(meta['upstream_aggregation_incomplete'])


if __name__=='__main__':
    unittest.main()
