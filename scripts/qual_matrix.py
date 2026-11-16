"""Verified block access to qualifying matrices and one-column overlays."""
import json
from pathlib import Path
import numpy as np
from streaming_blend import digest,design


class InsertedColumn:
    def __init__(self,base,column,index):
        self.base,self.column,self.index=base,column,index
        self.shape=(len(base),base.shape[1]+1)

    def __len__(self):return self.shape[0]

    def __getitem__(self,rows):
        b=self.base[rows];v=self.column[rows];n=self.index
        if b.ndim==1:return np.r_[b[:n],v,b[n:]].astype(np.float32)
        out=np.empty((len(b),self.shape[1]),dtype=np.float32,order='F')
        out[:,:n]=b[:,:n];out[:,n]=v;out[:,n+1:]=b[:,n:]
        return out


class SelectedColumns:
    def __init__(self,base,columns):
        self.base,self.columns=base,np.asarray(columns,dtype=np.int64)
        if self.columns.ndim!=1 or not len(self.columns) or len(np.unique(self.columns))!=len(self.columns) or np.any(self.columns<0) or np.any(self.columns>=base.shape[1]):
            raise ValueError('invalid selected columns')
        self.shape=(len(base),len(self.columns))
    def __len__(self):return self.shape[0]
    def __getitem__(self,rows):return self.base[rows][...,self.columns]


class InsertedColumns:
    def __init__(self,base,columns,index):
        self.base,self.columns,self.index=base,columns,index
        self.shape=(len(base),base.shape[1]+len(columns))
        if not 0<=index<=base.shape[1] or any(len(v)!=len(base) for v in columns):
            raise ValueError('invalid column block')
    def __len__(self):return self.shape[0]
    def __getitem__(self,rows):
        b=self.base[rows];n=self.index
        if b.ndim==1:return np.r_[b[:n],[v[rows] for v in self.columns],b[n:]].astype(np.float32)
        out=np.empty((len(b),self.shape[1]),dtype=np.float32,order='F')
        out[:,:n]=b[:,:n]
        for j,v in enumerate(self.columns):out[:,n+j]=v[rows]
        out[:,n+len(self.columns):]=b[:,n:]
        return out


def verified_matrix(prefix,report,_depth=0):
    if _depth>16:raise ValueError('too many nested column overlays')
    if report.get('matrix_layout')=='column_block_insertion':
        path=prefix+'.matrix.json'
        if digest(path)!=report['composite_spec_sha256']:raise ValueError('matrix specification changed')
        s=json.loads(Path(path).read_text());parent_path=s['base_prefix']+'.json'
        if digest(parent_path)!=s['base_manifest_sha256']:raise ValueError('base block manifest changed')
        parent=json.loads(Path(parent_path).read_text())
        if parent.get('state')!='complete' or parent.get('qualifying_labels_read') is not False:
            raise ValueError('unverified block parent')
        base=verified_matrix(s['base_prefix'],parent,_depth+1);columns=[]
        for entry in s['columns']:
            if digest(entry['path'])!=entry['sha256']:raise ValueError('block column changed')
            if entry['format']=='npy':v=np.load(entry['path'],mmap_mode='r',allow_pickle=False)
            elif entry['format']=='f32':v=np.fromfile(entry['path'],dtype='<f4')
            else:raise ValueError('unknown block column format')
            if v.shape!=(len(base),) or not np.isfinite(v).all():raise ValueError('invalid block column')
            columns.append(v)
        if not columns:raise ValueError('empty inserted column block')
        result=InsertedColumns(base,columns,s['insert_at'])
        if list(result.shape)!=s['shape']:raise ValueError('block shape changed')
        return result
    if report.get('matrix_layout')=='column_selection':
        path=prefix+'.matrix.json'
        if digest(path)!=report['composite_spec_sha256']:raise ValueError('matrix specification changed')
        s=json.loads(Path(path).read_text());parent_path=s['base_prefix']+'.json'
        if digest(parent_path)!=s['base_manifest_sha256']:raise ValueError('base selection manifest changed')
        parent=json.loads(Path(parent_path).read_text())
        if parent.get('state')!='complete' or parent.get('qualifying_labels_read') is not False:
            raise ValueError('unverified selection parent')
        result=SelectedColumns(verified_matrix(s['base_prefix'],parent,_depth+1),s['columns'])
        if list(result.shape)!=s['shape']:raise ValueError('selection shape changed')
        return result
    if report.get('matrix_layout')!='column_insertion':
        if digest(prefix+'.x.npy')!=report['x_sha256']:raise ValueError('qualifying matrix changed')
        return np.load(prefix+'.x.npy',mmap_mode='r')
    path=prefix+'.matrix.json'
    if digest(path)!=report['composite_spec_sha256']:raise ValueError('matrix specification changed')
    s=json.loads(Path(path).read_text())
    if digest(s['column_path'])!=s['column_sha256']:
        raise ValueError('overlay input changed')
    if 'base_prefix' in s:
        if digest(s['base_prefix']+'.json')!=s['base_manifest_sha256']:raise ValueError('base overlay manifest changed')
        parent=json.loads(Path(s['base_prefix']+'.json').read_text())
        if parent.get('state')!='complete' or parent.get('qualifying_labels_read') is not False:
            raise ValueError('unverified parent overlay')
        base=verified_matrix(s['base_prefix'],parent,_depth+1)
    else:
        if digest(s['base_path'])!=s['base_sha256']:raise ValueError('overlay input changed')
        base=np.load(s['base_path'],mmap_mode='r')
    column=np.fromfile(s['column_path'],dtype='<f4')
    if len(column)!=len(base) or not np.isfinite(column).all() or not 0<=s['insert_at']<=base.shape[1]:
        raise ValueError('invalid column overlay')
    result=InsertedColumn(base,column,s['insert_at'])
    if list(result.shape)!=s['shape']:raise ValueError('overlay shape changed')
    return result


def predict_matrix_ridge(raw,context,columns,weights):
    result=np.empty(len(raw))
    for lo in range(0,len(raw),8192):
        hi=min(lo+8192,len(raw))
        result[lo:hi]=3.6+design(raw[lo:hi][:,:columns],context[lo:hi])@weights
    return result
