//! Owned, C-order f32 matrices with an optional file-backed allocation.
//! The ndarray view never outlives its backing, which remains at a stable address.
use ndarray::{ArrayViewMut2,ArrayViewMut1,Ix2,NdIndex};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::{fs::{File, OpenOptions}, os::fd::AsRawFd, path::PathBuf, ops::{Deref,Index,IndexMut}};
use std::sync::atomic::{AtomicUsize,Ordering};

static NEXT: AtomicUsize=AtomicUsize::new(0);
enum Backing {
    Heap { _values: Box<[f32]> },
    Map { ptr: *mut libc::c_void, bytes: usize, _file: File, path: PathBuf },
}
pub struct StorageMatrix {
    view: ArrayViewMut2<'static,f32>,
    backing: Backing,
    pending: AtomicUsize,
}
// The pointer is uniquely owned; mutation is available only through &mut self.
unsafe impl Send for StorageMatrix {}
unsafe impl Sync for StorageMatrix {}
impl StorageMatrix {
    pub fn zeros(shape:(usize,usize))->Self {
        let count=shape.0.checked_mul(shape.1).expect("matrix size overflow");
        let bytes=count.checked_mul(4).expect("matrix byte size overflow");
        let directory=std::env::var_os("TX_MMAP_DIR");
        let (ptr,backing)=if let Some(dir)=directory.filter(|_|bytes>0) {
            std::fs::create_dir_all(&dir).unwrap();
            let path=PathBuf::from(dir).join(format!("tx-{}-{}.f32",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
            let file=OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
            file.set_len(bytes as u64).unwrap();
            let ptr=unsafe {libc::mmap(std::ptr::null_mut(),bytes,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_SHARED,file.as_raw_fd(),0)};
            assert_ne!(ptr,libc::MAP_FAILED,"mmap failed: {}",std::io::Error::last_os_error());
            (ptr as *mut f32,Backing::Map{ptr,bytes,_file:file,path})
        } else {
            let mut values=vec![0.0;count].into_boxed_slice();
            (values.as_mut_ptr(),Backing::Heap{_values:values})
        };
        let view=unsafe {ArrayViewMut2::from_shape_ptr(shape,ptr)};
        Self{view,backing,pending:AtomicUsize::new(0)}
    }
    pub fn normal<R:Rng>(rows:usize,cols:usize,rng:&mut R,sigma:f32)->Self {
        let mut out=Self::zeros((rows,cols));
        let dist=Normal::<f32>::new(0.0,sigma).unwrap();
        for row in 0..rows{
            for value in out.view.row_mut(row).iter_mut(){*value=dist.sample(rng);}
            out.release_rows(row,row+1);
        }
        out.release_rows(0,rows);
        out
    }
    pub fn row_mut(&mut self,row:usize)->ArrayViewMut1<'_,f32>{self.view.row_mut(row)}
    pub fn fill(&mut self,value:f32){self.view.fill(value);}
    pub fn as_slice_mut(&mut self)->Option<&mut[f32]>{self.view.as_slice_mut()}
    /// Batch completed users, then flush and advise eviction of this allocation.
    /// Batching avoids a synchronous disk operation for every user.
    /// Values are retained in the backing file; subsequent accesses reload them.
    pub fn release_rows(&self,start:usize,end:usize) {
        if let Backing::Map{ptr,bytes,..}=&self.backing {
            if *bytes<64*1024*1024 || start==end{return;}
            assert!(start<=end && end<=self.view.nrows());
            let force=start==0 && end==self.view.nrows();
            if !force && (self.pending.fetch_add(1,Ordering::Relaxed)+1)%2048!=0{return;}
            assert_eq!(unsafe{libc::msync(*ptr,*bytes,libc::MS_SYNC)},0,"msync: {}",std::io::Error::last_os_error());
            assert_eq!(unsafe{libc::madvise(*ptr,*bytes,libc::MADV_DONTNEED)},0,"madvise: {}",std::io::Error::last_os_error());
        }
    }
}
impl Deref for StorageMatrix {
    type Target=ArrayViewMut2<'static,f32>;
    fn deref(&self)->&Self::Target{&self.view}
}
impl<I:NdIndex<Ix2>> Index<I> for StorageMatrix {
    type Output=f32;
    fn index(&self,index:I)->&f32{&self.view[index]}
}
impl<I:NdIndex<Ix2>> IndexMut<I> for StorageMatrix {
    fn index_mut(&mut self,index:I)->&mut f32{&mut self.view[index]}
}
impl Drop for StorageMatrix {
    fn drop(&mut self){
        if let Backing::Map{ptr,bytes,path,..}=&self.backing {
            assert_eq!(unsafe{libc::munmap(*ptr,*bytes)},0);
            std::fs::remove_file(path).expect("remove owned mapping");
        }
    }
}
