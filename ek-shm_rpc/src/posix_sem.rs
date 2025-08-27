use std::ffi::CString;
use ek_base::error::{ EKResult, EKError };
use libc::{ O_CREAT, SEM_FAILED };

pub struct PosixSem {
    sem_name: String,
    sem: *mut libc::sem_t,
    owned: bool,
}

unsafe impl Send for PosixSem {}
unsafe impl Sync for PosixSem {}

impl PosixSem {
    pub fn open(sem_name: String, owned: bool) -> EKResult<Self> {
        let sem = unsafe {
            let name = CString::new(sem_name.as_str())
                .map_err(|e| EKError::InvalidInput(format!("CString conversion failed: {}", e)))?;
            let oflag = match owned {
                true => O_CREAT,
                false => 0,
            };
            libc::sem_open(name.as_ptr(), oflag, 0o666, 0)
        };
        if sem == SEM_FAILED {
            log::error!("failed to create semaphores: {}", sem_name);
            return Err(EKError::IoError(std::io::Error::last_os_error().into()));
        }

        Ok(Self {
            sem_name: sem_name,
            sem: sem,
            owned: owned,
        })
    }

    pub fn reset(&self) {
        unsafe {
            while libc::sem_trywait(self.sem) == 0 {}
        }
    }

    pub fn wait(&self) {
        unsafe {
            if libc::sem_wait(self.sem) != 0 {
                log::error!("sem_wait failed: {}", std::io::Error::last_os_error());
            }
        };
    }

    // -1 error, 0 success, 1 timeout
    #[cfg(target_os = "linux")]
    pub fn timedwait(&self, time: i64) -> i32 {
        let ret = unsafe {
            let ts = libc::timespec {
                tv_sec: time,
                tv_nsec: 0,
            };
            libc::sem_timedwait(self.sem, &ts)
        };
        if ret == -1 {
            let errno = std::io::Error::last_os_error();
            if errno.kind() == std::io::ErrorKind::TimedOut {
                return 1;
            } else {
                log::error!("sem wait error {}", errno);
                return -1;
            }
        }

        return 0;
    }

    #[cfg(not(target_os = "linux"))]
    pub fn timedwait(&self, timeout_ms: i64) -> i32 {
        use std::time::{Duration, Instant};

        let start = Instant::now();
        let timeout = Duration::from_millis(timeout_ms as u64);

        loop {
            let ret = unsafe { libc::sem_trywait(self.sem) };

            if ret == 0 {
                return 0;
            }

            let errno = std::io::Error::last_os_error();
            if errno.raw_os_error() != Some(libc::EAGAIN) {
                log::error!("sem_trywait error: {}", errno);
                return -1;
            }

            if start.elapsed() >= timeout {
                return 1;
            }

            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn post(&self) {
        unsafe {
            if libc::sem_post(self.sem) != 0 {
                log::error!("sem_post failed: {}", std::io::Error::last_os_error());
            }
        };
    }
}

impl Drop for PosixSem {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                libc::sem_close(self.sem);
                let _ = CString::new(self.sem_name.as_str()).map(|n| libc::sem_unlink(n.as_ptr()));
            }
        }
    }
}

impl Clone for PosixSem {
    fn clone(&self) -> Self {
        Self {
            sem: self.sem.clone(),
            sem_name: self.sem_name.clone(),
            owned: self.owned
        }
    }
}
