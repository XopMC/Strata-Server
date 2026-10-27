"""Private N1 owner. No model import; issuer-thread close errors reach supervisor."""
import copy,queue,sys,threading,time,traceback

def make_owner_class(Base):
    canonical=sys.modules[Base.__module__]
    Job=canonical.InferenceJob;QueueFull=canonical.QueueFull
    RequestError=canonical.RequestError;RequestCancelled=canonical.RequestCancelled
    class Owner:
        def __init__(self,factory,*,queue_limit=8,clock=time.monotonic,request_seconds=1800):
            if type(queue_limit)is not int or not 1<=queue_limit<=128:raise ValueError('Bounded FIFO required')
            if not 0<request_seconds<=7200:raise ValueError('Finite request deadline required')
            self.factory=factory;self.clock=clock;self.request_seconds=request_seconds
            self.queue=queue.Queue(maxsize=queue_limit);self.ready=threading.Event();self.closed=threading.Event();self.finished=threading.Event()
            self.lifecycle_lock=threading.RLock();self.admission_lock=self.lifecycle_lock
            self.startup_error=self.close_error=None;self.engine=None;self.phase='loading';self.active=False;self.active_job=None
            self.admission_draining=False;self.last_activity=clock()
            self.thread=threading.Thread(target=self._run,name='pair252-inference-owner',daemon=True);self.thread.start()
        def _send(self,job,event):
            while not job.cancel.is_set() and self.clock()<job.deadline and not self.closed.is_set():
                try:job.output.put(event,timeout=.1);return True
                except queue.Full:continue
            job.cancel.set();return False
        def _run(self):
            try:
                try:self.engine=self.factory();self.phase='ready'
                except BaseException as error:self.startup_error=error;self.phase='failed_startup';return
                finally:self.ready.set()
                while not self.closed.is_set():
                    try:job=self.queue.get(timeout=.1)
                    except queue.Empty:continue
                    try:
                        with self.lifecycle_lock:self.active=True;self.active_job=job
                        if job.cancel.is_set() or self.clock()>=job.deadline or self.closed.is_set():job.cancel.set();continue
                        def emit(event):return self._send(job,event)
                        self.engine.run(job.body,job.cancel,emit)
                    except RequestCancelled:job.cancel.set()
                    except Exception as error:
                        if not isinstance(error,RequestError):traceback.print_exc()
                        self._send(job,dict(event='error',status=400 if isinstance(error,RequestError)else 500,
                            message=str(error),type='invalid_request_error'if isinstance(error,RequestError)else'inference_error'))
                        if any(x in str(error)for x in('misaligned address','illegal memory access','device-side assert')):
                            self.phase='failed_cuda_context';self.closed.set()
                    finally:
                        self._send(job,dict(event='done'))
                        with self.lifecycle_lock:self.active=False;self.active_job=None;self.last_activity=self.clock();self.queue.task_done()
            finally:
                self.closed.set();self.ready.set()
                while True:
                    try:job=self.queue.get_nowait()
                    except queue.Empty:break
                    job.cancel.set();self.queue.task_done()
                if self.engine is not None:
                    try:self.engine.close()
                    except BaseException as error:self.close_error=error;self.phase='failed_close';traceback.print_exc()
                if self.phase=='ready':self.phase='stopped'
                self.finished.set()
        def submit(self,body):
            job=Job(copy.deepcopy(body));job.deadline=self.clock()+self.request_seconds
            with self.lifecycle_lock:
                if self.admission_draining or not self.ready.is_set()or self.startup_error or self.closed.is_set():raise RuntimeError('Inference owner unavailable: '+self.phase)
                try:self.queue.put_nowait(job)
                except queue.Full:raise QueueFull('Bounded inference queue full')from None
                self.last_activity=self.clock()
            return job
        def touch_activity(self):
            with self.lifecycle_lock:self.last_activity=self.clock()
        def expire_if_idle(self,seconds):
            with self.lifecycle_lock:
                with self.queue.all_tasks_done:pending=self.queue.unfinished_tasks
                if self.admission_draining or pending or self.active or self.clock()-self.last_activity<seconds:return False
                self.admission_draining=True;return True
        def begin_draining(self,cancel_active=False):
            with self.lifecycle_lock:
                self.admission_draining=True
                if cancel_active and self.active_job is not None:self.active_job.cancel.set()
        def status(self):
            with self.lifecycle_lock:
                return dict(phase=self.phase,ready=self.ready.is_set()and not self.closed.is_set()and not self.startup_error and not self.admission_draining,
                    queued_requests=self.queue.qsize(),active_request=self.active,admission_draining=self.admission_draining,
                    inference_concurrency=1,configuration=self.engine.configuration()if self.engine is not None else None)
        def close(self,timeout=600):
            self.begin_draining(cancel_active=True);self.closed.set();self.thread.join(timeout)
            if self.thread.is_alive()or not self.finished.is_set():raise TimeoutError('Issuer did not finish close; replacement forbidden')
            if self.close_error is not None:raise RuntimeError('Issuer runtime.close failed; replacement forbidden')from self.close_error
    return Owner
