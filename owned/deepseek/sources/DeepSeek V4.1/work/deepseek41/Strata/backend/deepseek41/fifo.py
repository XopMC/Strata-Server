"""One inference owner, FIFO clients and bounded streaming backpressure."""
from dataclasses import dataclass,field
import copy
import queue
import threading
import traceback


class RequestError(ValueError):pass
class QueueFull(RuntimeError):pass
class RequestCancelled(RuntimeError):pass


@dataclass
class InferenceJob:
    body:dict
    output:queue.Queue=field(default_factory=lambda:queue.Queue(maxsize=256))
    cancel:threading.Event=field(default_factory=threading.Event)


class InferenceOwner:
    def __init__(self,engine_factory,*,queue_limit=8):
        if type(queue_limit) is not int or not 1<=queue_limit<=128:raise ValueError('Invalid bounded inference queue')
        self.factory=engine_factory;self.queue=queue.Queue(maxsize=queue_limit)
        self.ready=threading.Event();self.closed=threading.Event();self.startup_error=None;self.engine=None
        self.phase='loading';self.active=False;self.lifecycle_lock=threading.Lock()
        self.thread=threading.Thread(target=self._run,name='deepseek41-inference-owner',daemon=True)
        self.thread.start()

    def _send(self,job,event):
        terminal=event.get('event') in ('error','done')
        while not job.cancel.is_set() and (terminal or not self.closed.is_set()):
            try:job.output.put(event,timeout=.25);return True
            except queue.Full:continue
        return False

    def _run(self):
        try:
            self.engine=self.factory();self.phase='ready'
        except BaseException as error:
            self.startup_error=error;self.phase='failed_startup';self.ready.set();return
        self.ready.set()
        try:
            while not self.closed.is_set():
                try:job=self.queue.get(timeout=.25)
                except queue.Empty:continue
                try:
                    if job.cancel.is_set():continue
                    self.active=True
                    self.active_job=job
                    if hasattr(self.engine,'run'):
                        def emit(event):
                            if not self._send(job,event):job.cancel.set();return False
                            return True
                        self.engine.run(job.body,job.cancel,emit)
                    else:
                        for event in self.engine.infer(job.body,job.cancel):
                            if not self._send(job,event):break
                except RequestCancelled:
                    job.cancel.set()
                except Exception as error:
                    if not isinstance(error,RequestError):traceback.print_exc()
                    self._send(job,{'event':'error','status':400 if isinstance(error,RequestError) else 500,
                        'message':str(error),'type':'invalid_request_error' if isinstance(error,RequestError) else 'inference_error'})
                    if any(term in str(error) for term in ('misaligned address','illegal memory access','device-side assert')):
                        with self.lifecycle_lock:
                            self.phase='failed_cuda_context';self.closed.set()
                finally:
                    self.active=False;self.queue.task_done()
                    self.active_job=None
                    self._send(job,{'event':'done'})
        finally:
            while True:
                try:pending=self.queue.get_nowait()
                except queue.Empty:break
                self._send(pending,{'event':'error','status':503,'message':'Inference owner stopped','type':'server_unavailable'})
                self._send(pending,{'event':'done'});self.queue.task_done()
            if self.engine is not None:self.engine.close()
            if self.phase=='ready':self.phase='stopped'

    def submit(self,body):
        job=InferenceJob(copy.deepcopy(body))
        with self.lifecycle_lock:
            if not self.ready.is_set() or self.startup_error or self.closed.is_set():raise RuntimeError('Inference owner is unavailable: '+self.phase)
            try:self.queue.put_nowait(job)
            except queue.Full:raise QueueFull('Inference queue is full; retry after current requests finish') from None
        return job

    def status(self):
        return {'phase':self.phase,'ready':self.ready.is_set() and self.startup_error is None and not self.closed.is_set(),
            'queued_requests':self.queue.qsize(),'active_request':self.active,
            'configuration':self.engine.configuration() if self.engine is not None else None}

    def close(self,timeout=30):
        with self.lifecycle_lock:
            self.closed.set()
            active=getattr(self,'active_job',None)
            if active is not None:active.cancel.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():raise TimeoutError('Inference owner has not stopped yet')
