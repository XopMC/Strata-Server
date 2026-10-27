"""Bounded OpenAI/Anthropic HTTP frontend for one FIFO inference owner.

HTTP threads never create CUDA contexts. Long prefill has SSE keepalives;
disconnected clients cancel their own job rather than the inference process.
"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import queue
import select
import socket
import threading
import time
import uuid

from .api_codec import MODEL_ID,native_request,anthropic_request
from .fifo import QueueFull


def diagnostic_fields(result):
    return {'deepseek_timings':result.get('timings',{}),'deepseek_mtp':result.get('mtp'),
        'deepseek_runtime_configuration':result.get('runtime_configuration',{}),'deepseek_checkpoint':result.get('checkpoint',{})}


def openai_result(result,identifier,created):
    return {'id':identifier,'object':'chat.completion','created':created,'model':MODEL_ID,
        'choices':[{'index':0,'message':result['message'],'finish_reason':result['finish_reason']}],
        'usage':result['usage'],**diagnostic_fields(result)}


def anthropic_blocks(message):
    blocks=[]
    if message.get('reasoning_content'):blocks.append({'type':'thinking','thinking':message['reasoning_content']})
    if message.get('content'):blocks.append({'type':'text','text':message['content']})
    for call in message.get('tool_calls',[]):
        arguments=json.loads(call['function']['arguments'])
        if not isinstance(arguments,dict):raise ValueError('Native tool arguments must be a JSON object')
        blocks.append({'type':'tool_use','id':call['id'],'name':call['function']['name'],'input':arguments})
    return blocks


def stop_reason(reason):return {'tool_calls':'tool_use','length':'max_tokens','stop':'end_turn'}[reason]


def anthropic_result(result,identifier):
    return {'id':identifier,'type':'message','role':'assistant','model':MODEL_ID,
        'content':anthropic_blocks(result['message']),'stop_reason':stop_reason(result['finish_reason']),
        'stop_sequence':None,'usage':{'input_tokens':result['usage']['prompt_tokens'],
        'output_tokens':result['usage']['completion_tokens']},**diagnostic_fields(result)}


class ApiServer(ThreadingHTTPServer):
    daemon_threads=True
    allow_reuse_address=True

    def __init__(self,address,owner,*,context_capacity,max_connections=32,body_limit=128*1024**2,heartbeat_seconds=10,request_timeout_seconds=1800):
        self.owner=owner;self.context_capacity=context_capacity;self.body_limit=body_limit
        if not 0<request_timeout_seconds<=7200:raise ValueError('Bounded request timeout required')
        self.request_timeout_seconds=request_timeout_seconds
        self.heartbeat_seconds=heartbeat_seconds;self.connections=threading.BoundedSemaphore(max_connections)
        super().__init__(address,ApiHandler)

    def process_request(self,request,address):
        if not self.connections.acquire(blocking=False):
            try:request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n')
            finally:self.shutdown_request(request)
            return
        try:super().process_request(request,address)
        except BaseException:self.connections.release();raise

    def process_request_thread(self,request,address):
        try:super().process_request_thread(request,address)
        finally:self.connections.release()


class ApiHandler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    server_version='DeepSeek41/1'

    def setup(self):
        super().setup();self.connection.settimeout(30)

    def log_message(self,format,*arguments):
        # No request bodies, images, tool schemas or client credentials in logs.
        print(json.dumps({'http_client':self.client_address[0],'http':format%arguments}),flush=True)

    def json_reply(self,status,value):
        payload=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(payload)));self.send_header('Connection','close');self.end_headers()
        self.close_connection=True;self.wfile.write(payload);self.wfile.flush()

    def error_reply(self,status,message,kind='invalid_request_error'):
        if self.path=='/v1/messages':value={'type':'error','error':{'type':kind,'message':message}}
        else:value={'error':{'type':kind,'message':message,'code':status}}
        self.json_reply(status,value)

    def do_GET(self):
        if self.path in ('/health','/healthz'):
            state=self.server.owner.status();self.json_reply(200 if state['ready'] else 503,state)
        elif self.path=='/v1/models':
            self.json_reply(200,{'object':'list','data':[{'id':MODEL_ID,'object':'model','created':0,'owned_by':'local',
                'context_length':self.server.context_capacity}]})
        else:self.error_reply(404,'Unknown endpoint')

    def read_body(self):
        if self.headers.get('Transfer-Encoding'):raise ValueError('Use Content-Length for request bodies')
        value=self.headers.get('Content-Length')
        if value is None or not value.isdecimal():raise ValueError('Expected a valid Content-Length')
        size=int(value)
        if not 0<size<=self.server.body_limit:raise ValueError('Request body exceeds the configured byte budget')
        payload=self.rfile.read(size)
        if len(payload)!=size:raise ValueError('Incomplete request body')
        body=json.loads(payload,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON number')))
        if not isinstance(body,dict):raise ValueError('Expected a request object')
        if type(body.get('stream',False)) is not bool:raise ValueError('stream must be boolean')
        return body

    def disconnected(self):
        if not select.select([self.connection],[],[],0)[0]:return False
        try:return self.connection.recv(1,socket.MSG_PEEK|socket.MSG_DONTWAIT)==b''
        except BlockingIOError:return False
        except OSError:return True

    def sse(self,value,*,event=None):
        prefix=('event: '+event+'\n') if event else ''
        data=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)
        self.wfile.write((prefix+'data: '+data+'\n\n').encode());self.wfile.flush()

    def do_POST(self):
        if self.path not in ('/v1/chat/completions','/v1/messages'):
            self.error_reply(404,'Unknown endpoint');return
        job=None;stream_started=False;deadline=time.monotonic()+self.server.request_timeout_seconds
        try:
            body=self.read_body();anthropic=self.path=='/v1/messages'
            if anthropic:body=anthropic_request(body)
            native_request(body,context_capacity=self.server.context_capacity)
            job=self.server.owner.submit(body)
            streaming=body.get('stream',False);identifier=('msg_' if anthropic else 'chatcmpl_')+uuid.uuid4().hex
            created=int(time.time());result=None;blocks={};next_block=0;active_block=None;message_started=False
            next_ping=time.monotonic()+self.server.heartbeat_seconds
            def start_message(prompt_tokens):
                nonlocal message_started
                if message_started:return
                self.sse({'type':'message_start','message':{'id':identifier,'type':'message','role':'assistant','model':MODEL_ID,
                    'content':[],'stop_reason':None,'stop_sequence':None,
                    'usage':{'input_tokens':prompt_tokens,'output_tokens':0}}},event='message_start')
                message_started=True
            if streaming:
                self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
                self.send_header('Cache-Control','no-cache');self.send_header('X-Accel-Buffering','no')
                self.send_header('Connection','close');self.end_headers();self.close_connection=True;stream_started=True
                if not anthropic:self.chat_chunk(identifier,created,{'role':'assistant'})
            while True:
                if time.monotonic()>=deadline:
                    job.cancel.set();raise TimeoutError('Request total deadline exceeded; cancellation requested')
                if job.cancel.is_set():return
                if self.disconnected():job.cancel.set();return
                try:event=job.output.get(timeout=min(.5,self.server.heartbeat_seconds))
                except queue.Empty:
                    if streaming and time.monotonic()>=next_ping:
                        self.wfile.write(b': processing\n\n');self.wfile.flush();next_ping=time.monotonic()+self.server.heartbeat_seconds
                    if not self.server.owner.thread.is_alive():raise RuntimeError('Inference owner stopped before returning this request')
                    continue
                if event['event']=='error':
                    if streaming:self.sse({'type':'error','error':{'type':event['type'],'message':event['message']}},event='error' if anthropic else None)
                    else:self.error_reply(event['status'],event['message'],event['type'])
                    return
                if event['event']=='usage' and streaming and anthropic:start_message(event['prompt_tokens'])
                elif event['event']=='result':result=event['result']
                elif event['event']=='delta' and streaming:
                    delta=event['delta']
                    if not anthropic:self.chat_chunk(identifier,created,delta)
                    else:
                        for field,text in delta.items():
                            kind='thinking' if field=='reasoning_content' else 'text'
                            if not message_started:raise RuntimeError('Runtime did not report the input token count before generation')
                            if kind not in blocks:
                                if active_block is not None:self.sse({'type':'content_block_stop','index':active_block},event='content_block_stop')
                                index=next_block;next_block+=1;blocks[kind]=index
                                self.sse({'type':'content_block_start','index':index,'content_block':{'type':kind,kind:''}},event='content_block_start')
                                active_block=index
                            self.sse({'type':'content_block_delta','index':blocks[kind],
                                'delta':{'type':kind+'_delta',kind:text}},event='content_block_delta')
                elif event['event']=='done':break
            if result is None:raise RuntimeError('Inference finished without a result')
            if not streaming:
                self.json_reply(200,anthropic_result(result,identifier) if anthropic else openai_result(result,identifier,created));return
            if anthropic:
                start_message(result['usage']['prompt_tokens'])
                if active_block is not None:self.sse({'type':'content_block_stop','index':active_block},event='content_block_stop')
                for block in anthropic_blocks(result['message']):
                    if block['type']!='tool_use':continue
                    index=next_block;next_block+=1
                    self.sse({'type':'content_block_start','index':index,'content_block':{**block,'input':{}}},event='content_block_start')
                    self.sse({'type':'content_block_delta','index':index,
                        'delta':{'type':'input_json_delta','partial_json':json.dumps(block['input'],ensure_ascii=False)}},event='content_block_delta')
                    self.sse({'type':'content_block_stop','index':index},event='content_block_stop')
                self.sse({'type':'message_delta','delta':{'stop_reason':stop_reason(result['finish_reason']),'stop_sequence':None},
                    'usage':{'output_tokens':result['usage']['completion_tokens']},**diagnostic_fields(result)},event='message_delta')
                self.sse({'type':'message_stop'},event='message_stop')
            else:
                calls=result['message'].get('tool_calls',[])
                if calls:self.chat_chunk(identifier,created,{'tool_calls':[{'index':i,**call} for i,call in enumerate(calls)]})
                self.sse({'id':identifier,'object':'chat.completion.chunk','created':created,'model':MODEL_ID,
                    'choices':[{'index':0,'delta':{},'finish_reason':result['finish_reason']}],**diagnostic_fields(result)})
                if body.get('stream_options',{}).get('include_usage'):
                    self.sse({'id':identifier,'object':'chat.completion.chunk','created':created,'model':MODEL_ID,'choices':[],'usage':result['usage']})
                self.sse('[DONE]')
        except (ValueError,TypeError,KeyError) as error:
            if stream_started:self.sse({'error':{'type':'invalid_request_error','message':str(error)}})
            else:self.error_reply(400,str(error))
        except TimeoutError as error:
            if job is not None:job.cancel.set()
            if stream_started:self.sse({'error':{'type':'request_timeout','message':str(error)}})
            else:self.error_reply(504,str(error),'request_timeout')
        except QueueFull as error:self.error_reply(429,str(error),'rate_limit_error')
        except (BrokenPipeError,ConnectionResetError,socket.timeout):
            if job is not None:job.cancel.set()
        except Exception as error:
            if stream_started:self.sse({'error':{'type':'server_unavailable','message':str(error)}})
            else:self.error_reply(503,str(error),'server_unavailable')
        finally:
            if job is not None:job.cancel.set()

    def chat_chunk(self,identifier,created,delta,*,reason=None):
        self.sse({'id':identifier,'object':'chat.completion.chunk','created':created,'model':MODEL_ID,
            'choices':[{'index':0,'delta':delta,'finish_reason':reason}]})
