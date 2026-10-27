"""Native incremental UTF8/DSML streaming and compatible request conversion."""
import codecs
import copy
import json
import math
from pathlib import Path
import sys

REFERENCE=Path(__file__).resolve().parents[3]/'reference/encoding'
sys.path.insert(0,str(REFERENCE))
from encoding import (encode_messages,parse_message_from_completion_text,eos_token,
                      thinking_end_token,dsml_token,tool_calls_block_name)

MODEL_ID='deepseek-v4.1-flash-abliterated'


class ByteLevelDecoder:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer;self.added=set(tokenizer.get_added_vocab().values())
        values=list(range(33,127))+list(range(161,173))+list(range(174,256));characters=values.copy();extra=0
        for byte in range(256):
            if byte not in values:values.append(byte);characters.append(256+extra);extra+=1
        self.byte_map={chr(char):byte for char,byte in zip(characters,values)}
        self.decoder=codecs.getincrementaldecoder('utf-8')('replace')

    def token(self,token_id):
        text=self.tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(text,str):raise ValueError('Generated token is outside the standalone tokenizer')
        if token_id in self.added:
            pending=self.decoder.decode(b'',final=True);self.decoder.reset()
            return pending+text
        return self.decoder.decode(bytes(self.byte_map[char] for char in text),final=False)

    def finish(self):return self.decoder.decode(b'',final=True)


class NativeDeltaParser:
    def __init__(self,thinking_mode='chat'):
        if thinking_mode not in ('chat','thinking'):raise ValueError('Invalid native thinking mode')
        self.mode=thinking_mode;self.state='reasoning_content' if thinking_mode=='thinking' else 'content'
        self.buffer='';self.raw=[];self.completed=False
        self.tool_start=f'\n\n<{dsml_token}{tool_calls_block_name}'

    def feed(self,text):
        if self.completed and text:raise ValueError('Model emitted text after its native EOS')
        self.raw.append(text);self.buffer+=text;events=[]
        while self.buffer and self.state in ('content','reasoning_content'):
            boundaries=[thinking_end_token] if self.state=='reasoning_content' else [eos_token,self.tool_start]
            locations=[(self.buffer.find(tag),tag) for tag in boundaries if tag in self.buffer]
            if locations:
                index,tag=min(locations)
                if index:events.append({self.state:self.buffer[:index]})
                self.buffer=self.buffer[index+len(tag):]
                if tag==thinking_end_token:self.state='content'
                elif tag==self.tool_start:self.state='tool_calls';self.buffer=tag+self.buffer
                else:self.state='done';self.completed=True
                continue
            withheld=max((length for tag in boundaries for length in range(1,min(len(tag),len(self.buffer))+1)
                          if tag.startswith(self.buffer[-length:])),default=0)
            safe=len(self.buffer)-withheld
            if safe:events.append({self.state:self.buffer[:safe]});self.buffer=self.buffer[safe:]
            break
        if self.state=='tool_calls' and eos_token in self.buffer:
            self.state='done';self.completed=True;self.buffer=''
        return events

    def finish(self,eos_reached):
        text=''.join(self.raw)
        if eos_reached:
            if not self.completed:raise ValueError('EOS token did not reach the native output decoder')
            message=parse_message_from_completion_text(text,self.mode)
            message['tool_calls']=[{**row,'id':f'call_{index}'} for index,row in enumerate(message['tool_calls'])]
            return message,'tool_calls' if message['tool_calls'] else 'stop'
        # Never invent EOS or a complete tool invocation when output is cut.
        content='';reasoning=''
        if self.mode=='thinking':
            if thinking_end_token in text:reasoning,text=text.split(thinking_end_token,1)
            else:reasoning,text=text,''
        content=text.split(self.tool_start,1)[0]
        return {'role':'assistant','content':content,'reasoning_content':reasoning,'tool_calls':[]},'length'


def native_request(body,*,context_capacity):
    if not isinstance(body,dict) or body.get('model',MODEL_ID)!=MODEL_ID:raise ValueError('Unknown model ID')
    if body.get('stop') not in (None,[], '') or body.get('tool_choice','auto')!='auto':
        raise ValueError('This native backend currently supports native EOS and automatic tool choice')
    options=body.get('stream_options',{})
    if not isinstance(options,dict) or type(options.get('include_usage',False)) is not bool:
        raise ValueError('Invalid stream_options.include_usage')
    messages=copy.deepcopy(body.get('messages'))
    if not isinstance(messages,list) or not 1<=len(messages)<=4096:raise ValueError('Expected1–4096conversation messages')
    for row in messages:
        if not isinstance(row,dict) or row.get('role') not in ('system','user','assistant','tool'):
            raise ValueError('Invalid native message role')
        if not isinstance(row.get('content'),(str,list,type(None))):raise ValueError('Expected text or image content parts')
    maximum=body.get('max_completion_tokens',body.get('max_tokens',512))
    if type(maximum) is not int or not 1<=maximum<=context_capacity:raise ValueError('Invalid generation token budget')
    mode=body.get('thinking_mode','thinking' if body.get('enable_thinking',False) else 'chat')
    if mode not in ('chat','thinking'):raise ValueError('Invalid thinking mode')
    tools=body.get('tools')
    if tools is not None:
        if not isinstance(tools,list) or len(tools)>128 or len(json.dumps(tools))>1024**2:
            raise ValueError('Invalid bounded function tool schemas')
        for tool in tools:
            if (not isinstance(tool,dict) or tool.get('type')!='function' or not isinstance(tool.get('function'),dict) or
                not isinstance(tool['function'].get('name'),str) or not tool['function']['name'] or
                not isinstance(tool['function'].get('parameters',{}),dict)):
                raise ValueError('Invalid native function tool schema')
        if messages[0]['role']!='system':messages.insert(0,{'role':'system','content':''})
        messages[0]['tools']=copy.deepcopy(tools)
    effort=body.get('reasoning_effort','high')
    if effort=='medium':effort='high'
    if not ((type(effort) is int and 1<=effort<=100) or effort in ('low','high','xhigh','max')):
        raise ValueError('Invalid native reasoning effort')
    temperature=body.get('temperature',0);top_p=body.get('top_p',1);top_k=body.get('top_k',0);penalty=body.get('repetition_penalty',1)
    if (type(temperature) not in (int,float) or not math.isfinite(temperature) or not 0<=temperature<=2 or
        type(top_p) not in (int,float) or not math.isfinite(top_p) or not 0<top_p<=1 or
        type(top_k) is not int or not 0<=top_k<=129280 or
        type(penalty) not in (int,float) or not math.isfinite(penalty) or not 1<=penalty<=2):
        raise ValueError('Invalid sampling parameters')
    # Multimodal URLs are resolved by the engine within the request lifetime.
    return {'messages':messages,'max_new_tokens':maximum,'thinking_mode':mode,'reasoning_effort':effort,
        'temperature':temperature,'top_p':top_p,'top_k':top_k,'repetition_penalty':penalty}


def anthropic_request(body):
    if not isinstance(body,dict):raise ValueError('Expected a messages request')
    messages=[]
    system=body.get('system')
    if system:
        messages.append({'role':'system','content':system if isinstance(system,str) else '\n'.join(row['text'] for row in system)})
    for row in body.get('messages',[]):
        parts=row.get('content');content=[];calls=[];reasoning=[]
        if isinstance(parts,str):messages.append({'role':row['role'],'content':parts});continue
        if not isinstance(parts,list):raise ValueError('Invalid Anthropic content blocks')
        for part in parts:
            kind=part.get('type')
            if kind=='text':content.append({'type':'text','text':part['text']})
            elif kind=='image':
                source=part['source']
                if source['type']=='base64':url=f"data:{source['media_type']};base64,{source['data']}"
                elif source['type']=='url':url=source['url']
                else:raise ValueError('Unsupported image source')
                content.append({'type':'image_url','image_url':{'url':url}})
            elif kind=='tool_use':calls.append({'id':part['id'],'type':'function',
                'function':{'name':part['name'],'arguments':json.dumps(part['input'],ensure_ascii=False)}})
            elif kind=='tool_result':
                messages.append({'role':'tool','tool_call_id':part['tool_use_id'],'content':part['content']})
            elif kind=='thinking':reasoning.append(part['thinking'])
            else:raise ValueError('Unsupported Anthropic content block')
        if content or calls or reasoning:
            message={'role':row['role'],'content':content,'tool_calls':calls}
            if reasoning:message['reasoning_content']=''.join(reasoning)
            messages.append(message)
    tools=[{'type':'function','function':{'name':row['name'],'description':row.get('description',''),
            'parameters':row['input_schema']}} for row in body.get('tools',[])]
    return {**body,'messages':messages,'tools':tools,'model':body.get('model',MODEL_ID)}
