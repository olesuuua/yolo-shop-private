"""Measurement-only runner. Original application/model/scheduler code is unchanged."""
import sys,os,json,time,threading,functools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['LIGHTSTORE_DEVICE']='cpu'
os.environ['LIGHTSTORE_MODEL']='ppyoloe_objects365'
os.environ['YOLO_AUTOINSTALL']='false'
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
lock=threading.Lock();events=0
prior=OUT/"server-events.jsonl"
count=sum(json.loads(line).get("event")=="jev_http_begin" for line in prior.read_text().splitlines()) if prior.exists() else 0
log=(OUT/'server-events.jsonl').open('a',buffering=1)
def emit(kind,**data):
 global events
 with lock:
  if events>=20000:return
  events+=1;log.write(json.dumps({'event':kind,'server_monotonic':time.monotonic(),**data},ensure_ascii=False,default=str)+'\n')
import jev_catalog
original_classify=jev_catalog.classify
def measured_classify(*args,**kwargs):
 global count
 with lock:
  if count>=40: raise RuntimeError('Comparison Jev authorization cap reached; no external call made.')
  count+=1;call_id=count
 emit('jev_http_begin',call_id=call_id,lines=args[0])
 started=time.monotonic()
 try:
  result=original_classify(*args,**kwargs)
  emit('jev_http_end',call_id=call_id,duration_ms=(time.monotonic()-started)*1000,result=result)
  return result
 except Exception as e:
  emit('jev_http_error',call_id=call_id,error_type=type(e).__name__)
  raise
jev_catalog.classify=measured_classify
import identification,vision

def wrap(cls,name,detail):
 original=getattr(cls,name)
 @functools.wraps(original)
 def measured(self,*args,**kwargs):
  data=detail(self,args,kwargs);emit(name+'_begin',**data)
  try:
   result=original(self,*args,**kwargs)
   extra={}
   if name=='process_detect': extra={'frame_id':result['frame_id'],'session_version':result['session_version'],'tracks':result['tracks'],'crop_requests':result['crop_requests'],'identification':result['identification'],'detect_ms':result['detect_ms']}
   elif name=='process_crop_response':extra={'ack':result}
   elif name=='_identify':extra={'result':args[1].result,'complete':args[1].complete,'applied_to_current':self.tracks.get(args[0]) is args[1]}
   elif name=='submit':extra={'accepted':result}
   elif name=='reset':extra={'state':result}
   emit(name+'_end',**data,**extra);return result
  except Exception as e:
   emit(name+'_error',**data,error_type=type(e).__name__);raise
 setattr(cls,name,measured)
wrap(vision.FrameProcessor,'process_detect',lambda s,a,k:{'session_at_entry':s.session_version})
wrap(vision.FrameProcessor,'process_crop_response',lambda s,a,k:{'request_id':a[0].get('request_id'),'track_id':a[0].get('track_id'),'session_version':a[0].get('session_version')})
wrap(vision.FrameProcessor,'reset',lambda s,a,k:{'session_before':s.session_version})
wrap(identification.IdentificationService,'submit',lambda s,a,k:{'track_id':a[0],'request_id':(k.get('diagnostic') or {}).get('request_id')})
wrap(identification.IdentificationService,'_identify',lambda s,a,k:{'track_id':a[0],'sources':sorted({r for ids in a[1].sources.values() for r in ids})})
wrap(identification.IdentificationService,'_ensure_ocr',lambda s,a,k:{'already_loaded':s.ocr is not None})
import app,uvicorn
emit('server_start',device='cpu',profile='ppyoloe_objects365',jev_call_cap=40)
uvicorn.run(app.app,host='127.0.0.1',port=8002,log_level='warning')
