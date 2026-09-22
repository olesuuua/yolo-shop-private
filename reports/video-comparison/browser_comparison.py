"""Isolated Chrome CDP driver; uses unmodified replay UI and binary pipeline."""
import asyncio,json,urllib.request,time,base64
from pathlib import Path
import websockets
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'evidence'
class CDP:
 def __init__(self,ws):self.ws=ws;self.seq=0;self.pending={};self.events=[]
 async def read(self):
  async for raw in self.ws:
   d=json.loads(raw)
   if 'id' in d:
    f=self.pending.pop(d['id'],None)
    if f:f.set_result(d)
   elif d.get('method') in ['Runtime.exceptionThrown','Log.entryAdded']: self.events.append(d)
 async def call(self,method,**params):
  self.seq+=1;f=asyncio.get_running_loop().create_future();self.pending[self.seq]=f
  await self.ws.send(json.dumps({'id':self.seq,'method':method,'params':params}));d=await f
  if 'error' in d:raise RuntimeError(d['error'])
  return d.get('result',{})
 async def evaluate(self,expression):
  r=await self.call('Runtime.evaluate',expression=expression,returnByValue=True,awaitPromise=True)
  if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
  return r.get('result',{}).get('value')
OBSERVE=r'''(() => {
window.comparisonEvents=[];
const originalSend=WebSocket.prototype.send;
WebSocket.prototype.send=function(data) {
 if (!this.comparisonObserved) {
  this.comparisonObserved=true;
  this.addEventListener('message',e=>{try {
    const d=JSON.parse(e.data); delete d.image;
    comparisonEvents.push({event:'received',browser_ms:performance.now(),data:d});
  } catch {}});
 }
 const detect=data instanceof Blob && data.type==='image/jpeg';
 comparisonEvents.push({event:'sent',browser_ms:performance.now(),kind:detect?'detection':typeof data==='string'?'command':'crop',
   video_timestamp:detect&&pendingUpload?pendingUpload.video_timestamp:null,bytes:data.size||0});
 return originalSend.call(this,data);
};
return true;
})()'''
async def main():
 tabs=json.load(urllib.request.urlopen('http://127.0.0.1:9223/json/list'));target=next(t for t in tabs if t['type']=='page')
 async with websockets.connect(target['webSocketDebuggerUrl'],max_size=100*1024*1024) as ws:
  c=CDP(ws);reader=asyncio.create_task(c.read());await c.call('Page.enable');await c.call('Runtime.enable')
  await c.call('Page.navigate',url='http://127.0.0.1:8002/?diag=1')
  for _ in range(100):
   try:
    if await c.evaluate('typeof exportEvidence === "function"'):break
   except:pass
   await asyncio.sleep(.1)
  await c.evaluate(OBSERVE)
  runs=[('v1-ocr-cold','20260921_205640.mp4'),('v2-warm-a','20260921_205735.mp4'),('v3-warm-a','20260921_205754.mp4'),('v1-warm','20260921_205640.mp4'),('v2-warm-b','20260921_205735.mp4'),('v3-warm-b','20260921_205754.mp4')]
  for run,filename in runs:
   directory=OUT/run;directory.mkdir(exist_ok=True)
   await c.call('Browser.setDownloadBehavior',behavior='allow',downloadPath=str(directory),eventsEnabled=True)
   await c.evaluate('sourceSelect.value="video"; sourceSelect.dispatchEvent(new Event("change"));')
   await c.evaluate('sourceChange')
   doc=await c.call('DOM.getDocument');q=await c.call('DOM.querySelector',nodeId=doc['root']['nodeId'],selector='#videoFile')
   await c.call('DOM.setFileInputFiles',nodeId=q['nodeId'],files=[str(ROOT/'reports/video-comparison/compatible'/filename)])
   await c.evaluate('sourceChange')
   for _ in range(100):
    if await c.evaluate('camera.readyState>=2'):break
    await asyncio.sleep(.1)
   config=json.load(urllib.request.urlopen('http://127.0.0.1:8002/api/session'))
   before=json.load(urllib.request.urlopen('http://127.0.0.1:8002/api/ident-readiness'))
   start=await c.evaluate('performance.now()');(directory/'run.json').write_text(json.dumps({'file':filename,'browser_start_ms':start,'config':config,'readiness_before':before,'input_note':'H264 8-bit compatible derivative, same native dimensions/timestamps; detector already warmed by HEVC black-frame probe'},indent=2))
   await c.evaluate('document.getElementById("startButton").click()')
   log=(directory/'browser-events.jsonl').open('w');records={};frames={};post=None;wall=time.monotonic()
   while time.monotonic()-wall<110:
    sample=await c.evaluate('({browser_ms:performance.now(),video_time:camera.currentTime,duration:camera.duration,ended:camera.ended,status:statusText.textContent,events:comparisonEvents.splice(0),requests:evidenceRequests,frames:evidenceFrames,dropped:evidenceDropped,backendDropped})')
    for event in sample['events']:log.write(json.dumps(event,ensure_ascii=False)+'\n')
    for record in sample['requests']:
     rec=dict(record)
     for key,ext in [('crop_jpeg','jpg'),('original_png','png')]:
      encoded=rec.pop(key,None)
      if encoded:
       imagepath=directory/(rec['request_id'].replace(':','_')+'-'+key+'.'+ext)
       if not imagepath.exists():imagepath.write_bytes(base64.b64decode(encoded.split(',',1)[1]))
       rec[key+'_path']=imagepath.name
     records[rec['request_id']]=rec
    for frame in sample['frames']:frames[frame['frame_id']]=frame
    (directory/'collected.json').write_text(json.dumps({'frames':list(frames.values()),'requests':list(records.values()),'browser_dropped':sample['dropped'],'backend_dropped':sample['backendDropped'],'last_state':{k:v for k,v in sample.items() if k not in ['requests','frames','events']}},ensure_ascii=False,indent=2))
    if sample['ended'] and post is None:post=time.monotonic();print(run,'ended',sample['video_time'],'records',len(records),flush=True)
    if post and time.monotonic()-post>20:break
    if 'Cannot' in sample['status'] or 'Unsupported' in sample['status']:print('ERROR',sample['status'],flush=True);break
    await asyncio.sleep(1)
   await c.evaluate('document.getElementById("exportEvidence").click()')
   shot=await c.call('Page.captureScreenshot',format='png',captureBeyondViewport=False);(directory/'ui.png').write_bytes(base64.b64decode(shot['data']))
   await asyncio.sleep(.5);log.close()
   (directory/'debug-final.json').write_text(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8002/api/ident-debug')),ensure_ascii=False,indent=2))
   (directory/'readiness-final.json').write_text(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8002/api/ident-readiness')),indent=2))
   print(run,'saved',len(frames),'frames',len(records),'crops','dropped',sample['dropped'],flush=True)
  await c.evaluate('stop()');(OUT/'browser-errors.json').write_text(json.dumps(c.events,indent=2));reader.cancel()
if __name__ == "__main__": asyncio.run(main())
