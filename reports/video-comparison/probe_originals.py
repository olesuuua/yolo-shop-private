import asyncio,json,urllib.request
from pathlib import Path
import websockets
from browser_comparison import CDP,ROOT,OUT
async def main():
 tab=next(t for t in json.load(urllib.request.urlopen('http://127.0.0.1:9223/json/list')) if t['type']=='page')
 rows=[]
 async with websockets.connect(tab['webSocketDebuggerUrl']) as ws:
  c=CDP(ws);reader=asyncio.create_task(c.read());await c.evaluate('stop(); sourceSelect.value="video";')
  for p in sorted((ROOT/'data/local/videos').glob('*.mp4')):
   doc=await c.call('DOM.getDocument');q=await c.call('DOM.querySelector',nodeId=doc['root']['nodeId'],selector='#videoFile');await c.call('DOM.setFileInputFiles',nodeId=q['nodeId'],files=[str(p)]);await c.evaluate('sourceChange');await c.evaluate('camera.play()');await asyncio.sleep(.5)
   row=await c.evaluate('({video_width:camera.videoWidth,video_height:camera.videoHeight,ready_state:camera.readyState,error:camera.error,current_time:camera.currentTime,duration:camera.duration,total_video_frames:camera.getVideoPlaybackQuality().totalVideoFrames})');row['file']=p.name;rows.append(row);await c.evaluate('camera.pause()')
  await c.evaluate('stop()');reader.cancel()
 (OUT/'original-codec-probes.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
asyncio.run(main())
