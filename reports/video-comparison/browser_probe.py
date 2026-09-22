import asyncio,json,urllib.request,sys,base64
from pathlib import Path
from browser_comparison import CDP
import websockets
async def main():
 tab=next(t for t in json.load(urllib.request.urlopen('http://127.0.0.1:9223/json/list')) if t['type']=='page')
 async with websockets.connect(tab['webSocketDebuggerUrl'],max_size=100000000) as ws:
  c=CDP(ws);read=asyncio.create_task(c.read());print(json.dumps(await c.evaluate(sys.argv[1]),ensure_ascii=False));read.cancel()
asyncio.run(main())
