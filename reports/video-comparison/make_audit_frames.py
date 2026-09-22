from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parent;E=BASE/'evidence';O=BASE/'frames';font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for run,video,times in [('v1-ocr-cold','20260921_205640.mp4',[3.3,9.6,18.8,24,28.5,39.4,47.5]),('v2-warm-a','20260921_205735.mp4',[.7,3.1,8.3]),('v3-warm-a','20260921_205754.mp4',[.7,2.7,4.4,5.8,7.1,8.6,10.7,12]),('v3-warm-b','20260921_205754.mp4',[3.6,7.2,9.2,12.2])]:
 d=json.loads((E/run/'collected.json').read_text());cap=cv2.VideoCapture(str(BASE/'compatible'/video));sheet=Image.new('RGB',(1280,((len(times)+1)//2)*405),'#1b1b1b');draw=ImageDraw.Draw(sheet)
 for i,t in enumerate(times):
  f=min(d['frames'],key=lambda f:abs(f['video_timestamp']-t));cap.set(cv2.CAP_PROP_POS_MSEC,f['video_timestamp']*1000);ok,img=cap.read()
  if not ok:continue
  im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).resize((640,360));dr=ImageDraw.Draw(im)
  for tr in f['tracks']:
   x1,y1,x2,y2=tr['bbox'];dr.rectangle((x1,y1*.75,x2,y2*.75),outline='yellow',width=2);dr.text((x1,max(0,y1*.75)),str(tr['track_id']),fill='black',stroke_fill='yellow',stroke_width=2,font=font)
  x=i%2*640;y=i//2*405;sheet.paste(im,(x,y+40));draw.text((x+5,y+5),f"{run} t={f['video_timestamp']:.2f} f={f['frame_id']} s={f['session_version']}",font=font,fill='white')
 sheet.save(O/(run+'-track-audit.jpg'))
