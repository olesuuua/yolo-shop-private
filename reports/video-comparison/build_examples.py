from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,cv2,numpy as np,html
B=Path(__file__).resolve().parent;rows=json.loads((B/'per-request.json').read_text());R={r['request_id']:r for r in rows};O=B/'examples';O.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
ids=['361:2:4','374:4:13','388:2:31','404:2:61','422:6:109','437:4:122','553:4:285','437:3:121','452:4:140','463:7:160','455:3:148','580:3:329','571:3:311']
notes={
'571:3:311':'Domik front label is readable, crop accepted (sharpness 31.6), but completed OCR returns no text; not a Jev failure.',
'361:2:4':'Aqua: brand evidence; accepted candidate. Variant/size not independently confirmed.',
'374:4:13':'Aqua: readable brand rejected by global blur gate. Neighboring Senezhskaya label at lower left.',
'388:2:31':'Saint Spring: readable brand and still-water text; rejected before OCR.',
'404:2:61':'Prostokvashino: curved brand and milk/fat text visible; rejected before OCR.',
'422:6:109':'Domik: OCR mixes its brand with Saint Spring behind/through bottle; later accumulated candidate accepted.',
'437:4:122':'Raised view, initial Aqua frame: AKBA survives and yields accepted candidate.',
'553:4:285':'Raised view repeat: brand wraps away; only AKB (3 chars) survives. No Jev call for this track.',
'437:3:121':'Prostokvashino: accepted crop, OCR only generic milk/2.5%; Jev correctly remains unresolved.',
'452:4:140':'Aqua plus neighboring Senezhskaya tail CKA; candidate 0.50, not completed.',
'463:7:160':'Saint Spring crop includes neighboring milk label; OCR 2,50 comes from the wrong physical label.',
'455:3:148':'Domik during rotation: blank/dark side of label; accepted crop returns no text.',
'580:3:329':'Senezhskaya reacquired as old Domik track 3; crop includes neighboring milk 2.5%. Jev blocked by cap.'}
examples=[]
for rid in ids:
 r=R[rid];run=r['run'];t=r['video_timestamp'];stem={'v1':'20260921_205640','v2':'20260921_205735','v3':'20260921_205754'}[run[:2]];cap=cv2.VideoCapture(str(B/'compatible'/(stem+'.mp4')))
 actual=cv2.imread(str(B/r['crop_path']));rect=r['mapped_rect'];x,y,w,h=[rect[k] for k in ['x','y','w','h']]
 target=cv2.resize(cv2.cvtColor(actual,cv2.COLOR_BGR2GRAY),(96,160)).astype(float);target-=target.mean();target/=np.linalg.norm(target)+1e-8
 cap.set(cv2.CAP_PROP_POS_MSEC,max(0,t-.45)*1000);best=None
 for _ in range(40):
  ok,frame=cap.read()
  if not ok:break
  tm=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
  crop=frame[y:y+h,x:x+w]
  if crop.size==0:continue
  f=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(96,160)).astype(float);f-=f.mean();f/=np.linalg.norm(f)+1e-8;score=float((f*target).sum())
  if best is None or score>best[0]:best=(score,tm,frame.copy())
  if tm>t+.12:break
 cap.release();score,tm,frame=best;im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));prefix=rid.replace(':','_');im.save(O/(prefix+'-source-reconstructed.jpg'),quality=95)
 origreq=next(q for q in json.loads((B/'evidence'/run/'collected.json').read_text())['requests'] if q['request_id']==rid);box=origreq['bbox'];annot=im.resize((960,540));draw=ImageDraw.Draw(annot)
 draw.rectangle((box[0]*1.5,box[1]*1.125,box[2]*1.5,box[3]*1.125),outline='yellow',width=3);draw.rectangle((x/4,y/4,(x+w)/4,(y+h)/4),outline='orange',width=2)
 panel=Image.new('RGB',(1440,760),'#171b25');dr=ImageDraw.Draw(panel);panel.paste(annot,(0,75));cropim=Image.open(B/r['crop_path']);cropim.thumbnail((450,590));panel.paste(cropim,(975,75));dr.text((15,10),f"{run} | physical {r['physical_id']} | t={t:.3f}s | frame {r['frame_id']} | track {r['track_id']} | request {rid}",font=font,fill='white');dr.text((15,40),'Reconstructed source (yellow=request; orange=mapped crop)                         Exact uploaded crop',font=small,fill='#e2e8f0')
 text=f"{notes[rid]}\nStatus: {r['reason']}; native JPEG {actual.shape[1]}x{actual.shape[0]}; sharpness {r['sharpness']}; OCR: {r['ocr_text'] or '[not run / empty; see status]'}"
 import textwrap
 wrapped='\n'.join('\n'.join(textwrap.wrap(line,135)) for line in text.splitlines());dr.multiline_text((15,640),wrapped,font=small,fill='white',spacing=6);panel.save(O/(prefix+'-panel.jpg'),quality=92)
 examples.append({'request_id':rid,'run':run,'frame_id':r['frame_id'],'track_id':r['track_id'],'session_version':r['session_version'],'sample_video_timestamp':t,'reconstructed_time':tm,'match_ncc':score,'note':notes[rid],'panel':prefix+'-panel.jpg','source':prefix+'-source-reconstructed.jpg','crop':'../'+r['crop_path'],'mapped_rect':rect,'ocr_text':r['ocr_text'],'sharpness':r['sharpness']})
(B/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
page='''<!doctype html><meta charset="utf-8"><title>Bottle replay evidence</title><style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1450px;margin:24px auto;padding:16px}a{color:#93c5fd}img{max-width:100%}article{margin:40px 0}pre{white-space:pre-wrap}</style><h1>Bottle replay evidence</h1><p>Original-frame export is unavailable at 4K. Left panels reconstruct a source frame by best visual crop match near the recorded media timestamp; they are not retained browser originals. Right panels use the exact uploaded crop bytes. Open native images for full-resolution inspection.</p>'''
for e in examples:
 page+=f'<article><h2>{html.escape(e["run"]+" / "+e["request_id"])}</h2><img src="{e["panel"]}"><p>{html.escape(e["note"])}</p><a href="{e["crop"]}">Exact native JPEG crop</a> · <a href="{e["source"]}">Reconstructed native source</a><pre>{html.escape(json.dumps(e,ensure_ascii=False,indent=2))}</pre></article>'
(O/'index.html').write_text(page)
print(json.dumps([{k:e[k] for k in ['request_id','sample_video_timestamp','reconstructed_time','match_ncc']} for e in examples],indent=2))
