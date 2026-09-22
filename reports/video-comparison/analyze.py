from pathlib import Path
import json,csv,collections,statistics,math,cv2
B=Path(__file__).resolve().parent;E=B/'evidence'
SKUS={'A':'aqua-minerale-0-5l','S':'senezhskaya-0-5l','W':'saint-spring-0-75l','P':'prostokvashino-2-5-930ml','D':'domik-v-derevne-2-5-930ml'}
RUNS=['v1-ocr-cold','v1-warm','v2-warm-a','v2-warm-b','v3-warm-a','v3-warm-b']
def physical(run,tid,t,box=None):
 if run.startswith('v1'):
  if tid==2:return 'A' if t<7 else 'S' if t<18 else 'W' if t<27 else 'P'
  if tid in [3,4]:return 'A'
  if tid==5:return 'S'
  if tid==6:return 'W' if t<37.3 else 'D'
  if tid==8:return 'W'
 if run.startswith('v2'):return {1:'D',2:'W',3:'P',4:'A',5:'S'}.get(tid,'ambiguous')
 if run.startswith('v3'):
  if tid==1:return 'P'
  if tid==4:return 'A'
  if tid==5:return 'S'
  if tid in [6,7]:return 'W'
  if tid==8:return 'D'
  if tid==2:return 'W' if t<4.4 or run.endswith('-b') else 'ambiguous' if t<5.2 else 'D'
  if tid==3:return 'D' if t<5 else 'S' if t>11.8 else 'ambiguous'
 return 'ambiguous'
def stat(v):
 v=sorted(x for x in v if isinstance(x,(int,float)))
 return None if not v else {'n':len(v),'median':round(statistics.median(v),2),'min':round(v[0],2),'p90':round(v[min(len(v)-1,math.ceil(.9*len(v))-1)],2),'max':round(v[-1],2)}
server=[json.loads(l) for l in (E/'server-events.jsonl').read_text().splitlines()]
submit={e['request_id']:e['server_monotonic'] for e in server if e['event']=='submit_begin'}
httpstarts={e['call_id']:e for e in server if e['event']=='jev_http_begin'}
httpends={e['call_id']:e for e in server if e['event'] in ['jev_http_end','jev_http_error']}
allreq={};frameby={};summaries={};bottles=[];reqrows=[];allwrong=[]
for run in RUNS:
 d=json.loads((E/run/'collected.json').read_text());runmeta=json.loads((E/run/'run.json').read_text());start=runmeta['browser_start_ms'];frames={f['frame_id']:f for f in d['frames']};frameby.update(frames)
 reqs={r['request_id']:r for r in d['requests']}
 for r in reqs.values():r['physical_id']=physical(run,r['track_id'],r['video_timestamp'],r['bbox']);r['run']=run
 allreq.update(reqs)
 events=[json.loads(l) for l in (E/run/'browser-events.jsonl').read_text().splitlines()]
 received=[e for e in events if e['event']=='received'];det=[e for e in received if e['data'].get('type')=='detection' and e['data'].get('frame_id') in frames]
 detwall={e['data']['frame_id']:e['browser_ms'] for e in det}
 firstcand={};firstdone={};firstocr={};lastmap={};wrongs=[]
 for e in received:
  data=e['data'];f=frames.get(data.get('frame_id'))
  if f:
   for tr in f['tracks']:lastmap[tr['track_id']]=physical(run,tr['track_id'],f['video_timestamp'],tr['bbox'])
  for tid,v in data.get('identification',{}).items():
   p=lastmap.get(int(tid));choice=v.get('choice')
   if not p or p=='ambiguous' or not choice:continue
   obj={'browser_wall_since_play_s':round((e['browser_ms']-start)/1000,3),'browser_ms':e['browser_ms'],'track_id':int(tid),'choice':choice,'confidence':v.get('confidence'),'source_video_time_latest':f['video_timestamp'] if f else None}
   if choice==SKUS[p]:
    firstcand.setdefault(p,obj)
    if v.get('complete'):firstdone.setdefault(p,obj)
   elif v.get('complete'):wrongs.append({**obj,'physical_id':p})
  for r in data.get('diagnostics',{}).get('requests',[]):
   if r.get('ocr_status')=='done' and r['request_id'] in reqs:
    p=reqs[r['request_id']]['physical_id'];firstocr.setdefault(p,{'request_id':r['request_id'],'browser_observed_wall_s':round((e['browser_ms']-start)/1000,3)})
 for r in reqs.values():
  b=r.get('backend',{});rect=r['mapped_rect'];jpeg=E/run/r.get('crop_jpeg_path','absent')
  sharp=None
  if jpeg.is_file():
   im=cv2.imread(str(jpeg));sharp=float(cv2.Laplacian(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
  row={'run':run,'physical_id':r['physical_id'],'session_version':r['session_version'],'frame_id':r['frame_id'],'track_id':r['track_id'],'request_id':r['request_id'],'video_timestamp':r['video_timestamp'],'status':r.get('status'),'reason':r.get('reason'),'mapped_rect':rect,'sharpness':round(sharp,3) if sharp is not None else None,'ocr_status':b.get('ocr_status'),'ocr_text':' | '.join(x['text'] for x in b.get('lines',[])),'ocr_scores':[x.get('score') for x in b.get('lines',[])],'crop_transport_ms':r.get('ack',{}).get('request_to_receipt_ms'),'queue_wait_ms':b.get('queue_wait_ms'),'ocr_ms':b.get('ocr_ms'),'submit_server_monotonic':submit.get(r['request_id']),'crop_path':str(jpeg.relative_to(B)) if jpeg.is_file() else None,'jev_latest':b.get('jev')}
  if row['submit_server_monotonic'] is not None and b.get('ocr_ms') is not None:row['ocr_done_server_estimate']=row['submit_server_monotonic']+(b.get('queue_wait_ms',0)+b['ocr_ms'])/1000
  reqrows.append(row)
 for p in SKUS:
  appearances=[(f,tr) for f in frames.values() for tr in f['tracks'] if physical(run,tr['track_id'],f['video_timestamp'],tr['bbox'])==p]
  appearances.sort(key=lambda x:x[0]['video_timestamp']);prs=[r for r in reqrows if r['run']==run and r['physical_id']==p];accepted=[r for r in prs if r['reason']=='accepted'];done=[r for r in prs if r['ocr_status']=='done'];fd=appearances[0][0] if appearances else None
  first_submit=min(accepted,key=lambda r:r['video_timestamp']) if accepted else None
  ocrfirst=min(done,key=lambda r:r.get('ocr_done_server_estimate',float('inf'))) if done else None
  entry={'run':run,'physical_id':p,'expected_sku':SKUS[p],'first_detection_video_s':fd['video_timestamp'] if fd else None,'first_detection_frame_id':fd['frame_id'] if fd else None,'track_ids':sorted({tr['track_id'] for _,tr in appearances}),'first_ocr_submission_video_s':first_submit['video_timestamp'] if first_submit else None,'first_ocr_submission_request_id':first_submit['request_id'] if first_submit else None,'first_ocr_completion_request_id':ocrfirst['request_id'] if ocrfirst else None,'first_ocr_completion_video_of_crop_s':ocrfirst['video_timestamp'] if ocrfirst else None,'first_ocr_completion_observed':firstocr.get(p),'first_correct_candidate':firstcand.get(p),'first_correct_completed':firstdone.get(p),'crop_requests':len(prs),'accepted_submissions':len(accepted),'ocr_runs':len(done),'duplicate_skips':sum(r['ocr_status']=='duplicate' for r in prs),'rejections':dict(collections.Counter(r['reason'] for r in prs if r['reason']!='accepted')),'source_bottle_height_px':stat([(tr['bbox'][3]-tr['bbox'][1])*4.5 for _,tr in appearances])}
  if fd and p in firstdone:entry['observed_detection_to_completed_wall_s']=round((firstdone[p]['browser_ms']-detwall[fd['frame_id']])/1000,3)
  bottles.append(entry)
 summaries[run]={'session_version':next(iter(frames.values()))['session_version'],'frames':len(frames),'duration_s':runmeta.get('duration',d['last_state']['duration']),'sample_interval_s':stat([b['video_timestamp']-a['video_timestamp'] for a,b in zip(list(frames.values()),list(frames.values())[1:])]),'detect_ms':stat([e['data'].get('detect_ms') for e in det]),'crop_status':dict(collections.Counter(r.get('reason') for r in reqs.values())),'ocr_status':dict(collections.Counter(r.get('backend',{}).get('ocr_status','not_queued') for r in reqs.values())),'browser_dropped':d['browser_dropped'],'backend_dropped':d['backend_dropped'],'wrong_accepted_observations':wrongs,'first_candidates':firstcand,'completed':firstdone}
 allwrong+=wrongs
# Assign HTTP invocations by the nearest preceding identification entry (no foreign clocks).
identbeg=[e for e in server if e['event']=='_identify_begin']
calls=[]
for cid,e in httpstarts.items():
 preceding=[x for x in identbeg if 0<=e['server_monotonic']-x['server_monotonic']<.1]
 ident=preceding[-1] if preceding else {};sources=ident.get('sources',[]);rr=[allreq[r] for r in sources if r in allreq];run=rr[-1]['run'] if rr else None;physical_ids=sorted({r['physical_id'] for r in rr});end=httpends.get(cid,{})
 calls.append({'call_id':cid,'run':run,'physical_ids':physical_ids,'track_id':ident.get('track_id'),'source_request_ids':sources,'server_start':e['server_monotonic'],'duration_ms':round((end.get('server_monotonic',e['server_monotonic'])-e['server_monotonic'])*1000,2),'result':end.get('result'),'error':end.get('error_type'),'input_lines':e['lines']})
for b in bottles:
 cs=[c for c in calls if c['run']==b['run'] and b['physical_id'] in c['physical_ids']];b['jev_calls']=len(cs);b['jev_errors']=sum(bool(c['error']) for c in cs)
for run,s in summaries.items():
 rows=[r for r in reqrows if r['run']==run];cs=[c for c in calls if c['run']==run]
 for field in ['crop_transport_ms','queue_wait_ms','ocr_ms']:s[field]=stat([r.get(field) for r in rows])
 s['jev_calls']=len(cs);s['jev_errors']=sum(bool(c['error']) for c in cs);s['jev_ms']=stat([c['duration_ms'] for c in cs])
for name,data in [('comparison-summary',summaries),('per-bottle',bottles),('per-request',reqrows),('jev-calls',calls)]:
 (B/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
 if isinstance(data,list) and data:
  keys=list(dict.fromkeys(k for r in data for k in r));f=(B/(name+'.csv')).open('w');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in data);f.close()
print(json.dumps(summaries,ensure_ascii=False,indent=2));print('WRONG ACCEPTED',allwrong)
