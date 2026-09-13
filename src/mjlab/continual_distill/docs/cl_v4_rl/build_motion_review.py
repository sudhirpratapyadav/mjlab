"""Build the concise paired Lift motion review from verified measurements/videos."""
import argparse
import html
import json
from pathlib import Path


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--audit',type=Path,required=True)
  p.add_argument('--confirmation',type=Path)
  p.add_argument('--videos',type=Path,required=True)
  a=p.parse_args();data=json.loads(a.audit.read_text());meta=json.loads((a.videos/'videos.json').read_text())
  assert meta['paired_initial_states_verified'] and len(meta['clips'])==6
  before=data['baseline'];after=data;confirm=json.loads(a.confirmation.read_text()) if a.confirmation else None
  if confirm:assert confirm['checkpoint_sha256']==data['checkpoint_sha256']
  clips={(r['label'],r['env_id']):r for r in meta['clips']}
  for r in meta['clips']:
    expected=before['checkpoint_sha256'] if r['label']=='before' else after['checkpoint_sha256']
    assert r['checkpoint_sha256']==expected and r['fps']==50 and r['simulation_seconds']==r['encoded_seconds']
  fields=[('Peak joint speed','max_joint_speed_rad_s','rad/s'),('Acceleration estimate','sampled_acceleration_rms_rad_s2','rad/s²'),('Largest target step','max_consecutive_target_jump_rad','rad')]
  metric_rows=''.join(f'<tr><td>{name}</td><td>{before["summary"][key]["p95"]:.2f} {unit}</td><td>{after["summary"][key]["p95"]:.2f} {unit}</td></tr>' for name,key,unit in fields)
  validation=f'{data["successes"]}/{data["episodes"]}'
  confirmation=f'{confirm["successes"]}/{confirm["episodes"]}' if confirm else 'Not qualified / not run'
  motion_pass=all(data['provisional_motion_gates'].values())
  rate_pass=data['successes']/data['episodes']>.9 and confirm is not None and confirm['successes']/confirm['episodes']>.9
  review_status='awaiting_user' if motion_pass and rate_pass else ('motion_targets_not_met' if not motion_pass else 'task_success_not_met')
  verdict=('Ready for your visual review' if motion_pass and rate_pass else
           'Motion targets not met — further Lift tuning needed' if not motion_pass else
           'Task success below target — further Lift tuning needed')
  pairs=[]
  for lane in [0,1,2]:
    cards=[]
    for label,title in [('before','Original teacher'),('after','Motion trial')]:
      r=clips[label,lane];outcome='Task succeeded' if r['success'] else 'Task failed'
      cards.append(f'<div><h3>{title} <span>{outcome}</span></h3><video controls playsinline preload="metadata" poster="{r["poster"]}" aria-label="{title}, episode {lane}"><source src="{r["video"]}" type="video/mp4"></video></div>')
    pairs.append(f'<article data-pair="{lane}"><div class="pair-title"><h2>Episode {lane+1}</h2><button type="button">Play both from start</button></div><div class="pair">'+''.join(cards)+'</div></article>')
  page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lift-Cube motion review</title>
<style>*{box-sizing:border-box}body{margin:0;background:#10151b;color:#e7edf4;font:16px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px 22px}a{color:#90d5ff}.eyebrow{color:#8fb3c8;font-size:13px;letter-spacing:.12em;text-transform:uppercase}h1{font-size:clamp(28px,5vw,44px);margin:8px 0}h2{font-size:20px}h3{font-size:16px;margin:8px 0}h3 span{display:block;font-size:13px;color:#b1c5d2;font-weight:400}.intro{max-width:790px;color:#c2cfdb}.status{border-left:3px solid #7bd7ac;padding:12px 18px;background:#1a2630;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair-title{display:flex;align-items:center;justify-content:space-between;gap:12px}article{margin:36px 0}video{width:100%;display:block;background:#000;border-radius:10px}button{background:#7bd7ac;border:0;border-radius:8px;padding:10px 14px;font-weight:650;cursor:pointer}button:disabled{opacity:.6}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;max-width:780px}td,th{text-align:left;padding:12px;border-bottom:1px solid #334350}th{color:#adc3d4}.note{font-size:13px;color:#a9bac8}.error{color:#ffc3ad}@media(max-width:650px){.pair{grid-template-columns:1fr}main{padding:24px 14px}.pair-title{align-items:flex-start}td,th{padding:8px}}</style>
<main><a href="/v4-rl/">← All RL teachers</a><p class="eyebrow">One teacher • motion experiment</p><h1>Lift-Cube: speed &amp; smoothness</h1>
<p class="intro">Compare the original teacher with the motion trial on the same three starting conditions. Every video plays at <strong>1× real time, 50 fps</strong>. These episodes were selected before evaluating the trial.</p>
'''+f'<div class="status"><strong>{verdict}</strong><br><strong>Trial success:</strong> {validation} validation · {confirmation} independent confirmation.<br>The original teacher passed 118/128 and 117/128. Other teachers remain paused pending your review.</div>'+'''
<h2>Motion across all 128 evaluation episodes</h2><p class="note">Values are the 95th percentile of episode measurements, including failures. Acceleration is estimated from recorded 20 ms velocity samples. A target step is the change in commanded joint angle, not physical joint travel.</p><div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Original</th><th>Motion trial</th></tr></thead><tbody>'''+metric_rows+'''</tbody></table></div>
'''+''.join(pairs)+'''
<p class="note">The physical task, success rules, 20-second episode limit, observations and action interface are unchanged. The trial changes only training rewards for motion. The original checkpoint is retained.</p><p class="error" id="error" role="status"></p>
</main><script>
document.querySelectorAll('[data-pair]').forEach(pair=>{const button=pair.querySelector('button'),videos=[...pair.querySelectorAll('video')];button.addEventListener('click',async()=>{button.disabled=true;document.querySelector('#error').textContent='';try{for(const v of document.querySelectorAll('video'))v.pause();for(const v of videos){v.muted=true;v.playbackRate=1;v.currentTime=0;if(v.readyState<2)await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Video loading timed out')),15000);v.addEventListener('loadeddata',()=>{clearTimeout(timer);resolve()},{once:true});v.addEventListener('error',()=>{clearTimeout(timer);reject(Error('Video could not load'))},{once:true});v.load()})}await Promise.all(videos.map(v=>v.play()))}catch(e){document.querySelector('#error').textContent=e.message+' — use the individual video controls.'}finally{button.disabled=false}})});
</script></html>'''
  (a.videos/'index.html').write_text(page)
  public=dict(task=data['task'],validation_successes=data['successes'],episodes=data['episodes'],confirmation_successes=confirm['successes'] if confirm else None,motion_gates=data.get('provisional_motion_gates'),before_summary=before['summary'],after_summary=after['summary'],review_status=review_status,paired_env_ids=[0,1,2])
  (a.videos/'summary.json').write_text(json.dumps(public,indent=2)+'\n');print('Built paired motion review',a.videos)


if __name__=='__main__':main()
