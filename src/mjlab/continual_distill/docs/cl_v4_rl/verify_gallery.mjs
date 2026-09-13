// Run on the VPS with its isolated Playwright install and existing Chromium.
import fs from 'node:fs';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE);
const url = process.argv[2];
const output = process.argv[3];
const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,args:['--no-sandbox']});
const page = await browser.newPage({viewport:{width:1440,height:1050}});
const errors=[];
page.on('pageerror', error => errors.push(String(error)));
if(url.startsWith('https://')){
  await page.goto(new URL('/',url).href);
  const card=page.locator('.cards > a').first();
  if(await card.getAttribute('href')!=='/v4-rl/')throw new Error('RL card is not the newest home-page card');
  await card.click();
  await page.waitForURL('**/v4-rl/');
}
await page.goto(url);
await page.locator('.task').last().waitFor();
const expect = (value,message) => {if(!value)throw new Error(message)};
expect(await page.locator('.task').count()===24,'Expected 24 task cards');
await page.locator('[data-filter="certified"]').click();
expect(await page.locator('.task:visible').count()===17,'Expected 17 certified cards');
await page.locator('[data-filter="progress"]').click();
expect(await page.locator('.task:visible').count()===7,'Expected 7 unfinished cards');
await page.locator('[data-filter="all"]').click();
await page.locator('#search').fill('lift cube');
expect(await page.locator('.task:visible').count()===1,'Task search failed');
await page.locator('#search').fill('no-such-task');
expect(await page.locator('#empty').isVisible(),'Empty state failed');
await page.locator('#search').fill('');
await page.screenshot({path:output+'-desktop.png',fullPage:false});
const videos=[];
for(let i=0;i<24;i++){
  const result=await page.locator('video').nth(i).evaluate(async video=>{
    video.muted=true;
    await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('Video metadata timeout: '+video.currentSrc)),15000);
      video.addEventListener('loadeddata',()=>{clearTimeout(timer);resolve()},{once:true});
      video.addEventListener('error',()=>{clearTimeout(timer);reject(new Error('Video load error'))},{once:true});
      video.load();
    });
    await video.play();
    await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('Playback did not advance: '+video.currentSrc)),15000);
      const check=()=>{if(video.currentTime>0.02){clearTimeout(timer);video.removeEventListener('timeupdate',check);resolve()}};
      video.addEventListener('timeupdate',check);check();
    });
    video.pause();
    return {src:video.currentSrc,width:video.videoWidth,height:video.videoHeight,duration:video.duration,time:video.currentTime};
  });
  expect(result.width>0&&result.height>0&&result.time>0,'Video failed playback');
  videos.push(result);
}
await page.setViewportSize({width:390,height:844});
await page.screenshot({path:output+'-mobile.png',fullPage:false});
expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'Mobile horizontal overflow');
expect(errors.length===0,'Browser errors: '+errors.join('; '));
fs.writeFileSync(output+'.json',JSON.stringify({url,verified_utc:new Date().toISOString(),cards:24,certified:17,in_progress:7,filters:true,search:true,mobile_overflow:false,browser_errors:errors,videos},null,2)+'\n');
await browser.close();
console.log('Verified 24 playable videos, filters, search and mobile layout');
