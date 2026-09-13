// Run with the isolated VPS Playwright installation.
import fs from 'node:fs';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE);
const [url, output] = process.argv.slice(2);
const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH,
  headless: true, args: ['--no-sandbox']});
const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
const check = (condition, message) => {if (!condition) throw Error(message)};
try {
  await page.goto(url);
  check(await page.locator('video').count() === 6, 'Expected six videos');
  check(await page.locator('[data-pair]').count() === 3, 'Expected three pairs');
  const videos = [];
  for (let pair = 0; pair < 3; pair++) {
    await page.locator(`[data-pair="${pair}"] button`).click();
    await page.waitForFunction(id => [...document.querySelectorAll(`[data-pair="${id}"] video`)]
      .every(v => v.currentTime > .15 && !v.paused), pair, {timeout: 20000});
    const values = await page.locator(`[data-pair="${pair}"] video`).evaluateAll(vs => vs.map(v => ({
      src: v.currentSrc, width: v.videoWidth, height: v.videoHeight,
      duration: v.duration, currentTime: v.currentTime, playbackRate: v.playbackRate,
      decodedFrames: v.getVideoPlaybackQuality().totalVideoFrames
    })));
    for (const v of values) check(v.width === 960 && v.height === 540 &&
      v.playbackRate === 1 && v.decodedFrames > 0, 'Invalid real-time video playback');
    check(Math.abs(values[0].currentTime - values[1].currentTime) < .2, 'Pair start drift');
    check(await page.locator('#error').innerText() === '', 'Pair playback error');
    videos.push(...values);
    await page.locator('video').evaluateAll(vs => vs.forEach(v => v.pause()));
  }
  await page.screenshot({path: output + '-desktop.png', fullPage: false});
  await page.setViewportSize({width: 390, height: 844});
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile overflow');
  await page.screenshot({path: output + '-mobile.png', fullPage: false});
  check(errors.length === 0, 'Browser errors: ' + errors.join('; '));
  fs.writeFileSync(output + '.json', JSON.stringify({url, verified_utc: new Date().toISOString(),
    paired_playback: true, real_time: true, mobile_overflow: false, browser_errors: errors, videos}, null, 2) + '\n');
  console.log('Verified three pairs, six real-time videos and mobile layout');
} finally {await browser.close()}
