"""UI end-to-end check. Speech events are injected: not a real acoustic ASR test."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('assets/demo');out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH'),headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1600,'height':1050},device_scale_factor=1)
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.add_init_script('''window.SpeechRecognition=class {start(){this.onstart?.();} abort(){this.onend?.();}}''')
 page.goto('http://127.0.0.1:7861',wait_until='domcontentloaded');page.locator('.data-panel .board-top b').first.wait_for(timeout=60000)
 assert page.get_by_label('自动推进仿真',exact=True).is_checked()
 assert page.locator('.data-panel table').first.locator('tbody tr').count()==4
 assert page.locator('svg[aria-label*=完整住宅]').count()==0
 page.wait_for_function("document.body.innerText.includes('全屋 00:15') || document.body.innerText.includes('全屋 00:30')",timeout=30000)
 page.get_by_label('自动推进仿真',exact=True).uncheck()
 page.wait_for_function("document.querySelector('#home-chart')?.width > 300")
 page.evaluate("window.savedPlot=document.querySelector('#home-chart')")
 page.get_by_label('自动推进仿真',exact=True).check()
 page.wait_for_timeout(5000)
 assert page.evaluate("window.savedPlot === document.querySelector('#home-chart')"), 'Canvas DOM replaced during tick'
 page.get_by_label('自动推进仿真',exact=True).uncheck()
 page.screenshot(path=str(out/'dashboard.png'),full_page=True)
 page.get_by_role('button',name='睡眠',exact=True).click()
 page.get_by_text('卧室进入睡眠目标，其余房间停止舒适供暖。',exact=False).wait_for(timeout=20000)
 page.get_by_role('button',name='开启持续聆听',exact=True).click()
 page.wait_for_function('window.homeVoice?.listening === true')
 for text in ['客厅有点冷','省电模式']:
  page.evaluate('''text=>{const r=[{transcript:text}];r.isFinal=true;window.homeVoice.recognition.onresult({resultIndex:0,results:[r]});}''',text)
  page.wait_for_timeout(1800)
  assert page.locator('#command-input textarea').input_value()==text
  page.locator('.caption').filter(has_text=text).wait_for(timeout=20000)
 page.get_by_role('button',name='关闭麦克风',exact=True).click();page.wait_for_function('window.homeVoice.listening === false')
 page.get_by_text('设计你的四区户型 · 修改后重新开始对照',exact=True).click()
 page.get_by_role('button',name='应用户型 · 重启同日对照',exact=True).click()
 page.get_by_text('新户型已生效：',exact=False).wait_for(timeout=20000)
 page.get_by_label('自动推进仿真',exact=True).check()
 page.locator('.data-panel .board-top small').first.filter(has_text='全屋 00:15').wait_for(timeout=20000)
 page.get_by_label('自动推进仿真',exact=True).uncheck()
 page.get_by_role('button',name='快速查看当日总结',exact=True).click()
 page.locator('.data-panel .board-top small').first.filter(has_text='全屋 24:00').wait_for(timeout=60000)
 page.wait_for_timeout(800)
 page.screenshot(path=str(out/'comparison.png'),full_page=True)
 page.get_by_label('简单恒温器',exact=True).check()
 page.locator('.data-panel .board-top b').first.filter(has_text='模型与简单恒温器').wait_for(timeout=20000)
 assert '00:00' in page.locator('.data-panel .board-top small').first.inner_text()
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(500)
 assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+2'), 'Mobile horizontal overflow'
 page.screenshot(path=str(out/'mobile.png'),full_page=True)
 assert not errors,errors
 (out/'browser_check.json').write_text(json.dumps({'passed':True,'javascript_errors':errors,'checks':['load','stable canvas node across automatic updates','named baseline switch resets paired day','scene','two continuous speech result events','mic stop','layout apply','default automatic clock without click','full day','mobile screenshot'],'limitation':'Speech events mocked; no microphone hardware or acoustic recognition tested.'},indent=2))
 browser.close()
