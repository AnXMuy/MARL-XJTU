() => {
  // Stable canvas: the timer only replaces the hidden data payload, never this drawing node.
  let lastPayload='', frame=null;
  const drawCharts=()=>{
    const canvas=document.querySelector('#home-chart'), payload=document.querySelector('#chart-payload');
    if(!canvas || !payload) return;
    let f;try {f=JSON.parse(payload.textContent);}catch(e){return;}
    const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    const w=rect.width,h=rect.height;
    if(w<1 || h<1)return;
    const buffer=document.createElement('canvas');buffer.width=w*dpr;buffer.height=h*dpr;
    const c=buffer.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#fffef5';c.fillRect(0,0,w,h);
    const columns=w<540?1:2, rows=4/columns, cw=w/columns,ch=h/rows;
    const titles=['总功率 (kW)','累计电费 (元)','四房温度 (℃)','累计舒适偏离 (℃·h)'];
    const roomColors=['#146c94','#9966bd','#48834a','#bd7934'];
    for(let panel=0;panel<4;panel++) {
      const ox=(panel%columns)*cw,oy=Math.floor(panel/columns)*ch;
      const left=ox+52,right=ox+cw-18,top=oy+48,bottom=oy+ch-35;
      const axis=panel===0?'x':`x${panel+1}`;
      const traces=f.data.filter(t=>(t.xaxis||'x')===axis);
      let max=panel===0?9:panel===2?28:panel===1?40:2;
      if(panel!==0 && panel!==2) max=Math.max(max,...traces.flatMap(t=>t.y||[]))*1.05;
      const min=panel===2?10:0;
      c.fillStyle='#17352f';c.font='bold 14px sans-serif';c.fillText(titles[panel],ox+16,oy+23);
      if(panel===2){c.font='11px sans-serif';['客厅','卧室','厨房','浴室'].forEach((v,i)=>{c.fillStyle=roomColors[i];c.fillText(v,left+i*48,oy+40);});}
      c.font='11px sans-serif';
      for(let i=0;i<=4;i++) {
        const y=bottom-(bottom-top)*i/4;c.strokeStyle='#e2e5dc';c.setLineDash([]);c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();
        c.fillStyle='#53665e';c.fillText((min+(max-min)*i/4).toFixed(panel===3?1:0),ox+8,y+4);
        const x=left+(right-left)*i/4;c.fillText(String(i*6),x-4,bottom+18);
      }
      c.save();c.beginPath();c.rect(left,top,right-left,bottom-top);c.clip();
      for(const t of traces) {
        c.strokeStyle=t.line?.color||'#146c94';c.lineWidth=t.line?.width||2;
        c.setLineDash(t.line?.dash==='dash'?[7,5]:t.line?.dash==='dot'?[2,4]:[]);c.beginPath();
        (t.y||[]).forEach((v,i)=>{const x=left+(right-left)*t.x[i]/24,y=bottom-(v-min)/(max-min)*(bottom-top);if(i===0)c.moveTo(x,y);else c.lineTo(x,y);});c.stroke();
      }
      if(panel===0){const y=bottom-4/max*(bottom-top);c.strokeStyle='#9a5260';c.setLineDash([3,4]);c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();}
      c.restore();
    }
    // Commit a complete frame in one operation; no blank intermediate image.
    if(canvas.width!==buffer.width)canvas.width=buffer.width;
    if(canvas.height!==buffer.height)canvas.height=buffer.height;
    canvas.getContext('2d').drawImage(buffer,0,0);
  };
  const scheduleDraw=()=>{if(frame===null)frame=requestAnimationFrame(()=>{frame=null;drawCharts();});};
  new MutationObserver(()=>{const value=document.querySelector('#chart-payload')?.textContent;if(value && value!==lastPayload){lastPayload=value;scheduleDraw();}}).observe(document.documentElement,{childList:true,subtree:true,characterData:true});
  window.addEventListener('resize',scheduleDraw);scheduleDraw();
  const status = () => document.querySelector('#voice-status');
  const tell = text => { if(status()) status().textContent = text; };
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  window.homeVoice = window.homeVoice || { listening:false };
  const state = window.homeVoice;
  window.stopHomeVoice = () => {state.listening=false; if(state.recognition) state.recognition.abort(); tell('麦克风已关闭');};
  window.startHomeVoice = () => {
    if (!Recognition) {tell('此浏览器不支持持续语音，请使用 Chrome / Edge 或文字输入');return;}
    if (state.listening) return;
    const recognition = new Recognition(); state.recognition=recognition;
    recognition.lang='zh-CN'; recognition.continuous=true; recognition.interimResults=true;
    state.listening=true;
    recognition.onstart=()=>tell('正在聆听 · 可连续说出需求');
    recognition.onresult=event=>{
      for(let i=event.resultIndex;i<event.results.length;i++) {
        const text=event.results[i][0].transcript.trim();
        if(!event.results[i].isFinal) {tell('听到：'+text);continue;}
        const field=document.querySelector('#command-input textarea');
        if(field && text) {
          Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(field,text);
          field.dispatchEvent(new Event('input',{bubbles:true}));
          setTimeout(()=>(document.querySelector('#command-send button') || document.querySelector('button#command-send'))?.click(),100);
          tell('已接收：'+text+' · 继续聆听');
        }
      }
    };
    recognition.onerror=event=>{
      if(['not-allowed','service-not-allowed','audio-capture','network'].includes(event.error)) state.listening=false;
      tell('语音不可用：'+event.error+'。可继续用文字输入。');
    };
    recognition.onend=()=>{if(state.listening) setTimeout(()=>{try{recognition.start();}catch(e){tell('请重新开启麦克风');state.listening=false;}},300);};
    try {recognition.start();} catch(e) {state.listening=false;tell('无法启动麦克风，请检查权限');}
  };
  window.addEventListener('pagehide',window.stopHomeVoice);
}
