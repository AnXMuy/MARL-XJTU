"""Paired live control-room demo for the v2 learned policy, no EPyMARL imports.
Run: python -m demo.redesign --checkpoint .../actor_export.pt
"""
import argparse,html,json,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
# Import before Gradio worker threads: Plotly probes pandas during validation.
import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from experiments.v2.env import HomeBatch
from experiments.v2.evaluate import Learned,Rule,preference
from experiments.v2.controllers import FeasibleLearned,PolicyLookahead
from demo.intent import IntentParser,merge

NAMES=['客厅','卧室','厨房','浴室']
CSS='''
.data-panel{background:#fffef5;border:1px solid #cad0c3;border-radius:10px;padding:16px}.data-panel small{color:#456257}.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:12px 8px;text-align:right;border-bottom:1px solid #dde1d6;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:#456257;background:#eef0e6}
.generating{opacity:1!important}.wrap.generating{display:none!important} .data-panel{min-height:280px} #live-chart{height:650px;min-height:650px} #home-chart{width:100%;height:610px;display:block} #chart-data{display:none!important} .chart-key{font-size:12px;padding:10px} @media(max-width:650px){#live-chart{height:1040px;min-height:1040px}#home-chart{height:990px}} .data-panel p{font-size:12px;margin:8px 0}

.gradio-container{max-width:1540px!important;background:#f1f0e9!important;color:#17352f!important;font-family:"Noto Sans SC","Microsoft YaHei",sans-serif!important}
#masthead{border-bottom:2px solid #17352f;padding:18px 0 22px;margin-bottom:16px}#masthead h1{font-size:36px;letter-spacing:-1.8px;font-weight:750;margin:4px 0}#masthead p{color:#55716a;margin:6px 0}
@media(max-width:650px){.kpis{grid-template-columns:repeat(2,1fr)}#masthead h1{font-size:26px}}
.eyebrow{font:12px monospace;letter-spacing:2px;color:#607b70}.board-top{display:flex;justify-content:space-between;font-size:15px;margin-bottom:10px}.board-top small{font-size:12px;color:#456257}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#cad0c3;border:1px solid #cad0c3;border-radius:12px;overflow:hidden;margin-bottom:14px}.kpi{padding:15px 18px;background:#fffef5}.kpi strong{display:block;font:25px monospace;margin:6px 0}.kpi span{font-size:12px;color:#60766b}.eventline{padding:14px 18px;border-left:4px solid #c76c3d;background:#e7e4d7;border-radius:4px;min-height:62px}.caption{font-size:12px;color:#66786e}.constraint{margin:10px 0 0;font:13px monospace}.meter{height:8px;background:#486157;border-radius:4px;margin-top:8px}.meter div{height:100%;border-radius:4px;background:#d4dba1}
'''

class Session:
    def __init__(self,checkpoint,seed=100001,llm_url=None,controller="feasible",horizon=16,baseline="coordinated"):
        self.baseline_kind=baseline
        self.controller=controller;self.horizon=horizon
        self.layout=([36,28,20,16],[[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]]);self.checkpoint=checkpoint;self.seed=seed;self.parser=IntentParser(__import__('demo.llm_client',fromlist=['LLMClient']).LLMClient(llm_url) if llm_url else None);self.lock=threading.RLock();self.reset()
    def reset(self):
        with self.lock:
            self.policy=({"raw":Learned,"feasible":FeasibleLearned,"lookahead":lambda path:PolicyLookahead(path,self.horizon)}[self.controller](self.checkpoint)) if self.checkpoint else Rule(True)
            self.name=({'raw':'MAPPO 学习策略','feasible':'MAPPO + 功率约束分配','lookahead':'MAPPO + 前瞻调控'}[self.controller]) if self.checkpoint else '协调规则 · 未加载模型'
            self.env=HomeBatch();self.other=HomeBatch();self.rule=Rule(self.baseline_kind=="coordinated")
            self.baseline_name="协调规则" if self.baseline_kind=="coordinated" else "简单恒温器"
            for e in (self.env,self.other):e.reset(seed=self.seed,preference=preference('default'),switches=False)
            if self.layout:
                for e in (self.env,self.other):e.set_layout(*self.layout)
            self.rows=[];self.baseline=[];self.paused=True;self.trace={};self.previous=[]
            self.subtitle='输入你的需求，或从下方选择一个生活场景。';self.events=[]
    def configure(self,areas,doors):
        graph=np.zeros((4,4),dtype=np.float32)
        for edge in doors:
            i,j=map(int,edge.split(':')[0].split('-'));graph[i,j]=graph[j,i]=1
        # Validate before mutating a running session.
        probe=HomeBatch();probe.set_layout(areas,graph)
        with self.lock:
            self.layout=(list(map(float,areas)),graph.tolist());self.reset()
            self.subtitle='新户型已生效：面积改变热容量，连通关系改变房间传热。两套控制器同时重新开始。'
            self.trace={'房间面积㎡':self.layout[0],'门连通矩阵':self.layout[1]}

    def manual(self,temperatures,brightness,eco):
        with self.lock:
            p=self.env.preference[0].clone();self.previous.append(p.numpy().copy())
            p[:4]=torch.tensor(temperatures)-22;p[4:8]=torch.tensor(brightness);p[8]=eco
            for e in (self.env,self.other):e.set_preference(p)
            self.subtitle='各房间温度、照明和省电目标已更新。'
            self.events.append((self.env.t,'手动调整偏好'))
            self.trace={'生效偏好':p.tolist()}

    def command(self,text):
        with self.lock:
            current=self.env.preference[0].numpy().copy();intent,source=self.parser.parse(text,current.tolist())
            if not intent:self.subtitle='请告诉我想调整哪个房间、温度或生活场景。';return
            if intent['intent']=='query':
                self.subtitle=' / '.join(f'{n} {float(self.env.temps[0,i]):.1f}℃' for i,n in enumerate(NAMES));return
            if intent['intent']=='undo':
                p=self.previous.pop() if self.previous else current
            else:
                self.previous.append(current);p=merge(current,intent,self.env.targets()[2][0].numpy())
            for e in (self.env,self.other):e.set_preference(p)
            self.trace={'输入':text,'解析路径':source,'意图':intent,'生效偏好':list(map(float,p))}
            self.subtitle={'sleep':'卧室进入睡眠目标，其余房间停止舒适供暖。','away':'已切换离家目标，关闭照明并保留低温保护。','movie':'客厅进入观影场景，照明目标已调暗。','home':'客厅准备迎接回家，策略将重新分配供暖。'}.get(intent.get('scene'),'目标已更新，下一控制步由策略重新决定设备动作。')
            self.events.append((self.env.t,text))
    def step(self,count=1):
        with self.lock:
            for _ in range(count):
                if self.env.t>=96:self.paused=True;break
                for e,controller,rows in ((self.env,self.policy,self.rows),(self.other,self.rule,self.baseline)):
                    _,_,_,info=e.step(controller.act(e));rows.append({k:v[0].tolist() for k,v in info.items()})
    def day(self):self.step(96-self.env.t)

def data_table(e,rows,title,reference=None):
    target,_,active=e.targets();now=rows[-1] if rows else None
    header=['房间','有人','当前 / 目标 ℃','规则温度 ℃','超温度带 ℃','供暖 kW','照明 %','累计用电 kWh']
    body=[]
    for i,name in enumerate(NAMES):
        temp=float(e.temps[0,i]);goal=float(target[0,i]);occupied=bool(active[0,i])
        violation=max(abs(temp-goal)-1,0) if occupied else 0
        energy=sum(([0,.8,2][r['hvac'][i]]+[0,.02,.06][r['lights'][i]])*.25 for r in rows)
        ref=f'{float(reference.temps[0,i]):.2f}' if reference is not None else '—'
        vals=[name,'是' if occupied else '否',f'{temp:.2f} / {goal:.1f}',ref,f'{violation:.2f}',f'{[0,.8,2][int(e.hvac[0,i])]:.1f}',str([0,50,100][int(e.lights[0,i])]),f'{energy:.3f}']
        body.append('<tr>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
    return ('<div class="data-panel"><div class="board-top"><b>'+html.escape(title)+f'</b><small>全屋 {e.t//4:02d}:{e.t%4*15:02d} · '+str(round(float(e.area.sum())))+'㎡</small></div><div class="table-scroll"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in header)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div></div>')


def snapshot(s):
    with s.lock:
        cost=sum(r['cost'] for r in s.rows);base=sum(r['cost'] for r in s.baseline)
        degree=sum(r['degree_hours'] for r in s.rows);bd=sum(r['degree_hours'] for r in s.baseline)
        over=sum(r['overload_kw']*.25 for r in s.rows)
        delta=base-cost
        cards=[('累计电费',f'¥ {cost:.2f}',f'规则 ¥{base:.2f} · '+('少花' if delta>=0 else '多花')+f' ¥{abs(delta):.2f}'),
            ('舒适偏离',f'{degree:.3f} °C·h',f'规则 {bd:.2f} °C·h · 越低越好'),('累计用电',f"{sum(r['energy_kwh'] for r in s.rows):.3f} kWh",f"规则 {sum(r['energy_kwh'] for r in s.baseline):.3f} kWh · 超容量 {over:.3f} kWh"),
            ('当前电价',f'¥ {float(s.env.prices[min(s.env.t,95)]):.1f}','元 / kWh · 15分钟一个控制步')]
        rate=f'费用变化 {(cost/base-1)*100:+.2f}%' if base>=.1 else '费用变化 —（基数不足0.1元）'
        cards[0]=(cards[0][0],cards[0][1],cards[0][2]+' · '+rate)
        kpi='<div class="kpis">'+''.join(f'<div class="kpi"><span>{a}</span><strong>{b}</strong><span>{c}</span></div>' for a,b,c in cards)+'</div>'
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig=make_subplots(rows=2,cols=2,subplot_titles=['总功率 · 实线模型 / 虚线规则','累计电费 · 越低越好','四房温度 · 实线模型 / 虚线规则 / 点线目标','累计舒适偏离 · 越低越好'],vertical_spacing=.17)
        hours=(np.arange(1,len(s.rows)+1)/4).tolist()
        for col,key,cumulative in [(1,'power_kw',False),(2,'cost',True)]:
            for rows,name,color,dash in [(s.rows,s.name,'#146c94','solid'),(s.baseline,s.baseline_name,'#db7135','dash')]:
                values=[r[key] for r in rows]
                fig.add_trace(go.Scatter(x=hours,y=np.cumsum(values).tolist() if cumulative else values,name=name,legendgroup=name,showlegend=col==1,line=dict(color=color,width=3,dash=dash)),row=1,col=col)
        fig.add_hline(y=4,line_dash='dot',line_color='#9a5260',row=1,col=1)
        for i,(room,color) in enumerate(zip(NAMES,['#146c94','#9966bd','#48834a','#bd7934'])):
            for rows,suffix,dash,width in [(s.rows,'模型','solid',2.5),(s.baseline,'规则','dash',1.8),(s.rows,'目标','dot',1)]:
                key='target' if suffix=='目标' else 'temps'
                fig.add_trace(go.Scatter(x=hours,y=[r[key][i] for r in rows],name=room+' '+suffix,legendgroup=room,showlegend=False,line=dict(color=color,dash=dash,width=width)),row=2,col=1)
        for rows,name,color,dash in [(s.rows,s.name,'#146c94','solid'),(s.baseline,s.baseline_name,'#db7135','dash')]:
            fig.add_trace(go.Scatter(x=hours,y=np.cumsum([r['degree_hours'] for r in rows]).tolist(),name=name,showlegend=False,line=dict(color=color,width=3,dash=dash)),row=2,col=2)
        for row,col,label in [(1,1,'kW'),(1,2,'元'),(2,1,'℃'),(2,2,'℃·h')]:
            fig.update_xaxes(range=[0,24],title_text='模拟小时',row=row,col=col)
            fig.update_yaxes(title_text=label,row=row,col=col)
        fig.update_yaxes(range=[0,9],row=1,col=1)
        fig.update_yaxes(range=[10,28],row=2,col=1)
        fig.update_layout(height=600,margin=dict(l=50,r=20,t=70,b=45),paper_bgcolor='#f1f0e9',plot_bgcolor='#fffef5',font=dict(family='sans-serif',color='#17352f'),legend=dict(orientation='h',y=1.14),uirevision='home-stable',transition=dict(duration=0))
        benefit=(f'本日对照：累计'+('少花' if delta>=0 else '多花')+f' {abs(delta):.2f} 元；有人房间温度偏离比规则'+('减少' if bd>=degree else '增加')+f' {abs(bd-degree):.2f} °C·h。') if s.rows else '两套住宅将接收相同天气、日程与需求。'
        event=f'<div class="eventline"><b>同步对照</b>　{benefit}</div><div class="eventline"> <b>当前需求</b>　{html.escape(s.subtitle)}</div>'+('<div class="caption">'+ ' → '.join(f'{t//4:02d}:{t%4*15:02d} {html.escape(text)}' for t,text in s.events[-6:])+'</div>')
        return kpi,comparison_table(s),data_table(s.env,s.rows,s.name,s.other),'<pre id="chart-payload">'+html.escape(fig.to_json())+'</pre>',event,s.trace

def comparison_table(s):
    metrics=[('累计电费','cost','元',1),('累计用电','energy_kwh','kWh',1),('舒适偏离','degree_hours','℃·h',1),('超容量用电','overload_kw','kWh',.25)]
    rows=[]
    for name,key,unit,scale in metrics:
        a=sum(r[key] for r in s.rows)*scale;b=sum(r[key] for r in s.baseline)*scale
        difference=a-b
        change=f'{difference/b:+.1%}' if b>=.1 else '—（基数不足）'
        tone='#26754b' if difference<0 else '#b34d2e' if difference>0 else '#52685f'
        rows.append(f'<tr><td>{name} ({unit})</td><td><b style="color:#146c94">{a:.3f}</b></td><td><b style="color:#db7135">{b:.3f}</b></td><td style="color:{tone}">{difference:+.3f}</td><td style="color:{tone}">{change}</td></tr>')
    return '<div class="data-panel"><div class="board-top"><b>模型与'+s.baseline_name+' · 同条件实时对照</b><small>全屋 '+f'{s.env.t//4:02d}:{s.env.t%4*15:02d}'+'</small></div><p>蓝色＝模型，橙色＝规则。差值＝模型−规则；以下指标越低越好。</p><div class="table-scroll"><table><thead><tr><th>指标</th><th>模型</th><th>'+s.baseline_name+'</th><th>绝对差值</th><th>变化比例</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></div>'


def heldout_table():
    path=ROOT/'assets/final_study.json'
    if not path.exists():return '<p>独立测试结果尚未生成。</p>'
    data=json.loads(path.read_text());rows=[]
    names={'default':'默认','eco':'省电','sleep':'睡眠','changes':'连续改口'}
    for scene,controllers in data['results'].items():
        m=controllers['mappo_lookahead']['summary'];b=controllers['coordinated_rule']['summary']
        values=[names[scene],f"{m['cost']:.3f} / {b['cost']:.3f}",f"{(m['cost']/b['cost']-1)*100:+.2f}%",f"{m['energy_kwh']:.3f} / {b['energy_kwh']:.3f}",f"{m['occupied_degree_hours']:.3f} / {b['occupied_degree_hours']:.3f}",f"{m['occupied_violation_rate']:.1%} / {b['occupied_violation_rate']:.1%}",f"{m['overload_kwh']:.4f}"]
        rows.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    headers=['场景','电费：前瞻 / 规则 元','电费变化','耗电：前瞻 / 规则 kWh','偏离：前瞻 / 规则 °C·h','不达标比例：前瞻 / 规则','超容量 kWh']
    return '<div class="data-panel"><b>独立测试量化结果 · MAPPO + 8小时前瞻</b><p>每场景20个新测试日（100020–100039）的均值。与上方当前运行统计分开；默认场景电费上升，不代表全面节费。仿真结果。</p><div class="table-scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p>温度偏离：有人房间超出目标±1℃后的度时；不达标比例另计。完整配对区间见技术报告第10节。</p></div>'


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--controller',choices=['raw','feasible','lookahead'],default='feasible');p.add_argument('--horizon',type=int,default=16);p.add_argument('--llm-url',help='Optional OpenAI-compatible intent endpoint');p.add_argument('--port',type=int,default=7861);p.add_argument('--host',default='127.0.0.1');p.add_argument('--scripted',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    if a.scripted:
        s=Session(a.checkpoint,llm_url=a.llm_url,controller=a.controller,horizon=a.horizon);out=[]
        for command in ['我到家了','客厅有点冷','省电模式','我要睡觉了']:
            s.command(command);s.step(12);out.append({'command':command,'trace':s.trace,'subtitle':s.subtitle})
        s.day();path=ROOT/'assets/v2';path.mkdir(parents=True,exist_ok=True)
        data={'commands':out,'final':s.rows[-1],'cost':sum(r['cost'] for r in s.rows),'baseline_cost':sum(r['cost'] for r in s.baseline)}
        (path/'scripted_trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print(json.dumps(data,ensure_ascii=False));return
    import gradio as gr
    with gr.Blocks(js=(ROOT/'demo/voice.js').read_text(),css=CSS,title='栖居 · 多房间协同控制',theme=gr.themes.Base(primary_hue='emerald',neutral_hue='stone')) as app:
        state=gr.State(None)
        gr.Markdown('控制模型：'+('BenchMARL MAPPO · '+{'raw':'独立执行','feasible':'联合功率约束','lookahead':'模型前瞻 + 联合约束'}[a.controller] if a.checkpoint else '未加载模型 · 当前为协调规则演示')+' ｜ 意图：'+('本地 LLM，失败自动规则兜底' if a.llm_url else '本地规则'))
        gr.HTML('<div id="masthead"><div class="eyebrow">MARL HOME / COOPERATIVE CONTROL LAB</div><h1>栖居 · 多房间调控量化对照</h1><p>仿真默认自动推进。用实时数据、四项曲线和同条件规则基线，检验控制效果。</p></div>')
        kpi=gr.HTML();
        with gr.Row():
            with gr.Column(scale=1,min_width=260):
                gr.Markdown('### 此刻，你需要什么？')
                text=gr.Textbox(label='生活需求',placeholder='客厅有点冷 / 我要睡觉了',lines=2,elem_id='command-input');send=gr.Button('更新需求',variant='primary',elem_id='command-send')
                with gr.Row():
                    mic=gr.Button('开启持续聆听');mute=gr.Button('关闭麦克风')
                gr.HTML('<div id="voice-status" role="status" aria-live="polite">麦克风未开启</div>')
                gr.Markdown('语音使用浏览器识别服务，可能联网；需 HTTPS 或 localhost。仅最终识别结果更新需求。')
                with gr.Row():home=gr.Button('回家',min_width=0);sleep=gr.Button('睡眠',min_width=0)
                with gr.Row():eco=gr.Button('省电',min_width=0);away=gr.Button('离家',min_width=0)
                play=gr.Checkbox(label='自动推进仿真',value=True)
                speed=gr.Slider(1,8,value=1,step=1,label='仿真倍速 · 默认自动运行')
                narrative=gr.Checkbox(label='演示日程：回家 → 调温 → 省电 → 睡眠',value=False)
                baseline_choice=gr.Radio(['协调规则','简单恒温器'],value='协调规则',label='对照基线 · 切换会重新开始')
                gr.Markdown('简单恒温器：独立温控、人走灯灭，不感知电价与共享容量。协调规则额外预热和分配功率；两者实现不人为调差。')
                day=gr.Button('快速查看当日总结');reset=gr.Button('重新开始 · 相同测试日')
                gr.Markdown('舒适温度带：目标 ±1℃。无人房间不计舒适偏离。\n\n共享功率为 **4 kW**。两套控制器收到相同的偏好更新。\n\n意图解析默认使用本地规则；语音识别由浏览器提供。')
                with gr.Accordion('查看意图与偏好',open=False):trace=gr.JSON(label='输入 → 目标（不直接控制设备）')
            with gr.Column(scale=4):
                selected=gr.HTML()
                with gr.Accordion('房间控制明细',open=False):baseline=gr.HTML()
                event=gr.HTML()
                gr.HTML('<div id="live-chart"><div class="chart-key"><b style="color:#146c94">━━ 模型</b>　<b style="color:#db7135">┄┄ 规则</b>　<span>温度图：同色同房间，实线模型、虚线规则、点线目标</span></div><canvas id="home-chart" role="img" aria-label="模型与规则的四项实时对照曲线"></canvas></div>')
                chart=gr.HTML(elem_id='chart-data')
        gr.HTML(heldout_table())
        with gr.Accordion('设计你的四区户型 · 修改后重新开始对照',open=False):
            gr.Markdown('四个固定功能区，每区 16–36㎡。面积影响热容量和外墙散热；门连通关系影响传热。支持该范围内的参数编辑，不代表任意建筑泛化。')
            with gr.Row():
                areas=[gr.Slider(16,36,value=[36,28,20,16][i],step=1,label=n+'面积 / ㎡') for i,n in enumerate(NAMES)]
            doors=gr.CheckboxGroup([f'{i}-{j}: {NAMES[i]} ↔ {NAMES[j]}' for i in range(4) for j in range(i+1,4)],label='开启传热门',value=['0-1: 客厅 ↔ 卧室','0-2: 客厅 ↔ 厨房','0-3: 客厅 ↔ 浴室'])
            layout_apply=gr.Button('应用户型 · 重启同日对照')
        with gr.Accordion('精细调控 · 温度、照明与省电偏好',open=False):
            with gr.Row():temps=[gr.Slider(19,25,value=22,step=.5,label=n+'目标温度 / ℃') for n in NAMES]
            with gr.Row():lights=[gr.Radio([0,.5,1],value=1,label=n+'亮度') for n in NAMES]
            eco_weight=gr.Slider(0,1,value=.4,step=.05,label='省电偏好 · 越高越重视费用')
            apply_preferences=gr.Button('更新四个房间的目标')
        outputs=[kpi,selected,baseline,chart,event,trace]
        mic.click(None,js='() => window.startHomeVoice?.()')
        mute.click(None,js='() => window.stopHomeVoice?.()')
        layout_apply.click(lambda s,*v:(s.configure(v[:4],v[4]),snapshot(s))[1],[state,*areas,doors],outputs)
        apply_preferences.click(lambda s,*v:(s.manual(v[:4],v[4:8],v[8]),snapshot(s))[1],[state,*temps,*lights,eco_weight],outputs)
        def change_baseline(s,value):
            s.baseline_kind='coordinated' if value=='协调规则' else 'thermostat'
            s.reset()
            return snapshot(s)
        baseline_choice.change(change_baseline,[state,baseline_choice],outputs,show_progress='hidden')
        def command(s,t):s.command(t);return snapshot(s)
        send.click(command,[state,text],outputs);text.submit(command,[state,text],outputs)
        for button,cmd in [(home,'我到家了'),(sleep,'我要睡觉了'),(eco,'省电模式'),(away,'出门了拜拜')]:button.click(lambda s,c=cmd:command(s,c),state,outputs)
        day.click(lambda s:(s.day(),snapshot(s))[1],state,outputs)
        reset.click(lambda s:(s.reset(),snapshot(s))[1],state,outputs)
        def tick(s,playing,rate,narrative):
            if s is None or not playing:return [gr.skip()]*len(outputs)
            if playing:
                for _ in range(int(rate)):
                    if narrative and s.env.t in {28,32,68,84}:s.command({28:'我到家了',32:'客厅有点冷',68:'省电模式',84:'我要睡觉了'}[s.env.t])
                    if s.env.t>=96:
                        p=s.env.preference.clone();s.reset()
                        for e in (s.env,s.other):e.set_preference(p)
                        s.subtitle='测试日循环播放，保留你刚才设置的需求。'
                    s.step()
            return snapshot(s)
        gr.Timer(1.5).tick(tick,[state,play,speed,narrative],outputs,show_progress="hidden",queue=False)
        def initialize():
            s=Session(a.checkpoint,llm_url=a.llm_url,controller=a.controller,horizon=a.horizon)
            return (s,*snapshot(s))
        app.load(initialize,None,[state,*outputs],show_progress='hidden')
    app.launch(server_name=a.host,server_port=a.port)
if __name__=='__main__':main()
