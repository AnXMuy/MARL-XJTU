import json
import os
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import pydeck as pdk
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from traffic_marl.config import Config
from traffic_marl.env.sumo_env import SumoEnv
from traffic_marl.baselines.controllers import Controller
from traffic_marl.agents.checkpoint import load
from traffic_marl.utils.provenance import save_json
st.set_page_config(page_title='XJTU Traffic MARL',layout='wide')
st.title('城市交通信号协同控制')
st.sidebar.header('实验设置')
size=st.sidebar.selectbox('路网',['1×1 smoke','3×3 grid','4×4 grid'],index=1); n={'1×1 smoke':1,'3×3 grid':3,'4×4 grid':4}[size]
scenario=st.sidebar.selectbox('需求场景',['uniform','morning','evening','pulse']); rate=st.sidebar.slider('每入口需求 (veh/h)',0,1200,300,25)
duration=st.sidebar.slider('仿真时长 (s)',300,3600,600,100); warmup=min(60,duration//5); seed=st.sidebar.number_input('随机种子',0,2**31-1,10000)
run_root=Path(os.environ.get('TRAFFIC_RUNS',ROOT/'runs/demo')); ckpts={p.stem:p for p in (ROOT/'runs').glob('*/best.pt')}
options=['fixed','actuated','max_pressure','green_wave']+list(ckpts)
strategy=st.sidebar.selectbox('控制策略',options); replay=st.sidebar.checkbox('生成轨迹回放')
with st.sidebar.expander('入口流量调整'):
    custom={f'{s}{i}':st.slider(f'{s}{i}',0,1200,rate,25) for s in 'WENS' for i in range(n)}
if st.sidebar.button('运行 SUMO',type='primary'):
    cfg=Config(size=n,duration=duration,warmup=warmup,seed=int(seed),rate=rate,scenario=scenario,entrance_rates=custom)
    path=run_root/f'{cfg.digest()[:10]}_{strategy}'; model=meta=None
    if strategy in ckpts: model,meta=load(ckpts[strategy])
    try:
        with SumoEnv(cfg,path,replay) as env:
            obs,info=env.reset(); done=False; controller=Controller(strategy) if model is None else None
            while not done:
                if model is None: actions=controller.actions(env)
                elif meta['algorithm']=='idqn': actions=model.act(obs,info['action_mask'])
                else: actions=model.act(obs,info['action_mask'],True)[0]
                obs,_,_,done,info=env.step(actions)
            result=env.finish()
        result.update(policy=strategy,seed=int(seed),demand_sha256=env.manifest['demand_sha256'],checkpoint=meta)
        save_json(path/'summary.json',result); st.session_state.update(result=result,path=str(path))
    except Exception as exc: st.error(f'SUMO 运行失败：{type(exc).__name__}: {exc}')
if st.button('清除当前结果'): st.session_state.pop('result',None); st.session_state.pop('path',None)
if st.session_state.get('result'):
    result=st.session_state['result']; path=Path(st.session_state['path'])
    st.caption(f"策略：{result['policy']} · seed={result['seed']} · demand sha256={result['demand_sha256']}")
    cols=st.columns(5)
    for col,key,label in zip(cols,['travel_time_mean','waiting_mean','queue_mean','completion_rate','teleports'],['平均旅行时间 (s)','平均等待 (s)','平均队列','完成率','Teleport']):
        col.metric(label,'N/A' if result.get(key) is None else f"{result[key]:.2f}" if isinstance(result[key],float) else result[key])
    series=pd.read_csv(path/'timeseries.csv'); c1,c2=st.columns(2)
    with c1: st.subheader('全网队列'); st.line_chart(series.set_index('time')['queue'])
    with c2: st.subheader('累计完成车辆'); st.line_chart(series.set_index('time')['arrived'])
    if (path/'replay.json').exists():
        data=json.loads((path/'replay.json').read_text()); frame=st.slider('回放时间帧',0,len(data['frames'])-1,0); f=data['frames'][frame]
        project=lambda xy:[108.9+xy[0]/111000,34.2+xy[1]/111000]
        edges=[{'path':[project(xy) for xy in shape]} for shape in data['edges'].values()]
        vehicles=[{'position':project(v['xy']),'speed':v['speed']} for v in f['vehicles'].values()]
        layers=[pdk.Layer('PathLayer',edges,get_path='path',get_color=[85,98,110],width_scale=2,width_min_pixels=2),
                pdk.Layer('ScatterplotLayer',vehicles,get_position='position',get_fill_color='[240, 90, 45, 220]',get_radius=7,radius_min_pixels=4)]
        xs=[108.9+xy[0]/111000 for xy in data['positions'].values()]
        ys=[34.2+xy[1]/111000 for xy in data['positions'].values()]
        st.pydeck_chart(pdk.Deck(layers=layers,initial_view_state=pdk.ViewState(latitude=sum(ys)/len(ys),longitude=sum(xs)/len(xs),zoom=14,pitch=0)))
    for name in ('summary.json','timeseries.csv','vehicles.csv','executed_actions.csv','replay.json'):
        if (path/name).exists(): st.download_button(f'导出 {name}',(path/name).read_bytes(),file_name=name)
