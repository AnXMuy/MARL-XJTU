"""Architectural plan of a furnished four-zone residence; live simulator values."""
import html

def render(e,rows,title):
 target,_,active=e.targets();areas=e.area[0].tolist()
 zones=[(32,38,460,394),(510,38,358,394),(32,450,460,244),(510,450,358,244)]
 s=['<svg viewBox="0 0 900 780" width="100%" role="img" aria-label="完整住宅平面：客餐厅、卧室、厨房和浴室"><defs><pattern id="wood" width="32" height="16" patternUnits="userSpaceOnUse"><path d="M0 0H32V16H0Z" fill="#e6d9bf" stroke="#d9caac" stroke-width=".5"/></pattern><pattern id="tile" width="24" height="24" patternUnits="userSpaceOnUse"><rect width="24" height="24" fill="#dfe7e3" stroke="#c5d2cd" stroke-width=".7"/></pattern></defs><rect x="14" y="20" width="872" height="694" rx="5" fill="#faf8ee" stroke="#526157" stroke-width="14"/>']
 overlays=[]
 for i,(x,y,w,h) in enumerate(zones):
  temp=float(e.temps[0,i]);t=float(target[0,i]);occupied=bool(active[0,i]);level=int(e.hvac[0,i]);lamp=int(e.lights[0,i])
  color='#7fae96' if abs(temp-t)<=1 else '#d7966e'
  s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="url(#{"wood" if i<2 else "tile"})"/><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" opacity="{.14 if occupied else .035}"/>')
  overlays.append(f'<text x="{x+20}" y="{y+33}" font-size="20" fill="#233e36">{["客餐厅","主卧","厨房","浴室 / 家政"][i]} · {areas[i]:.0f}㎡</text>')
  overlays.append(f'<rect x="{x+18}" y="{y+45}" width="180" height="70" rx="8" fill="#fffdf2" opacity=".95"/><text x="{x+30}" y="{y+78}" font-size="30" fill="#234b3e">{temp:.1f}°</text><text x="{x+30}" y="{y+100}" font-size="13" fill="#536b60">{f"目标 {t-1:.0f}–{t+1:.0f}℃" if occupied else "无人 · 保留低温保护"}</text>')
  overlays.append(f'<rect x="{x+w-130}" y="{y+18}" width="108" height="28" rx="5" fill="#f7f8ef" stroke="#739388"/><text x="{x+w-120}" y="{y+37}" font-size="13" fill="#234b3e">供暖 {[0,.8,2][level]} kW</text>')
  if level:
   for j in range(3):overlays.append(f'<path class="heat-flow" d="M{x+w-110+j*25} {y+55}q-10 14 0 28t0 28" fill="none" stroke="#c77b52" stroke-width="3" opacity=".7"/>')
  overlays.append(f'<circle cx="{x+w-30}" cy="{y+h-28}" r="11" fill="{["#91a197","#e6c76d","#ffdf86"][lamp]}" stroke="#fff" stroke-width="3"/>')
  if occupied:overlays.append(f'<circle cx="{x+w-65}" cy="{y+h-35}" r="7" fill="#375c4e"/><path d="M{x+w-65} {y+h-26}v15m-8-9h16" stroke="#375c4e" stroke-width="4"/>')
 # Furnishings are spatially placed, not fake controllable devices.
 s+=['<g stroke="#8a8878" stroke-width="2" fill="#f5efe1"><rect x="65" y="204" width="118" height="176" rx="15" fill="#a8b6a7"/><rect x="74" y="214" width="96" height="46" rx="8"/><rect x="74" y="266" width="96" height="46" rx="8"/><rect x="74" y="318" width="96" height="46" rx="8"/><rect x="225" y="248" width="97" height="76" rx="15"/><rect x="410" y="215" width="24" height="135" rx="5" fill="#5b6d63"/><ellipse cx="340" cy="144" rx="57" ry="26"/><rect x="300" y="108" width="25" height="14" rx="4"/><rect x="350" y="108" width="25" height="14" rx="4"/><rect x="300" y="169" width="25" height="14" rx="4"/><rect x="350" y="169" width="25" height="14" rx="4"/></g>',
 '<g stroke="#919080" stroke-width="2"><rect x="570" y="190" width="210" height="200" rx="8" fill="#d5c7b1"/><rect x="579" y="229" width="192" height="150" rx="8" fill="#f7f2e8"/><rect x="590" y="201" width="72" height="40" rx="9" fill="#fffdf3"/><rect x="683" y="201" width="72" height="40" rx="9" fill="#fffdf3"/><rect x="811" y="204" width="34" height="160" fill="#c1baa6"/></g>',
 '<g fill="#f4f1e5" stroke="#8f9b8f" stroke-width="2"><path d="M58 480h410v48H106v140H58Z"/><rect x="140" y="484" width="70" height="37" rx="5" fill="#cbd6cf"/><circle cx="314" cy="501" r="13"/><circle cx="355" cy="501" r="13"/><rect x="371" y="583" width="88" height="83" rx="6" fill="#cbd5ca"/><rect x="543" y="486" width="128" height="73" rx="24"/><rect x="736" y="484" width="100" height="92" rx="7" fill="#cee0db"/><circle cx="786" cy="530" r="27"/><ellipse cx="582" cy="628" rx="25" ry="34"/><rect x="734" y="614" width="105" height="52" rx="10"/></g>',
 '<g stroke="#526157" stroke-width="12"><path d="M500 22v158m0 100v168m0 0v85m0 100v80M20 441h155m95 0h390m95 0h125"/></g>',
 '<g fill="none" stroke="#9eaa9a" stroke-width="2"><path d="M500 280h-100a100 100 0 0 1 100-100M175 441v95a95 95 0 0 0 95-95M660 441v95a95 95 0 0 0 95-95M500 533h90a90 90 0 0 1-90 90"/></g>',
 '<g stroke="#a0c1be" stroke-width="8"><path d="M95 21h240m245 0h210M18 90v190m864 30v105M110 712h240m260 0h170"/></g>',
 '<path d="M394 714h100" stroke="#f1f0e9" stroke-width="18"/><path d="M394 714v-92a92 92 0 0 1 92 92" fill="none" stroke="#788e7f" stroke-width="2"/><text x="410" y="755" font-size="17" fill="#52685d">入户玄关</text><text x="37" y="755" font-size="13" fill="#63786c">N ↑　完整住宅示意 · 家具仅作空间参照</text>']
 s+=overlays+['</svg>']
 power=rows[-1]['power_kw'] if rows else 0
 return f'<div class="residence"><div class="board-top"><b>{html.escape(title)}</b><small>全屋 {e.t//4:02d}:{e.t%4*15:02d} · {sum(areas):.0f}㎡</small></div>'+''.join(s)+f'<div class="constraint">全屋实时负荷 {power:.2f} / {e.config.capacity:.1f} kW</div></div>'
