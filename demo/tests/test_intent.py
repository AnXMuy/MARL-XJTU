import pytest
from demo.intent import IntentParser,merge
from demo.intent import default_preference

@pytest.mark.parametrize('text,kind,value',[
 ('我有点冷','delta_temp',2),('冻死了','delta_temp',2),('把家里弄暖和点','delta_temp',2),
 ('卧室太热了','delta_temp',-2),('我要睡觉了','scene','sleep'),('晚安','scene','sleep'),
 ('出门了拜拜','scene','away'),('我要看电影','scene','movie'),('我到家啦','scene','home'),
 ('电费太贵了省着点','w_eco',.9),('省电模式','w_eco',.9),('现在客厅多少度','intent','query')])
def test_rules(text,kind,value):
    result,source=IntentParser().parse(text,default_preference())
    assert result[kind]==value and source=='rule'

def test_invalid_llm_and_merge():
    class Broken:
        def complete(self,*args):return '{"intent":"set_preference","w_eco":100}'
    p=default_preference();i,s=IntentParser(Broken()).parse('省电模式',p)
    assert s=='rule' and merge(p,i,[1,0,0,0])[8]==pytest.approx(.9)
    i,s=IntentParser().parse('这是什么东西',p);assert i is None
