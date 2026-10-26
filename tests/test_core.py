import numpy as np, pandas as pd
from medvlm.data import patient_level_split
from medvlm.conformal import BinaryLabelConformal
from medvlm.grounding import pointing_game,energy_inside_box,threshold_iou

def test_patient_split_no_leakage():
    df=pd.DataFrame({"subject_id":np.repeat(np.arange(30),2)})
    s = patient_level_split(
    df,
    train_frac=.70,
    val_frac=.10,
    temp_calib_frac=.05,
    conformal_calib_frac=.05,
    seed=7
)
    sets=[set(x.subject_id) for x in s.values()]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            assert sets[i].isdisjoint(sets[j])

def test_conformal_direction():
    p=np.array([[.95],[.05],[.90],[.10],[.85],[.15]])
    y=np.array([[1],[0],[1],[0],[1],[0]])
    m=np.ones_like(y)
    c=BinaryLabelConformal(.2).fit(p,y,m)
    assert 1 in c.predict_sets(np.array([[.95]]))[0][0]
    assert 0 in c.predict_sets(np.array([[.05]]))[0][0]

def test_grounding():
    h=np.zeros((10,10)); h[4,4]=1
    b=[[3,3,3,3]]
    assert pointing_game(h,b)==1
    assert energy_inside_box(h,b)>.99
    assert 0<=threshold_iou(h,b)<=1
