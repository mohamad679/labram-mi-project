"""Deterministic, zero-network SVG figures from committed and independently audited JSON.

Usage:
    python -m scripts.render_final_figures
    python -m scripts.render_final_figures --check

No training, EEG signal reading, Test inference, or performance estimation.
The source of every number is a committed real-run scientific review JSON.
"""
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

SOURCE_DIR = Path('results/validation')
FIGURE_DIR = Path('results/figures/final_snapshot')
NAVY = '#0b162c'
BLUE = '#294f83'
TEAL = '#147d82'
CYAN = '#74d1c4'
GOLD = '#d7a14f'
INK = '#162439'
MUTED = '#607186'
BORDER = '#dbe5ec'
BG = '#f5f8fb'
WHITE = '#ffffff'
FONT = 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif'


def tag(kind, **a):
    return '<' + kind + ''.join(f' {k.replace("_", "-")}="{escape(str(v), quote=True)}"' for k, v in a.items()) + '/>'


def txt(x, y, value, size=16, color=INK, weight=400, anchor='start', **extra):
    attrs = {'x': x, 'y': y, 'font_size': size, 'fill': color, 'font_weight': weight,
             'text_anchor': anchor, 'font_family': FONT, **extra}
    return '<text' + ''.join(f' {k.replace("_", "-")}="{escape(str(v), quote=True)}"' for k, v in attrs.items()) + '>' + escape(str(value)) + '</text>'


def panel(x,y,w,h,fill=WHITE,r=18):
    return tag('rect',x=x,y=y,width=w,height=h,fill=fill,stroke=BORDER,stroke_width=1,rx=r)


def svg(w,h,title,description,parts):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+str(w)+' '+str(h)+'" role="img" aria-labelledby="chart-title chart-description">\n'
      + '<title id="chart-title">'+escape(title)+'</title>\n'
      + '<desc id="chart-description">'+escape(description)+'</desc>\n'
      + tag('rect',x=0,y=0,width=w,height=h,fill=BG)
      + '\n'.join(parts)+'\n</svg>\n')


def load(root, name):
    return json.loads((root / SOURCE_DIR / name).read_text(encoding='utf-8'))


def metrics(root):
    p3=load(root, 'phase3_validation_rerun_seed42_20261013.json')
    p4=load(root, 'phase4_validation_seed42_20261013.json')
    p5=load(root, 'phase5_cbramod_validation_seed42_20261013.json')
    oof=load(root, 'phase55_trainonly_oof_audit_20261013.json')
    final=load(root, 'phase55b_one_time_validation_seed42_20261013.json')
    assert final['archive']['sha256']=='8e88e4463d1fa96856cb127f57e4a706042ff441095ccb1c1ad5d9109f635391'
    assert final['design']['test_waveform_inference'] is False
    assert final['design']['validation_evaluations_reported']==1
    assert oof['selection']['selected']=='eegnet_epoch_standardized'
    assert p3['reproducibility']['participants']=={'train':75,'validation':14,'test_untouched':15}
    assert p5['models']['full']['class_collapse_to_rest'] is True
    return p3,p4,p5,oof,final


def overview(p3,p4,p5,oof,final):
    rows=[('EEGNet · original',p3['methods']['eegnet']['macro_f1'],TEAL),
          ('EEGNet · standardized',final['results']['macro_f1'],BLUE),
          ('Riemann · tangent space',p3['methods']['riemann']['macro_f1'],MUTED),
          ('LaBraM · partial pretrained',p4['results']['partial']['macro_f1'],MUTED),
          ('CBraMod · frozen probe',p5['models']['probe']['macro_f1'],GOLD),
          ('CBraMod · full pretrained',p5['models']['full']['macro_f1'],GOLD)]
    p=[txt(58,65,'NEURODECODE',15,TEAL,750),txt(58,112,'The evidence, not the headline',35,NAVY,750),
       txt(58,146,'Fixed PhysioNet EEGMMIDB cohort  /  five-class imagined movement  /  research snapshot',16,MUTED),
       panel(46,173,1188,100),txt(74,211,'104',32,NAVY,750),txt(74,237,'participants',13,MUTED),
       txt(309,211,'18,096',32,NAVY,750),txt(309,237,'epochs',13,MUTED),
       txt(553,211,'75 / 14 / 15',29,NAVY,750),txt(553,237,'Train / Validation / untouched Test',13,MUTED),
       txt(1042,211,'0',32,TEAL,750),txt(1042,237,'Test inferences',13,MUTED),
       panel(46,291,1188,402),txt(75,340,'Development Validation',24,NAVY,750),
       txt(75,365,'Macro-F1  ·  one recorded training seed  ·  higher is better',14,MUTED)]
    scale=810/0.5
    for i,(label,val,color) in enumerate(rows):
        y=406+i*44
        p.extend([txt(78,y+18,label,15,INK,580),tag('rect',x=397,y=y,width=800,height=26,rx=6,fill='#eaf0f5'),
                  tag('rect',x=397,y=y,width=round(val*scale,1),height=26,rx=6,fill=color),
                  txt(1200,y+19,f'{val:.4f}',15,NAVY,700,anchor='end')])
    p.append(txt(74,675,'Descriptive comparison only: architectures, transfer adapters and training protocols differ. Test never opened.',13,MUTED))
    return svg(1280,712,'NeuroDecode: development Validation macro-F1 across verified models',
       'Original EEGNet 0.4191, standardized EEGNet 0.4125, Riemann 0.2568, LaBraM partial 0.2546, CBraMod probe 0.2027 and CBraMod full 0.1302. Different training protocols, single seed, Test untouched.',p)


def confusion(final):
    labels=['Feet','Both fists','Left fist','Rest','Right fist']
    mat=final['results']['confusion_matrix']; true=final['results']['true_class_counts']
    p=[txt(42,59,'CONFUSION MATRIX',14,TEAL,750),txt(42,103,'A closer look at the selected model',29,NAVY,750),
       txt(42,130,'EEGNet with per-epoch channel standardization · 2,436 Validation epochs',14,MUTED),
       panel(30,153,920,588),txt(520,201,'Predicted class',15,MUTED,650,anchor='middle'),
       txt(53,439,'True class',15,MUTED,650,transform='rotate(-90 53 439)')]
    x0,y0,unit=255,260,103
    for j,l in enumerate(labels):p.append(txt(x0+unit*(j+.5),235,l,13,INK,600,anchor='middle'))
    for i,(l,row) in enumerate(zip(labels,mat)):
        y=y0+unit*i
        p.append(txt(235,y+unit*.53,l,14,INK,600,anchor='end'))
        for j,v in enumerate(row):
            r=v/true[i]; t=(r**.75)
            # interpolate bright pale-blue to deep midnight cyan
            base=(236,247,249); deep=(13,94,109)
            color='#'+''.join(f'{round(b*(1-t)+d*t):02x}' for b,d in zip(base,deep))
            p.extend([tag('rect',x=x0+j*unit+2,y=y+2,width=unit-4,height=unit-4,rx=12,fill=color),
                      txt(x0+j*unit+unit/2,y+unit*.50,str(v),24,WHITE if r>.28 else NAVY,720,anchor='middle'),
                      txt(x0+j*unit+unit/2,y+unit*.75,f'{r*100:.1f}% of row',11,WHITE if r>.28 else MUTED,500,anchor='middle')])
    p.append(txt(255,779,'Rows are true labels; shade and percentage use within-class row support. Counts are not normalized.',13,MUTED))
    return svg(980,807,'EEGNet standardized five-class confusion matrix',
     'Validation confusion matrix: feet [82,64,38,95,40]; both fists [27,140,40,66,38]; left fist [24,57,145,55,43]; rest [164,134,71,714,93]; right fist [27,71,39,60,109].',p)


def class_f1(final):
    vals=final['results']['class_f1']; counts=final['results']['true_class_counts']
    labels=['Feet','Both fists','Left fist','Rest','Right fist']
    p=[txt(44,62,'CLASS-LEVEL DIAGNOSTICS',14,TEAL,750),txt(44,105,'Performance is not uniform',31,NAVY,750),
       txt(44,134,'Five-class F1 on the fixed 14-subject Validation cohort',15,MUTED),panel(33,159,1095,350)]
    for i,(name,val,n) in enumerate(zip(labels,vals,counts)):
        y=196+i*60
        p.extend([txt(67,y+18,name,17,INK,620),tag('rect',x=280,y=y,width=678,height=28,rx=7,fill='#eaf0f5'),
                  tag('rect',x=280,y=y,width=round(val*678,1),height=28,rx=7,fill=TEAL if i!=3 else BLUE),
                  txt(978,y+20,f'{val:.3f}',18,NAVY,700),txt(1096,y+20,f'n={n}',13,MUTED,anchor='end')])
    p.append(txt(46,548,'The rest class has 48.3% of Validation epochs. Class-wise F1 is essential alongside accuracy.',14,MUTED))
    return svg(1160,578,'Classwise F1 of normalized EEGNet on real Validation',
      'Feet 0.255, both fists 0.360, left hand 0.441, rest 0.659, right hand 0.347. Rest comprises 1,176 of 2,436 trials.',p)


def subjects(final):
    items=sorted(((int(k),float(v)) for k,v in final['results']['per_subject_macro_f1'].items()), key=lambda a:a[1])
    p=[txt(48,64,'SUBJECT HETEROGENEITY',14,TEAL,750),txt(48,110,'The average conceals the spread',31,NAVY,750),
       txt(48,139,'Per-subject five-class Macro-F1  ·  same selected checkpoint  ·  fixed 14 held-out Validation participants',15,MUTED),
       panel(34,160,1130,585)]
    x0=232; x1=1060; scale=(x1-x0)/0.8
    for ix,(sid,score) in enumerate(items):
        y=202+ix*34
        p.extend([txt(195,y+6,f'S{sid:03d}',15,INK,600,anchor='end'),
                  tag('line',x1=x0,y1=y,x2=x1,y2=y,stroke='#ebf0f4',stroke_width=1),
                  tag('line',x1=x0,y1=y,x2=x0+score*scale,y2=y,stroke=CYAN,stroke_width=7,stroke_linecap='round'),
                  tag('circle',cx=x0+score*scale,cy=y,r=7,fill=TEAL),
                  txt(x0+score*scale+15,y+5,f'{score:.3f}',13,NAVY,600)])
    for tick in [0,.2,.4,.6,.8]:p.append(txt(x0+tick*scale,706,f'{tick:.1f}',13,MUTED,500,anchor='middle'))
    p.append(txt(48,779,'Participant-level scores are descriptive; these 14 subjects are not an independent multi-seed replication.',14,MUTED))
    return svg(1200,811,'Per-subject variability across 14 unseen development Validation participants',
      'Per-subject macro-F1 ranges from 0.218 (participant 58) to 0.704 (participant 15).',p)


def ablation(oof,final,p3):
    rows=[('Original EEGNet',oof['models']['eegnet_raw']['macro_f1'],MUTED),
          ('EEGNet + epoch z-score',oof['models']['eegnet_epoch_standardized']['macro_f1'],TEAL),
          ('EEGNeX + epoch z-score',oof['models']['eegnex_epoch_standardized']['macro_f1'],BLUE),
          ('Four-band log-power + LR',oof['models']['bandpower']['macro_f1'],GOLD)]
    p=[txt(46,65,'TRAIN-ONLY OPTIMIZATION',14,TEAL,750),txt(46,105,'What changed, and what held up?',31,NAVY,750),
       txt(46,135,'3-fold subject-grouped OOF; 75 original Train subjects · 13,050 predictions per candidate',14,MUTED),
       panel(34,153,1132,278),txt(62,194,'TRAIN-ONLY OOF MACRO-F1',14,NAVY,650)]
    for i,(label,val,color) in enumerate(rows):
        y=216+i*48
        p.extend([txt(62,y+18,label,15,INK,520),tag('rect',x=374,y=y,width=650,height=25,rx=6,fill='#eaf0f5'),
                  tag('rect',x=374,y=y,width=round(val*650/0.5,1),height=25,rx=6,fill=color),
                  txt(1090,y+19,f'{val:.4f}',16,NAVY,700,anchor='end')])
    p.extend([panel(34,452,1132,165),txt(65,495,'TRANSLATION TO HISTORICAL VALIDATION',14,NAVY,700),
             txt(65,533,f"EEGNet original:  {p3['methods']['eegnet']['macro_f1']:.4f}",23,BLUE,700),
             txt(624,533,f"EEGNet standardized:  {final['results']['macro_f1']:.4f}",23,TEAL,700),
             txt(65,571,'The Train-only +0.0700 OOF gain did not replicate as higher historical Validation macro-F1.',14,MUTED),
             txt(65,599,'Different training protocols; not a controlled Validation ablation. No Test results.',13,MUTED)])
    return svg(1200,654,'Train-only screening and one-time historical Validation outcome',
      'Three-fold Train-only macro-F1: EEGNet raw .3212, EEGNet standardized .3912, EEGNeX standardized .3697, bandpower .2544. Historical Validation original EEGNet .4191, normalized EEGNet .4125; no overall improvement reproduced.',p)


def study_design():
    p=[txt(48,62,'FROZEN SCIENTIFIC CONTRACT',14,TEAL,750),txt(48,107,'Disjoint by participant, not by epoch',30,NAVY,750),
       txt(48,136,'Original 104-subject PhysioNet motor-imagery cohort · 18,096 epochs · 64 EEG channels',15,MUTED)]
    dims=[(45,178,348,'TRAIN','75 participants','13,050 epochs',TEAL),
          (426,178,348,'DEVELOPMENT VALIDATION','14 participants','2,436 epochs',BLUE),
          (808,178,348,'FINAL TEST · LOCKED','15 participants','2,610 epochs',GOLD)]
    for x,y,w,label,n,epochs,c in dims:
        p.extend([panel(x,y,w,210),tag('rect',x=x,y=y,width=w,height=7,rx=4,fill=c),
                  txt(x+24,y+50,label,14,c,750),txt(x+24,y+101,n,24,NAVY,750),
                  txt(x+24,y+139,epochs,16,MUTED,500)])
    p.extend([txt(65,433,'Used for model fitting and Train-only model selection',14,MUTED),
              txt(427,433,'Used for recorded development comparisons',14,MUTED),
              txt(825,433,'Never evaluated in this study',14,MUTED),
              txt(48,491,'NO HELD-OUT TEST SCORE  ·  NO MULTI-SEED CLAIM  ·  NO CLINICAL VALIDATION',15,TEAL,750)])
    return svg(1200,524,'Subject-disjoint frozen data split and Test embargo',
      '104 persons, 18,096 EEG epochs; 75 subjects Train 13,050 epochs; 14 subjects development Validation 2,436 epochs; 15 Test subjects 2,610 epochs never evaluated.',p)


def generate(root:Path):
    p3,p4,p5,oof,final=metrics(root)
    return {
      'benchmark-overview.svg':overview(p3,p4,p5,oof,final),
      'validation-confusion-matrix.svg':confusion(final),
      'validation-class-f1.svg':class_f1(final),
      'validation-subject-spread.svg':subjects(final),
      'train-only-vs-validation.svg':ablation(oof,final,p3),
      'frozen-study-design.svg':study_design(),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    out=generate(args.root)
    folder=args.root/FIGURE_DIR
    if not args.check:folder.mkdir(parents=True,exist_ok=True)
    for name,content in out.items():
        path=folder/name
        if args.check:
            if not path.is_file() or path.read_text(encoding='utf-8')!=content:
                raise SystemExit(f'Figure out of sync with versioned evidence: {path}')
        else:
            path.write_text(content,encoding='utf-8')
    print(f'{len(out)} deterministic evidence SVG figures '+('verified' if args.check else 'generated'))


if __name__=='__main__':main()
