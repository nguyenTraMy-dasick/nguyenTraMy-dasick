#!/usr/bin/env python3
"""Static ASCII-art eye (left eye, glancing up) for the Features section.
Usage: python generate_eye.py [path/to/monospace-bold.ttf]  ->  assets/eye.png (420x420)
"""
import numpy as np, random, math
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
R=random.Random(7); rng=np.random.default_rng(7)
PW,PH=420,420; CW,CH=4,7; COLS,ROWS=PW//CW,PH//CH; S=3
FW,FH=COLS*CW*S,ROWS*CH*S
yy,xx=np.mgrid[0:FH,0:FW].astype(np.float32)/S
W_,A_,B_=150,60,38
def lids(u,side):
    o=np.clip(u*side,0,None)                       # 0..1 toward outer corner
    yu=-A_*np.clip(1-u**2,0,None)**0.8 - 16*o**2   # upper lid, outer corner flicks up
    yl= B_*np.clip(1-(u*1.04)**2,0,None)**1.1 - 9*o**2
    return yu,yl
def eye_field(cx,cy,side):
    u=(xx-cx)/W_; yu,yl=lids(u,side); yu=yu+cy; yl=yl+cy
    inside=(np.abs(u)<1)&(yy>yu)&(yy<yl)
    icx,icy=cx+10,cy-20; r=np.hypot(xx-icx,yy-icy)
    iris=np.where(r<38,0.55+0.4*(r/38)**3,0); iris=np.maximum(iris,(r<15)*1.0)
    for hx,hy,hr in [(12,-2,10),(-14,14,4.5)]: iris=np.where(np.hypot(xx-icx-hx,yy-icy-hy)<hr,0,iris)
    g=np.where(inside,iris,0)
    d_up=yy-yu
    lid=np.exp(-((d_up+6)/13)**2)*(np.abs(u)<1.04)
    above=np.clip(-d_up,0,None); o=np.clip(u*side,0,1)
    shade=np.exp(-above/(30+40*o))*(above>0)*(np.abs(u)<0.95+0.2*o)*1.3
    d_lo=yy-yl
    low=np.exp(-(d_lo/4.0)**2)*0.7*(np.abs(u)<0.98)
    smudge=np.exp(-(((xx-(cx+side*W_*0.8))/40)**2+((yy-(cy+B_*0.35))/22)**2))*0.8
    hook=np.exp(-((np.hypot(xx-(cx-side*W_*0.92),yy-(cy+6))-26)/5)**2)*((xx-(cx-side*W_*0.92))*side<6)*(yy>cy-14)*(yy<cy+34)*0.85
    return np.maximum.reduce([g,lid,shade,low,smudge,hook])
CX,CY=222,262
F=eye_field(CX,CY,-1)
lash=Image.new("L",(FW,FH),0); d=ImageDraw.Draw(lash)
STROKES=[]
def spike(x,y,ang,L,w,val):
    STROKES.append((x,y,ang,L,val)); return
    dx,dy=math.cos(ang),math.sin(ang); nx,ny=-dy,dx
    d.polygon([((x+nx*w)*S,(y+ny*w)*S),((x-nx*w)*S,(y-ny*w)*S),((x+dx*L)*S,(y+dy*L)*S)],fill=val)
def lashes(cx,cy,side):
    for i in range(34):
        u=-1.0+2.0*i/33+R.uniform(-.015,.015); o=max(0,u*side)
        yu,_=lids(np.array(u),side); x=cx+u*W_; y=cy+float(yu)
        ang=-math.pi/2+u*1.05+side*0.15*o+R.uniform(-.12,.12)
        L=R.uniform(28,52)*(1+0.9*o)*(0.6 if u*side<-0.7 else 1)
        spike(x,y,ang,L,2.2,255)
    for i in range(24):
        u=-0.92+1.84*i/23+R.uniform(-.03,.03); o=max(0,u*side)
        _,yl=lids(np.array(u),side); x=cx+u*W_; y=cy+float(yl)
        ang=math.pi/2+u*0.6+R.uniform(-.2,.2); L=R.uniform(10,24)*(1+0.9*o)
        if R.random()<0.8: spike(x,y,ang,L,1.6,220)
lashes(CX,CY,-1)
F=np.maximum(F,np.asarray(lash,np.float32)/255)
noise=ndimage.gaussian_filter(rng.random((FH,FW)),1.5); noise=(noise-noise.min())/(noise.max()-noise.min())
F=np.clip(F*(0.6+0.7*noise),0,1)
cell=F.reshape(ROWS,CH*S,COLS,CW*S)
cell=0.5*cell.mean((1,3))+0.5*cell.max((1,3))*0.9
RAMP=" .'`:-=+*#%@"
over={}
for (x,y,ang,L,val) in STROKES:
    dx,dy=math.cos(ang),math.sin(ang)
    up=-dy if dy<0 else dy
    slope=dx/max(up,1e-3)*(1 if dy<0 else -1)
    ch='|' if abs(slope)<0.38 else ('/' if slope>0 else '\\')
    if abs(slope)>2.6: ch='_' if dy>0 else '~'
    n=int(L/3)+1
    for i in range(n):
        t=i/max(n-1,1); px=x+dx*L*t; py=y+dy*L*t
        r,c=int(py/CH),int(px/CW)
        if 0<=r<ROWS and 0<=c<COLS: over[(r,c)]=(ch,(1-0.55*t)*val/255)

import os, sys
_here=os.path.dirname(os.path.abspath(__file__))
_c=[p for p in (sys.argv[1] if len(sys.argv)>1 else None,"/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf","C:/Windows/Fonts/consolab.ttf","/System/Library/Fonts/Menlo.ttc") if p and os.path.exists(p)]
if not _c: sys.exit("No monospace font found; pass one as the first argument")
mono=ImageFont.truetype(_c[0],7)
out=Image.new("RGB",(PW,PH),(0,0,0)); od=ImageDraw.Draw(out)
for r in range(ROWS):
    for c in range(COLS):
        v=cell[r,c]
        if (r,c) in over:
            ch,b=over[(r,c)]; L=int(255*min(1,b*(0.8+0.25*c/COLS))); od.text((c*CW,r*CH),ch,font=mono,fill=(L,L,L)); continue
        if v<0.10: continue
        k=int(np.clip(v*(len(RAMP)-1)+R.uniform(-.7,.7),1,len(RAMP)-1))
        L=int(255*min(1,(0.45+0.8*v)*(0.75+0.28*c/COLS)))
        od.text((c*CW,r*CH),RAMP[k],font=mono,fill=(L,L,L))
a=np.asarray(out.convert("L")); ys=np.nonzero(a.max(1)>0)[0]; xs=np.nonzero(a.max(0)>0)[0]

os.makedirs(os.path.join(_here,"assets"),exist_ok=True)
out.save(os.path.join(_here,"assets","eye.png")); print("wrote assets/eye.png")
