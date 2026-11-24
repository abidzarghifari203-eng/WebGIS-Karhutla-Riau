"""Rapikan data WebGIS: EPSG:32647 -> lon/lat, gabung atribut, kecilkan ukuran.
Pakai: python build_data.py <grid.geojson> <kab.geojson> <kerentanan_fr.csv> <out_dir> [csv_tambahan ...]
CSV tambahan (penduduk 'pop', bahaya bhy_01..10, DC dc_01..10) digabung lewat kolom grid_id (satu baris per sel)."""
import json, math, sys
import numpy as np, pandas as pd
A=6378137.0; E2=(1/298.257223563)*(2-1/298.257223563); EP2=E2/(1-E2); K0=0.9996
def utm47n_inv(x,y):
    x-=500000.0; mu=(y/K0)/(A*(1-E2/4-3*E2**2/64-5*E2**3/256))
    e1=(1-math.sqrt(1-E2))/(1+math.sqrt(1-E2))
    p=(mu+(3*e1/2-27*e1**3/32)*math.sin(2*mu)+(21*e1**2/16-55*e1**4/32)*math.sin(4*mu)
       +(151*e1**3/96)*math.sin(6*mu)+(1097*e1**4/512)*math.sin(8*mu))
    s,c,t=math.sin(p),math.cos(p),math.tan(p); n1=A/math.sqrt(1-E2*s*s)
    t1=t*t; c1=EP2*c*c; r1=A*(1-E2)/(1-E2*s*s)**1.5; d=x/(n1*K0)
    lat=p-(n1*t/r1)*(d*d/2-(5+3*t1+10*c1-4*c1*c1-9*EP2)*d**4/24+(61+90*t1+298*c1+45*t1*t1-252*EP2-3*c1*c1)*d**6/720)
    lon=(d-(1+2*t1+c1)*d**3/6+(5-2*c1+28*t1-3*c1*c1+8*EP2+24*t1*t1)*d**5/120)/c
    return math.degrees(lon)+(47*6-183), math.degrees(lat)
def clean(ring,nd):
    out=[]
    for pt in ring:
        q=[round(pt[0],nd),round(pt[1],nd)]
        if not out or q!=out[-1]: out.append(q)
    return out
def dp(pts,tol):
    P=np.asarray(pts); n=len(P); keep=np.zeros(n,bool); keep[0]=keep[-1]=True; st=[(0,n-1)]
    while st:
        i,j=st.pop()
        if j<=i+1: continue
        a,b=P[i],P[j]; seg=P[i+1:j]; d=b-a; L=math.hypot(*d)
        dist=np.hypot(*(seg-a).T) if L==0 else np.abs(d[0]*(seg[:,1]-a[1])-d[1]*(seg[:,0]-a[0]))/L
        k=int(dist.argmax())
        if dist[k]>tol: m=i+1+k; keep[m]=True; st+=[(i,m),(m,j)]
    return P[keep].tolist()
def polys(geom): return geom['coordinates'] if geom['type']=='MultiPolygon' else [geom['coordinates']]
def num(v,nd):
    return None if v is None or (isinstance(v,float) and math.isnan(v)) else (int(v) if float(v).is_integer() and nd==0 else round(float(v),nd))
def main(grid,kab,kerent,out,*extra):
    from matplotlib.path import Path
    g=json.load(open(grid))['features']
    kr=pd.read_csv(kerent).set_index('feat_idx')
    ex=[pd.read_csv(f).set_index('grid_id') for f in extra]
    kp={f['properties']['KAB_KOTA']:polys(f['geometry']) for f in json.load(open(kab))['features']}
    def inside(name,pts):
        m=np.zeros(len(pts),bool)
        for poly in kp[name]:
            o=Path([q[:2] for q in poly[0]]).contains_points(pts)
            for h in poly[1:]: o&=~Path([q[:2] for q in h]).contains_points(pts)
            m|=o
        return m
    idx={}
    for i,f in enumerate(g): idx.setdefault(int(f['properties']['grid_id']),[]).append(i)
    feats=[]
    for gid,ids in idx.items():           # SATU baris per sel (baris ganda = salinan geometri yang sama)
        f=g[ids[0]]; p=f['properties']; r=kr.loc[ids[0]]; kabname=p['KAB_KOTA']
        if len(ids)>1:                    # sel menyentuh >1 kabupaten: pilih yang areanya paling luas (sampling 15x15 titik)
            xs=np.linspace(p['left'],p['right'],17)[1:-1]; ys=np.linspace(p['bottom'],p['top'],17)[1:-1]
            pts=np.array([utm47n_inv(x,y) for x in xs for y in ys])
            cand=[g[j]['properties']['KAB_KOTA'] for j in ids]; cnt=[int(inside(c,pts).sum()) for c in cand]
            if max(cnt)>0: kabname=cand[int(np.argmax(cnt))]
        pr={'g':gid,'kab':kabname}
        for c in kr.columns:
            if c.startswith('fr_'): pr[c]=num(r[c],3)
        pr['indeks']=num(r['indeks_fr'],3); pr['rentan']=int(r['kelas_kerentanan'])
        for d in ex:
            if gid in d.index:
                for c in d.columns: pr[c]=num(d.loc[gid,c],3)
        mp=[]
        for poly in polys(f['geometry']):
            rings=[dp(clean([utm47n_inv(x,y) for x,y in ring],4),0.0002) for ring in poly]
            rings=[r_ for r_ in rings if len(r_)>=4]
            if rings: mp.append(rings)
        if not mp:                        # serpihan sangat kecil: simpan tanpa pembulatan agar baris tetap ada
            mp=[[[list(utm47n_inv(x,y)) for x,y in ring] for ring in poly] for poly in polys(f['geometry'])]
        feats.append({'type':'Feature','properties':pr,'geometry':{'type':'MultiPolygon','coordinates':mp}})
    json.dump({'type':'FeatureCollection','features':feats},open(out+'/grid.geojson','w'),separators=(',',':'))
    kf=[]
    for name,pl in kp.items():
        mp=[]
        for poly in pl:
            rings=[dp(clean([pt[:2] for pt in ring],4),0.001) for ring in poly]
            rings=[r_ for r_ in rings if len(r_)>=4]
            if rings: mp.append(rings)
        kf.append({'type':'Feature','properties':{'kab':name},'geometry':{'type':'MultiPolygon','coordinates':mp}})
    json.dump({'type':'FeatureCollection','features':kf},open(out+'/kab.geojson','w'),separators=(',',':'))
if __name__=='__main__': main(*sys.argv[1:])
