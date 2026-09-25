// ツール(tool/sss-lab.html)の計算コードをそのまま使ってウォークフォワード検証を再計算する
// 使い方: node analysis/walkforward.js drop   （drop = 最終日が場中の途中データなら除外）
// 設定はツールの初期値: 点数75以上 / 1日最大2銘柄 / 縮小K=3000 / 減点 15,20,10,15
const fs=require('fs');
const html=fs.readFileSync(require('path').join(__dirname,'..','tool','sss-lab.html'),'utf8');
const a=html.indexOf('function prepare('), b=html.indexOf('// ---------- UI build');
const core=html.slice(a,b);
const vals={thr:75,perDay:2,tp:5,sl:2.5,hold:5,cost:0.1,shrinkK:3000};
const $=id=>({value:String(vals[id])});
let D=null,R=null,SECTORS=[],W=[25,25,20,15,15];const P=[15,20,10,15];
eval(core);
let data=JSON.parse(fs.readFileSync(require('path').join(__dirname,'..','sss_data.json'),'utf8'));
const DROP_LAST=process.argv[2]==='drop';
if(DROP_LAST){ data.dates.pop(); data.benchmark.c.pop(); for(const s of data.stocks) for(const k of 'ohlcv') s[k].pop(); }
prepare(data); simulate();
const thr=75,perDay=2,K=3000,T=R.T;
const years=[];let cur='';
for(let t=60;t<T;t++){const y=D.dates[t].slice(0,4); if(y!==cur){years.push({y,t0:t});cur=y;}}
years.forEach((o,i)=>o.t1=i+1<years.length?years[i+1].t0:T);
const manual=makeWS('manual',null); W=[20,20,20,20,20]; const equal=makeWS('manual',null);
const out={period:[D.dates[0],D.dates[T-1]],nStocks:D.stocks.length,years:[]};
for(const Y of years){
  if(Y.t0-60<240) continue;
  const L=learn(60,Y.t0,K); if(!L) continue;
  const base=baseline(Y.t0,Y.t1);
  const f=r=>({cnt:r.cnt,sum:r.sum,win:r.win,exp:r.exp});
  out.years.push({y:Y.y,from:D.dates[Y.t0],to:D.dates[Y.t1-1],trainTo:D.dates[Y.t0-1],N:L.N,base,
    baseCnt:(()=>{let c=0;for(let k=R.dayStart[Y.t0];k<R.dayStart[Y.t1];k++)if(!isNaN(R.ret[k]))c++;return c})(),
    manual:f(select(manual,P,thr,perDay,Y.t0,Y.t1)),equal:f(select(equal,P,thr,perDay,Y.t0,Y.t1)),
    global:f(select(makeWS('global',L),P,thr,perDay,Y.t0,Y.t1)),sector:f(select(makeWS('sector',L),P,thr,perDay,Y.t0,Y.t1)),
    gb:L.global.b,gw:L.global.w,sec:L.sec.map(s=>({name:s.name,n:s.n,ratio:s.ratio,w:s.w,b:s.b,fb:!!s.fallback}))});
}
// 全期間の基準と全データ学習
let bs=0,bc=0,bw=0; const t0=out.years[0]?years.find(y=>y.y===out.years[0].y).t0:60;
for(let k=R.dayStart[t0];k<R.dayStart[T];k++){const r=R.ret[k]; if(!isNaN(r)){bs+=r;bc++;if(r>0)bw++;}}
out.baseAll={exp:bs/bc,cnt:bc,win:bw/bc};
const LA=learn(60,T,K); out.full={N:LA.N,gb:LA.global.b,gw:LA.global.w,sec:LA.sec.map(s=>({name:s.name,n:s.n,ratio:s.ratio,w:s.w,b:s.b,fb:!!s.fallback}))};
// 各要素の単独の効き：要素スコア別の平均損益（全期間）
const bands=[];for(let i=0;i<5;i++){const m=new Map();for(let k=R.dayStart[60];k<R.dayStart[T];k++){const r=R.ret[k];if(isNaN(r))continue;const v=R.F[i][k].toFixed(2);const e=m.get(v)||[0,0];e[0]+=r;e[1]++;m.set(v,e);}bands.push([...m].sort().map(([v,[s,c]])=>({v,exp:s/c,cnt:c})));}
out.bands=bands;
fs.writeFileSync(require("path").join(__dirname,`wf_${DROP_LAST?'drop':'all'}.json`),JSON.stringify(out,null,1));
console.log(JSON.stringify(out.years.map(y=>({y:y.y,base:y.base,m:y.manual,e:y.equal,g:y.global,s:y.sector}))));
let s=0,s2=0,c=0;for(let k=0;k<R.n;k++){const r=R.ret[k];if(isNaN(r))continue;s+=r;s2+=r*r;c++;}
const sd=Math.sqrt(s2/c-(s/c)**2);console.log('SD',sd,'SE450',sd/Math.sqrt(450),'SE1700',sd/Math.sqrt(1700));
