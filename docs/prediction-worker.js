"use strict";

// Fast macro prediction and the complete 421-second sandbox must obey the
// same terrain geometry.  Load the shared router in both Worker and Node test
// environments; the compact implementation below remains only as a fallback
// for old cached pages.
let sharedTerrainRouter = globalThis.RMUCTerrainRouter || null;
if (!sharedTerrainRouter && typeof importScripts === "function") {
  importScripts("./terrain-router.js?v=7");
  sharedTerrainRouter = globalThis.RMUCTerrainRouter || null;
}
if (!sharedTerrainRouter && typeof module === "object" && module.exports) {
  sharedTerrainRouter = require("./terrain-router.js");
}

// Browser-only trajectory inference.  Keeping this in a Worker means replay,
// seeking and canvas drawing never wait for the neural-network calculation.
const MOBILE_TYPES = ["英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"];
const GROUND_TYPES = new Set(["英雄", "工程", "步兵3", "步兵4", "哨兵"]);
const STRUCTURE_TYPES = ["基地", "前哨站"];
const FIELD_WIDTH = 28;
const FIELD_HEIGHT = 15;
const R = {id:0,type:1,side:2,hp:3,max:4,x:5,y:6,yaw:7,a17:8,a42:9,coins:10,vulnerable:11};
const MODEL_URL = "./data/models/trajectory_transformer.json?v=4";
const NAVIGATION_URL = "./data/models/terrain_navigation.json?v=24";

let modelPromise = null;

function emit(message) {
  if (typeof self !== "undefined" && self.postMessage) self.postMessage(message);
}

async function loadModel() {
  if (modelPromise) return modelPromise;
  modelPromise = (async () => {
    emit({type:"status", status:"loading", text:"模型与地形图加载中"});
    const manifestUrl = new URL(MODEL_URL, self.location.href);
    const navigationUrl = new URL(NAVIGATION_URL, self.location.href);
    const [manifestResponse,navigationResponse] = await Promise.all([
      fetch(manifestUrl, {cache:"force-cache"}),
      fetch(navigationUrl, {cache:"force-cache"}),
    ]);
    if (!manifestResponse.ok) throw new Error(`模型清单 HTTP ${manifestResponse.status}`);
    if (!navigationResponse.ok) throw new Error(`地形拓扑 HTTP ${navigationResponse.status}`);
    const manifest = await manifestResponse.json();
    const navigation = await navigationResponse.json();
    const weightsUrl = new URL(manifest.weights, manifestUrl);
    weightsUrl.search = manifestUrl.search;
    const weightsResponse = await fetch(weightsUrl, {cache:"force-cache"});
    if (!weightsResponse.ok) throw new Error(`模型权重 HTTP ${weightsResponse.status}`);
    const buffer = await weightsResponse.arrayBuffer();
    const weights = new Float32Array(buffer);
    const expected = manifest.tensors.reduce((sum,item) => sum + item.length, 0);
    if (weights.length !== expected) throw new Error(`模型权重长度错误：${weights.length}/${expected}`);
    const tensors = new Map();
    for (const item of manifest.tensors) {
      tensors.set(item.name, weights.subarray(item.offset, item.offset + item.length));
    }
    const model = {
      manifest,
      navigation,
      tensors,
      mean: Float32Array.from(manifest.feature_mean),
      std: Float32Array.from(manifest.feature_std),
      targetX: manifest.feature_names.indexOf("target.x"),
      targetY: manifest.feature_names.indexOf("target.y"),
      targetVx3: manifest.feature_names.indexOf("target.vx_3_norm_per_s"),
      targetVy3: manifest.feature_names.indexOf("target.vy_3_norm_per_s"),
    };
    emit({type:"status", status:"ready", text:"Temporal Transformer 预测已开启"});
    return model;
  })();
  return modelPromise;
}

function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function otherSide(side) { return side === "红" ? "蓝" : "红"; }
function validPosition(row) {
  if (!row || row[R.x] == null || row[R.y] == null) return false;
  const x=Number(row[R.x]), y=Number(row[R.y]);
  return x>=0 && x<=FIELD_WIDTH && y>=0 && y<=FIELD_HEIGHT && !(x===0 && y===0);
}
function canonicalXY(row, targetSide) {
  let x=Number(row[R.x]), y=Number(row[R.y]);
  if (targetSide === "蓝") { x=FIELD_WIDTH-x; y=FIELD_HEIGHT-y; }
  return [x/FIELD_WIDTH,y/FIELD_HEIGHT];
}
function hpRatio(row) {
  if (!row || !row[R.max]) return 0;
  return clamp(Number(row[R.hp]||0)/Number(row[R.max]),0,1.5);
}
function ammoTotal(row) { return row ? Number(row[R.a17]||0)+Number(row[R.a42]||0) : 0; }
function indexFrame(rows) {
  return new Map((rows||[]).map(row=>[`${row[R.side]}:${row[R.type]}`,row]));
}
function rowAt(frame,side,type) { return frame.get(`${side}:${type}`); }
function normalizedCoins(frame,side) {
  for (const type of [...MOBILE_TYPES,...STRUCTURE_TYPES]) {
    const row=rowAt(frame,side,type);
    if (row && row[R.coins]!=null) return Number(row[R.coins])/2000;
  }
  return 0;
}

function buildFeatures(history, second, targetSide, targetType, duration, schools=null, manifest=null) {
  const frames = new Map();
  for (const offset of [0,1,3,5]) frames.set(offset,indexFrame(history[String(offset)]||history[offset]));
  const enemy=otherSide(targetSide), values=[];
  const current=rowAt(frames.get(0),targetSide,targetType);
  if (!validPosition(current) || Number(current[R.hp]||0)<=0) return null;

  for (const offset of [0,1,3,5]) {
    const frame=frames.get(offset);
    for (const side of [targetSide,enemy]) {
      for (const type of MOBILE_TYPES) {
        const row=rowAt(frame,side,type), present=validPosition(row);
        if (present) {
          const [x,y]=canonicalXY(row,targetSide);
          values.push(x,y,hpRatio(row),1);
        } else values.push(0,0,hpRatio(row),0);
      }
    }
    for (const side of [targetSide,enemy]) {
      for (const type of STRUCTURE_TYPES) values.push(hpRatio(rowAt(frame,side,type)));
    }
    values.push(normalizedCoins(frame,targetSide),normalizedCoins(frame,enemy));
  }

  const [currentX,currentY]=canonicalXY(current,targetSide);
  const yaw=current[R.yaw];
  let headingSin=0,headingCos=0,headingPresent=0;
  if (yaw!=null) {
    const canonicalYaw=Number(yaw)+(targetSide==="蓝"?180:0), radians=canonicalYaw*Math.PI/180;
    headingSin=Math.sin(radians); headingCos=Math.cos(radians); headingPresent=1;
  }
  values.push(currentX,currentY,hpRatio(current),headingSin,headingCos,headingPresent,Number(Boolean(current[R.vulnerable])));

  for (const offset of [1,3,5]) {
    const previous=rowAt(frames.get(offset),targetSide,targetType);
    if (!validPosition(previous)) return null;
    const [previousX,previousY]=canonicalXY(previous,targetSide);
    values.push(
      (currentX-previousX)/offset,
      (currentY-previousY)/offset,
      (ammoTotal(current)-ammoTotal(previous))/offset/50,
    );
  }
  const safeDuration=Math.max(1,Number(duration)||420);
  values.push(second/safeDuration,Math.max(0,safeDuration-second)/safeDuration);
  for (const type of MOBILE_TYPES) values.push(Number(type===targetType));
  const schoolNames=manifest?.school_names||[];
  if (schoolNames.length) {
    const currentHp=hpRatio(current);
    for (const offset of [1,3,5]) {
      const previous=rowAt(frames.get(offset),targetSide,targetType);
      values.push(Math.max(0,hpRatio(previous)-currentHp));
    }
    const ownSchool=String(schools?.[targetSide]||"");
    const opponentSchool=String(schools?.[enemy]||"");
    for (const school of schoolNames) values.push(Number(school===ownSchool));
    for (const school of schoolNames) values.push(Number(school===opponentSchool));
  }
  const expected=Number(manifest?.input_dim||240);
  return values.length===expected ? Float32Array.from(values) : null;
}

function linear(input,weight,bias,outSize,inSize) {
  const output=new Float32Array(outSize);
  for (let row=0;row<outSize;row++) {
    let sum=bias[row], start=row*inSize;
    for (let column=0;column<inSize;column++) sum+=weight[start+column]*input[column];
    output[row]=sum;
  }
  return output;
}
function erf(x) {
  const sign=x<0?-1:1, ax=Math.abs(x), t=1/(1+0.3275911*ax);
  const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-ax*ax);
  return sign*y;
}
function gelu(values) {
  const out=new Float32Array(values.length), root2=Math.SQRT2;
  for (let i=0;i<values.length;i++) out[i]=0.5*values[i]*(1+erf(values[i]/root2));
  return out;
}
function layerNorm(values,gamma,beta,epsilon) {
  let mean=0;
  for (const value of values) mean+=value;
  mean/=values.length;
  let variance=0;
  for (const value of values) variance+=(value-mean)*(value-mean);
  variance/=values.length;
  const inverse=1/Math.sqrt(variance+epsilon), out=new Float32Array(values.length);
  for (let i=0;i<values.length;i++) out[i]=(values[i]-mean)*inverse*gamma[i]+beta[i];
  return out;
}
function forwardMlp(model,normalized) {
  const t=name=>model.tensors.get(name), m=model.manifest;
  let hidden=linear(normalized,t("backbone.0.weight"),t("backbone.0.bias"),256,m.input_dim);
  hidden=layerNorm(gelu(hidden),t("backbone.2.weight"),t("backbone.2.bias"),m.layer_norm_epsilon);
  hidden=linear(hidden,t("backbone.4.weight"),t("backbone.4.bias"),256,256);
  hidden=layerNorm(gelu(hidden),t("backbone.6.weight"),t("backbone.6.bias"),m.layer_norm_epsilon);
  hidden=linear(hidden,t("backbone.8.weight"),t("backbone.8.bias"),128,256);
  hidden=layerNorm(gelu(hidden),t("backbone.10.weight"),t("backbone.10.bias"),m.layer_norm_epsilon);
  return linear(hidden,t("head.weight"),t("head.bias"),m.horizons.length*2,128);
}

function addResidual(left,right) {
  const output=new Float32Array(left.length);
  for(let i=0;i<left.length;i++)output[i]=left[i]+right[i];
  return output;
}

function transformerAttention(tokens,model,prefix) {
  const t=name=>model.tensors.get(name),m=model.manifest,d=m.d_model,heads=m.nhead,headDim=d/heads,count=tokens.length;
  const normalized=tokens.map(token=>layerNorm(token,t(`${prefix}.norm1.weight`),t(`${prefix}.norm1.bias`),m.layer_norm_epsilon));
  const projected=normalized.map(token=>linear(token,t(`${prefix}.self_attn.in_proj_weight`),t(`${prefix}.self_attn.in_proj_bias`),d*3,d));
  const attended=Array.from({length:count},()=>new Float32Array(d));
  const scale=1/Math.sqrt(headDim);
  for(let head=0;head<heads;head++){
    const base=head*headDim;
    for(let query=0;query<count;query++){
      const scores=new Float64Array(count);let maximum=-Infinity;
      for(let key=0;key<count;key++){
        let score=0;
        for(let j=0;j<headDim;j++)score+=projected[query][base+j]*projected[key][d+base+j];
        score*=scale;scores[key]=score;if(score>maximum)maximum=score;
      }
      let denominator=0;
      for(let key=0;key<count;key++){scores[key]=Math.exp(scores[key]-maximum);denominator+=scores[key];}
      for(let j=0;j<headDim;j++){
        let value=0;
        for(let key=0;key<count;key++)value+=scores[key]/denominator*projected[key][d*2+base+j];
        attended[query][base+j]=value;
      }
    }
  }
  return attended.map((value,index)=>addResidual(
    tokens[index],linear(value,t(`${prefix}.self_attn.out_proj.weight`),t(`${prefix}.self_attn.out_proj.bias`),d,d),
  ));
}

function transformerLayer(tokens,model,index) {
  const t=name=>model.tensors.get(name),m=model.manifest,d=m.d_model,ff=m.dim_feedforward,prefix=`encoder.layers.${index}`;
  const attended=transformerAttention(tokens,model,prefix);
  return attended.map(token=>{
    const normalized=layerNorm(token,t(`${prefix}.norm2.weight`),t(`${prefix}.norm2.bias`),m.layer_norm_epsilon);
    const hidden=gelu(linear(normalized,t(`${prefix}.linear1.weight`),t(`${prefix}.linear1.bias`),ff,d));
    return addResidual(token,linear(hidden,t(`${prefix}.linear2.weight`),t(`${prefix}.linear2.bias`),d,ff));
  });
}

function forwardTransformer(model,normalized) {
  const t=name=>model.tensors.get(name),m=model.manifest,d=m.d_model,count=m.history_token_count,width=m.history_token_width;
  const tokens=[];
  for(let index=0;index<count;index++){
    const input=normalized.subarray(index*width,(index+1)*width);
    tokens.push(linear(input,t("history_projection.weight"),t("history_projection.bias"),d,width));
  }
  tokens.push(linear(
    normalized.subarray(count*width),t("context_projection.weight"),t("context_projection.bias"),d,m.context_dim,
  ));
  const position=t("position_embedding");
  for(let token=0;token<tokens.length;token++)for(let i=0;i<d;i++)tokens[token][i]+=position[token*d+i];
  let encoded=tokens;
  for(let layer=0;layer<m.num_layers;layer++)encoded=transformerLayer(encoded,model,layer);
  const target=layerNorm(encoded.at(-1),t("norm.weight"),t("norm.bias"),m.layer_norm_epsilon);
  return linear(target,t("head.weight"),t("head.bias"),m.horizons.length*2,d);
}

function forward(model,features) {
  const m=model.manifest,normalized=new Float32Array(m.input_dim);
  for(let i=0;i<m.input_dim;i++)normalized[i]=(features[i]-model.mean[i])/model.std[i];
  return m.model_kind==="temporal_battlefield_transformer"
    ? forwardTransformer(model,normalized) : forwardMlp(model,normalized);
}

function worldPoint(canonicalX,canonicalY,side) {
  let x=clamp(canonicalX,0,1)*FIELD_WIDTH, y=clamp(canonicalY,0,1)*FIELD_HEIGHT;
  if (side==="蓝") { x=FIELD_WIDTH-x; y=FIELD_HEIGHT-y; }
  return [x,y];
}
function destinationZone(x,y,perspectiveSide) {
  const ownBase=perspectiveSide==="红"?[2.2,7.5]:[25.8,7.5];
  const ownOutpost=perspectiveSide==="红"?[6.1,7.5]:[21.9,7.5];
  const enemyOutpost=perspectiveSide==="红"?[21.9,7.5]:[6.1,7.5];
  const enemyBase=perspectiveSide==="红"?[25.8,7.5]:[2.2,7.5];
  const distance=(point)=>Math.hypot(x-point[0],y-point[1]);
  if (distance(ownBase)<=2) return "己方基地";
  if (distance(ownOutpost)<=1.45) return "己方前哨";
  if (distance(enemyOutpost)<=1.45) return "敌方前哨";
  if (distance(enemyBase)<=2) return "敌方基地";
  const canonicalX=perspectiveSide==="红"?x:FIELD_WIDTH-x;
  const canonicalY=perspectiveSide==="红"?y:FIELD_HEIGHT-y;
  const depth=canonicalX<7?"己方后场":canonicalX<12?"己方前场":canonicalX<=16?"中央":canonicalX<=21?"敌方前场":"敌方后场";
  const lane=canonicalY>=11?"上路":canonicalY<=4?"下路":"中路";
  return `${depth}·${lane}`;
}
function confidence(model,horizon,moving) {
  const group=moving?"moving":"all";
  const metric=model.manifest.reliability[String(horizon)]?.[group]?.zone_accuracy;
  return Number.isFinite(metric)?metric:0;
}

function pointInPolygon(point,polygon) {
  let inside=false;
  for (let i=0,j=polygon.length-1;i<polygon.length;j=i++) {
    const [xi,yi]=polygon[i], [xj,yj]=polygon[j];
    if ((yi>point[1])!==(yj>point[1]) && point[0]<(xj-xi)*(point[1]-yi)/(yj-yi)+xi) inside=!inside;
  }
  return inside;
}
function orientation(a,b,c) { return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]); }
function onSegment(a,b,p) {
  return Math.abs(orientation(a,b,p))<1e-8&&p[0]>=Math.min(a[0],b[0])-1e-8&&p[0]<=Math.max(a[0],b[0])+1e-8&&p[1]>=Math.min(a[1],b[1])-1e-8&&p[1]<=Math.max(a[1],b[1])+1e-8;
}
function segmentsIntersect(a,b,c,d) {
  const o1=orientation(a,b,c),o2=orientation(a,b,d),o3=orientation(c,d,a),o4=orientation(c,d,b);
  if (Math.abs(o1)<1e-8&&onSegment(a,b,c)) return true;
  if (Math.abs(o2)<1e-8&&onSegment(a,b,d)) return true;
  if (Math.abs(o3)<1e-8&&onSegment(c,d,a)) return true;
  if (Math.abs(o4)<1e-8&&onSegment(c,d,b)) return true;
  return (o1>0)!==(o2>0)&&(o3>0)!==(o4>0);
}
function segmentHitsPolygon(start,end,polygon) {
  if (pointInPolygon(start,polygon)||pointInPolygon(end,polygon)) return true;
  for (let i=0;i<polygon.length;i++) if (segmentsIntersect(start,end,polygon[i],polygon[(i+1)%polygon.length])) return true;
  return false;
}
function distance(a,b) { return Math.hypot(a[0]-b[0],a[1]-b[1]); }
function pushPoint(route,point) {
  const clean=[clamp(point[0],0,FIELD_WIDTH),clamp(point[1],0,FIELD_HEIGHT)];
  if (!route.length||distance(route.at(-1),clean)>1e-4) route.push(clean);
}
function routeLength(route) {
  let total=0;
  for (let i=1;i<route.length;i++) total+=distance(route[i-1],route[i]);
  return total;
}
function regionEntries(navigation) {
  return [
    {id:"central_highland",polygon:navigation.regions.central_highland},
    {id:"blue_trapezoid_highland",polygon:navigation.regions.blue_trapezoid_highland},
    {id:"red_trapezoid_highland",polygon:navigation.regions.red_trapezoid_highland},
  ];
}
function staticObstaclePolygons(navigation) {
  return (navigation.static_obstacles||[])
    .filter(obstacle=>obstacle.blocks_movement!==false&&obstacle.polygon?.length>=3)
    .map(obstacle=>obstacle.polygon);
}
function regionAt(navigation,point) {
  return regionEntries(navigation).find(region=>pointInPolygon(point,region.polygon))||null;
}
function roleCapabilities(navigation,school,role) {
  const data=navigation.teams?.[school]?.[role];
  return {
    abilities:new Set(data?.abilities||[]),
    reverseFly:Boolean(data?.reverse_fly_ramp?.allowed),
    motionProfiles:data?.terrain_motion_profiles||navigation.routing?.default_terrain_motion_profiles||{},
  };
}
function polygonCentroid(polygon) {
  const sum=polygon.reduce((value,point)=>[value[0]+point[0],value[1]+point[1]],[0,0]);
  return [sum[0]/polygon.length,sum[1]/polygon.length];
}
function directionVector(direction) {
  return {negative_x:[-1,0],positive_x:[1,0],negative_y:[0,-1],positive_y:[0,1]}[direction]||null;
}
function gatePair(gate,region) {
  const center=gate.center, centroid=polygonCentroid(region.polygon);
  const vector=directionVector(gate.high_direction),length=Math.max(.001,distance(center,centroid));
  const ux=vector?.[0]??(centroid[0]-center[0])/length,uy=vector?.[1]??(centroid[1]-center[1])/length;
  let inside=center, outside=center;
  for (let step=.15;step<=3;step+=.15) {
    const candidate=[center[0]+ux*step,center[1]+uy*step];
    if (pointInPolygon(candidate,region.polygon)) { inside=candidate; break; }
  }
  for (let step=.15;step<=3;step+=.15) {
    const candidate=[center[0]-ux*step,center[1]-uy*step];
    if (!pointInPolygon(candidate,region.polygon)) { outside=candidate; break; }
  }
  return {gate,inside,outside};
}
function gatesForRegion(navigation,region) {
  if (region.id==="central_highland") {
    return navigation.gates.filter(gate=>gate.category==="central_highland_step").map(gate=>gatePair(gate,region));
  }
  const side=region.id.startsWith("blue")?"blue":"red";
  return navigation.gates.filter(gate=>gate.side===side&&["slope_43","trapezoid_highland_step"].includes(gate.category)).map(gate=>gatePair(gate,region));
}
function gateLabel(gate,direction) {
  const prefix=gate.side==="blue"?"B":"R";
  const name={central_highland_step:"中央高地台阶",slope_43:"43°坡",trapezoid_highland_step:"梯形高地台阶"}[gate.category]||gate.category;
  return `${prefix}${gate.gate_index}${direction}${name}`;
}
function gateRoutingBlocker(gate) {return gate.routing_blocker_polygon||gate.polygon;}
function ascendingThroughGate(gate,start,end) {
  const vector=directionVector(gate.high_direction);if(!vector)return null;
  return (end[0]-start[0])*vector[0]+(end[1]-start[1])*vector[1]>0;
}
function binaryHeapPush(heap,item) {
  heap.push(item); let index=heap.length-1;
  while (index>0) { const parent=(index-1)>>1; if (heap[parent][0]<=item[0]) break; heap[index]=heap[parent]; index=parent; }
  heap[index]=item;
}
function binaryHeapPop(heap) {
  if (!heap.length) return null;
  const root=heap[0], tail=heap.pop();
  if (heap.length) {
    let index=0;
    while (true) {
      let child=index*2+1;
      if (child>=heap.length) break;
      if (child+1<heap.length&&heap[child+1][0]<heap[child][0]) child+=1;
      if (heap[child][0]>=tail[0]) break;
      heap[index]=heap[child]; index=child;
    }
    heap[index]=tail;
  }
  return root;
}
function routeAvoiding(navigation,start,end,extraPolygons=[]) {
  const obstacles=[...regionEntries(navigation).map(region=>region.polygon),...staticObstaclePolygons(navigation),...extraPolygons];
  if (!obstacles.some(polygon=>segmentHitsPolygon(start,end,polygon))) return [start,end];
  const step=Number(navigation.routing.grid_m||.35), columns=Math.round(FIELD_WIDTH/step)+1, rows=Math.round(FIELD_HEIGHT/step)+1, total=columns*rows;
  const nodePoint=index=>{const x=index%columns,y=Math.floor(index/columns);return [Math.min(FIELD_WIDTH,x*step),Math.min(FIELD_HEIGHT,y*step)];};
  const blocked=index=>obstacles.some(polygon=>pointInPolygon(nodePoint(index),polygon));
  const nearestIndex=point=>{
    const baseX=clamp(Math.round(point[0]/step),0,columns-1),baseY=clamp(Math.round(point[1]/step),0,rows-1);
    const startsBlocked=obstacles.some(polygon=>pointInPolygon(point,polygon));
    for (let radius=0;radius<8;radius++) for (let dy=-radius;dy<=radius;dy++) for (let dx=-radius;dx<=radius;dx++) {
      if (Math.max(Math.abs(dx),Math.abs(dy))!==radius) continue;
      const x=baseX+dx,y=baseY+dy;if(x<0||x>=columns||y<0||y>=rows)continue;
      const index=y*columns+x;if(!blocked(index)&&(startsBlocked||!obstacles.some(polygon=>segmentHitsPolygon(point,nodePoint(index),polygon))))return index;
    }
    return -1;
  };
  const startIndex=nearestIndex(start),endIndex=nearestIndex(end);
  if(startIndex<0||endIndex<0)return[start];
  const scores=new Float64Array(total);scores.fill(Infinity);scores[startIndex]=0;
  const previous=new Int32Array(total);previous.fill(-1);
  const closed=new Uint8Array(total),heap=[];binaryHeapPush(heap,[distance(nodePoint(startIndex),end),startIndex]);
  const moves=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
  while(heap.length){const item=binaryHeapPop(heap),index=item[1];if(closed[index])continue;if(index===endIndex)break;closed[index]=1;
    const x=index%columns,y=Math.floor(index/columns),from=nodePoint(index);
    for(const [dx,dy] of moves){const nx=x+dx,ny=y+dy;if(nx<0||nx>=columns||ny<0||ny>=rows)continue;const next=ny*columns+nx;if(closed[next]||blocked(next))continue;
      const to=nodePoint(next);if(obstacles.some(polygon=>segmentHitsPolygon(from,to,polygon)))continue;
      const score=scores[index]+Math.hypot(dx,dy)*step;if(score>=scores[next])continue;scores[next]=score;previous[next]=index;binaryHeapPush(heap,[score+distance(to,end),next]);
    }
  }
  if(!Number.isFinite(scores[endIndex]))return[start];
  const reversed=[];for(let at=endIndex;at>=0;at=previous[at]){reversed.push(nodePoint(at));if(at===startIndex)break;}
  const safeEnd=obstacles.some(polygon=>pointInPolygon(end,polygon))?nodePoint(endIndex):end;
  const raw=[start,...reversed.reverse(),safeEnd],simplified=[raw[0]];
  let anchor=0;
  while(anchor<raw.length-1){let next=anchor+1;for(let candidate=raw.length-1;candidate>anchor+1;candidate--){if(!obstacles.some(polygon=>segmentHitsPolygon(raw[anchor],raw[candidate],polygon))){next=candidate;break;}}pushPoint(simplified,raw[next]);anchor=next;}
  return simplified;
}
function boundaryCrossing(start,end,polygon) {
  const dx=end[0]-start[0],dy=end[1]-start[1],hits=[];
  for(let i=0;i<polygon.length;i++){
    const a=polygon[i],b=polygon[(i+1)%polygon.length],ex=b[0]-a[0],ey=b[1]-a[1],den=dx*ey-dy*ex;
    if(Math.abs(den)<1e-9)continue;const t=((a[0]-start[0])*ey-(a[1]-start[1])*ex)/den,u=((a[0]-start[0])*dy-(a[1]-start[1])*dx)/den;
    if(t>=0&&t<=1&&u>=0&&u<=1)hits.push({t,point:[start[0]+dx*t,start[1]+dy*t]});
  }
  hits.sort((a,b)=>a.t-b.t);return hits[0]?.point||null;
}
function crossingPair(start,end,polygon) {
  const crossing=boundaryCrossing(start,end,polygon);if(!crossing)return null;
  const length=Math.max(.001,distance(start,end)),ux=(end[0]-start[0])/length,uy=(end[1]-start[1])/length;
  return {outside:[crossing[0]-ux*.24,crossing[1]-uy*.24],inside:[crossing[0]+ux*.24,crossing[1]+uy*.24]};
}
function bestGate(candidates,start,end,abilities,ascending) {
  const allowed=candidates.filter(candidate=>!ascending||abilities.has(candidate.gate.category));
  allowed.sort((a,b)=>distance(start,a.outside)+distance(a.inside,end)-distance(start,b.outside)-distance(b.inside,end));
  return allowed[0]||null;
}
function straightRoadStepSegment(gate,start,end) {
  const blocker=gateRoutingBlocker(gate),xs=gate.polygon.map(point=>point[0]),ys=blocker.map(point=>point[1]);
  const minX=Math.min(...xs)+.1,maxX=Math.max(...xs)-.1,minY=Math.min(...ys),maxY=Math.max(...ys),crossingY=(minY+maxY)/2;
  const ratio=Math.abs(end[1]-start[1])>1e-6?(crossingY-start[1])/(end[1]-start[1]):.5;
  const crossingX=clamp(start[0]+(end[0]-start[0])*ratio,minX,maxX),positiveY=end[1]>start[1];
  return [start,[crossingX,positiveY?minY-.08:maxY+.08],[crossingX,positiveY?maxY+.08:minY-.08],end];
}
function applyDirectionalGates(navigation,route,capabilities,passages) {
  const watched=navigation.gates.filter(gate=>["fly_ramp","road_step","rough_road","road_tunnel","highland_tunnel"].includes(gate.category));
  const output=[route[0]],encounteredBlockers=[];
  for(let i=1;i<route.length;i++){
    const start=route[i-1],end=route[i],blockers=[];let alignedRoadStep=null;
    for(const gate of watched){if(!segmentHitsPolygon(start,end,gateRoutingBlocker(gate)))continue;
      if(gate.category==="fly_ramp"){
        const forward=gate.side==="blue"?end[0]<start[0]:end[0]>start[0],hasBase=capabilities.abilities.has("fly_ramp"),allowed=hasBase&&(!forward?capabilities.reverseFly:true);
        if(!allowed)blockers.push(gateRoutingBlocker(gate));else passages.push(`${gate.side==="blue"?"B":"R"}1${forward?"飞坡":"反飞坡"}`);
      }else if(gate.category==="road_step"){
        const ascending=ascendingThroughGate(gate,start,end);
        if(ascending!==false&&!capabilities.abilities.has("road_step"))blockers.push(gateRoutingBlocker(gate));
        else {
          passages.push(`${gate.side==="blue"?"B":"R"}${gate.gate_index}${ascending===false?"下":"上"}公路台阶`);
          const roadStepProfile=capabilities.motionProfiles?.road_step,direction=ascending===false?"down":"up";
          const directionProfile=roadStepProfile?.directions?.[direction]||roadStepProfile;
          if(directionProfile?.route_alignment_enabled)alignedRoadStep=gate;
        }
      }else if(!capabilities.abilities.has(gate.category))blockers.push(gateRoutingBlocker(gate));
      else {
        const name={rough_road:"起伏路",road_tunnel:"公路隧道",highland_tunnel:"高地隧道"}[gate.category]||gate.category;
        passages.push(`${gate.side==="blue"?"B":"R"}${gate.gate_index}${name}`);
      }
    }
    for(const polygon of blockers)if(!encounteredBlockers.includes(polygon))encounteredBlockers.push(polygon);
    const segment=blockers.length?routeAvoiding(navigation,start,end,blockers):alignedRoadStep?straightRoadStepSegment(alignedRoadStep,start,end):[start,end];
    if(segment.length>1)for(const point of segment.slice(1))pushPoint(output,point);
  }
  return {route:output,blockers:encounteredBlockers};
}
function enforceSymmetricBlockers(navigation,route,blockers){
  if(!blockers.length||route.length<2)return route;
  const output=[route[0]];
  for(let i=1;i<route.length;i++){
    const start=output.at(-1),end=route[i];
    if(!blockers.some(polygon=>segmentHitsPolygon(start,end,polygon))){pushPoint(output,end);continue;}
    const detour=routeAvoiding(navigation,start,end,blockers);
    if(detour.length>1)for(const point of detour.slice(1))pushPoint(output,point);
  }
  return output;
}
function terrainRoute(navigation,start,target,school,role) {
  if (sharedTerrainRouter?.terrainRoute) {
    const planned=sharedTerrainRouter.terrainRoute(navigation,start,target,school,role);
    return {
      route:planned.route,target:planned.target,passages:planned.passages,
      corrected:planned.corrected,
    };
  }
  const capabilities=roleCapabilities(navigation,school,role),startRegion=regionAt(navigation,start),targetRegion=regionAt(navigation,target),route=[start],passages=[];
  const symmetricBlockers=navigation.gates.filter(gate=>(["rough_road","road_tunnel","highland_tunnel"].includes(gate.category)&&!capabilities.abilities.has(gate.category))||(gate.category==="fly_ramp"&&!capabilities.abilities.has("fly_ramp"))).map(gateRoutingBlocker);
  const avoid=(from,to)=>routeAvoiding(navigation,from,to,symmetricBlockers);
  let current=start,adjustedTarget=target,corrected=false;
  if(startRegion&&targetRegion&&startRegion.id===targetRegion.id){pushPoint(route,target);return{route,target,passages,corrected};}
  if(startRegion){
    const exit=bestGate(gatesForRegion(navigation,startRegion),start,target,capabilities.abilities,false);
    if(exit){pushPoint(route,exit.inside);pushPoint(route,exit.outside);current=exit.outside;passages.push(gateLabel(exit.gate,"下"));}
  }
  if(targetRegion){
    let entry=bestGate(gatesForRegion(navigation,targetRegion),current,target,capabilities.abilities,true);
    const jumpPair=targetRegion.id==="central_highland"&&capabilities.abilities.has("central_highland_400mm_jump")
      ?crossingPair(current,target,targetRegion.polygon):null;
    const jumpCentre=jumpPair?[(jumpPair.outside[0]+jumpPair.inside[0])/2,(jumpPair.outside[1]+jumpPair.inside[1])/2]:null;
    const crossesDesignedStep=jumpCentre&&navigation.gates.some(gate=>gate.category==="central_highland_step"&&pointInPolygon(jumpCentre,gate.polygon));
    const entryScore=entry?distance(current,entry.outside)+distance(entry.inside,target)+.35:Infinity;
    const jumpScore=jumpPair&&!crossesDesignedStep?distance(current,jumpPair.outside)+distance(jumpPair.inside,target)+.65:Infinity;
    if(jumpScore<entryScore){for(const point of avoid(current,jumpPair.outside).slice(1))pushPoint(route,point);pushPoint(route,jumpPair.inside);pushPoint(route,target);passages.push("400mm跳跃上高地");}
    else if(entry){for(const point of avoid(current,entry.outside).slice(1))pushPoint(route,point);pushPoint(route,entry.inside);pushPoint(route,target);passages.push(gateLabel(entry.gate,"上"));}
    else if(jumpPair&&!crossesDesignedStep){
      for(const point of avoid(current,jumpPair.outside).slice(1))pushPoint(route,point);pushPoint(route,jumpPair.inside);pushPoint(route,target);passages.push("400mm跳跃上高地");
    }else{
      const pair=crossingPair(current,target,targetRegion.polygon);
      adjustedTarget=pair?.outside||current;for(const point of avoid(current,adjustedTarget).slice(1))pushPoint(route,point);corrected=true;passages.push("能力不足·停在地形外");
    }
  }else{
    for(const point of avoid(current,target).slice(1))pushPoint(route,point);
  }
  const allBlockers=[...symmetricBlockers];let legal=route;
  for(let iteration=0;iteration<6;iteration++){
    const directional=applyDirectionalGates(navigation,legal,capabilities,passages);let added=false;
    for(const polygon of directional.blockers)if(!allBlockers.includes(polygon)){allBlockers.push(polygon);added=true;}
    legal=enforceSymmetricBlockers(navigation,directional.route,allBlockers);
    if(!added&&!legal.slice(1).some((point,index)=>allBlockers.some(polygon=>segmentHitsPolygon(legal[index],point,polygon))))break;
  }
  const finalTarget=legal.at(-1)||adjustedTarget;
  if(distance(finalTarget,adjustedTarget)>.15){corrected=true;passages.push("能力不足·停在地形外");}
  return{route:legal,target:finalTarget,passages:[...new Set(passages)],corrected};
}
function pointAlongRoute(route,fraction) {
  const total=routeLength(route);if(total<1e-6)return route.at(-1);
  let remaining=clamp(fraction,0,1)*total;
  for(let i=1;i<route.length;i++){const length=distance(route[i-1],route[i]);if(remaining<=length){const ratio=length?remaining/length:0;return[route[i-1][0]+(route[i][0]-route[i-1][0])*ratio,route[i-1][1]+(route[i][1]-route[i-1][1])*ratio];}remaining-=length;}
  return route.at(-1);
}

async function predict(message) {
  const started=performance.now(), model=await loadModel(), predictions=[];
  const current=indexFrame(message.history["0"]||message.history[0]);
  for (const side of ["红","蓝"]) {
    for (const role of GROUND_TYPES) {
      const robot=rowAt(current,side,role);
      if (!validPosition(robot) || Number(robot[R.hp]||0)<=0) continue;
      const features=buildFeatures(
        message.history,message.second,side,role,message.duration,message.schools,model.manifest,
      );
      if (!features) continue;
      const residuals=forward(model,features), rawPoints=[];
      for (let i=0;i<model.manifest.horizons.length;i++) {
        const horizon=model.manifest.horizons[i];
        if (message.second+horizon>message.duration) continue;
        const [x,y]=worldPoint(features[model.targetX]+residuals[i*2],features[model.targetY]+residuals[i*2+1],side);
        rawPoints.push({horizon,x,y});
      }
      if (!rawPoints.length) continue;
      const vx=features[model.targetVx3]*FIELD_WIDTH, vy=features[model.targetVy3]*FIELD_HEIGHT;
      const moving=Math.hypot(vx,vy)>=0.15;
      const primary=rawPoints.find(point=>point.horizon===10)||rawPoints.at(-1);
      const start=[Number(robot[R.x]),Number(robot[R.y])];
      const school=String(message.schools?.[side]||"");
      const planned=terrainRoute(model.navigation,start,[primary.x,primary.y],school,role);
      const points=rawPoints.filter(point=>point.horizon<=primary.horizon).map(point=>{
        const [x,y]=pointAlongRoute(planned.route,point.horizon/primary.horizon);
        return {horizon:point.horizon,x,y};
      });
      const terrainAdjusted=planned.corrected||planned.passages.length>0||routeLength(planned.route)>distance(start,planned.target)+.35;
      predictions.push({
        side, school, role, robotId:robot[R.id], current:start, points,
        route:planned.route.map(point=>({x:point[0],y:point[1]})),
        passages:planned.passages, terrainAdjusted, destination:destinationZone(planned.target[0],planned.target[1],side),
        confidence:confidence(model,primary.horizon,moving), moving,
      });
    }
  }
  emit({
    type:"result", requestId:message.requestId, generation:message.generation,
    second:message.second, predictions, latencyMs:performance.now()-started,
  });
}

const predictionCore = {
  loadModel, buildFeatures, forward, linear, gelu, layerNorm,
  terrainRoute, routeLength, regionAt,
};

if (typeof self !== "undefined") {
  self.RMUCPredictionCore = predictionCore;
}

if (typeof self !== "undefined" && !self.RMUC_EMBEDDED_PREDICTION) {
  self.onmessage = event => {
    if (event.data?.type !== "predict") return;
    predict(event.data).catch(error=>emit({
      type:"error", requestId:event.data.requestId, generation:event.data.generation,
      message:error?.message||String(error),
    }));
  };
}

if (typeof module !== "undefined") module.exports=predictionCore;
