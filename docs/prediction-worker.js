"use strict";

// Browser-only trajectory inference.  Keeping this in a Worker means replay,
// seeking and canvas drawing never wait for the neural-network calculation.
const MOBILE_TYPES = ["英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"];
const GROUND_TYPES = new Set(["英雄", "工程", "步兵3", "步兵4", "哨兵"]);
const STRUCTURE_TYPES = ["基地", "前哨站"];
const FIELD_WIDTH = 28;
const FIELD_HEIGHT = 15;
const R = {id:0,type:1,side:2,hp:3,max:4,x:5,y:6,yaw:7,a17:8,a42:9,coins:10,vulnerable:11};
const MODEL_URL = "./data/models/trajectory_mlp.json?v=15";

let modelPromise = null;

function emit(message) {
  if (typeof self !== "undefined" && self.postMessage) self.postMessage(message);
}

async function loadModel() {
  if (modelPromise) return modelPromise;
  modelPromise = (async () => {
    emit({type:"status", status:"loading", text:"模型加载中"});
    const manifestUrl = new URL(MODEL_URL, self.location.href);
    const manifestResponse = await fetch(manifestUrl, {cache:"force-cache"});
    if (!manifestResponse.ok) throw new Error(`模型清单 HTTP ${manifestResponse.status}`);
    const manifest = await manifestResponse.json();
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
      tensors,
      mean: Float32Array.from(manifest.feature_mean),
      std: Float32Array.from(manifest.feature_std),
      targetX: manifest.feature_names.indexOf("target.x"),
      targetY: manifest.feature_names.indexOf("target.y"),
      targetVx3: manifest.feature_names.indexOf("target.vx_3_norm_per_s"),
      targetVy3: manifest.feature_names.indexOf("target.vy_3_norm_per_s"),
    };
    emit({type:"status", status:"ready", text:"预测已开启"});
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

function buildFeatures(history, second, targetSide, targetType, duration) {
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
  return values.length===240 ? Float32Array.from(values) : null;
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
function forward(model,features) {
  const t=name=>model.tensors.get(name), m=model.manifest, normalized=new Float32Array(m.input_dim);
  for (let i=0;i<m.input_dim;i++) normalized[i]=(features[i]-model.mean[i])/model.std[i];
  let hidden=linear(normalized,t("backbone.0.weight"),t("backbone.0.bias"),256,m.input_dim);
  hidden=layerNorm(gelu(hidden),t("backbone.2.weight"),t("backbone.2.bias"),m.layer_norm_epsilon);
  hidden=linear(hidden,t("backbone.4.weight"),t("backbone.4.bias"),256,256);
  hidden=layerNorm(gelu(hidden),t("backbone.6.weight"),t("backbone.6.bias"),m.layer_norm_epsilon);
  hidden=linear(hidden,t("backbone.8.weight"),t("backbone.8.bias"),128,256);
  hidden=layerNorm(gelu(hidden),t("backbone.10.weight"),t("backbone.10.bias"),m.layer_norm_epsilon);
  return linear(hidden,t("head.weight"),t("head.bias"),m.horizons.length*2,128);
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

async function predict(message) {
  const started=performance.now(), model=await loadModel(), predictions=[];
  const current=indexFrame(message.history["0"]||message.history[0]);
  for (const side of ["红","蓝"]) {
    for (const role of GROUND_TYPES) {
      const robot=rowAt(current,side,role);
      if (!validPosition(robot) || Number(robot[R.hp]||0)<=0) continue;
      const features=buildFeatures(message.history,message.second,side,role,message.duration);
      if (!features) continue;
      const residuals=forward(model,features), points=[];
      for (let i=0;i<model.manifest.horizons.length;i++) {
        const horizon=model.manifest.horizons[i];
        if (message.second+horizon>message.duration) continue;
        const [x,y]=worldPoint(features[model.targetX]+residuals[i*2],features[model.targetY]+residuals[i*2+1],side);
        points.push({horizon,x,y});
      }
      if (!points.length) continue;
      const vx=features[model.targetVx3]*FIELD_WIDTH, vy=features[model.targetVy3]*FIELD_HEIGHT;
      const moving=Math.hypot(vx,vy)>=0.15;
      const primary=points.find(point=>point.horizon===10)||points.at(-1);
      predictions.push({
        side, role, robotId:robot[R.id], current:[Number(robot[R.x]),Number(robot[R.y])], points,
        destination:destinationZone(primary.x,primary.y,side),
        confidence:confidence(model,primary.horizon,moving), moving,
      });
    }
  }
  emit({
    type:"result", requestId:message.requestId, generation:message.generation,
    second:message.second, predictions, latencyMs:performance.now()-started,
  });
}

if (typeof self !== "undefined") {
  self.onmessage = event => {
    if (event.data?.type !== "predict") return;
    predict(event.data).catch(error=>emit({
      type:"error", requestId:event.data.requestId, generation:event.data.generation,
      message:error?.message||String(error),
    }));
  };
}

if (typeof module !== "undefined") module.exports={buildFeatures,forward,linear,gelu,layerNorm};
