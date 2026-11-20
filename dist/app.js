"use strict";
const byId = id => document.getElementById(id);
const purple = "#704bd6", teal = "#138c86";
const svgNS = "http://www.w3.org/2000/svg";
function shape(parent,tag,attrs,label) {
  const el = document.createElementNS(svgNS,tag);
  for (const [key,value] of Object.entries(attrs)) el.setAttribute(key,value);
  if (label !== undefined) el.textContent=label;
  parent.append(el);
  return el;
}
let chosen="Rings", model, points, epochs, losses, timer=null;
const edges=[];
function buildNetwork() {
  const svg=byId("network");
  const nodes=NeuralLab.sizes.map((count,l) => Array.from({length:count}, (_,j) =>
    ({x:67+l*169, y:157+(j-(count-1)/2)*55})));
  for (let l=1;l<nodes.length;l++) nodes[l].forEach((target,j) => nodes[l-1].forEach((src,k) => {
    const el=shape(svg,"line",{x1:src.x,y1:src.y,x2:target.x,y2:target.y});
    const title=shape(el,"title",{});
    edges.push({el,title,l:l-1,j,k});
  }));
  const titles=["INPUT","HIDDEN 1","HIDDEN 2","OUTPUT"];
  nodes.forEach((layer,l) => {
    const x=layer[0].x;
    shape(svg,"text",{x,y:27,"text-anchor":"middle",fill:"#66627a","font-size":12,"letter-spacing":1.4,"font-weight":650},titles[l]);
    layer.forEach(({x,y},j) => {
      const fill=l===0 ? "#e0f4ef" : l===3 ? "#704bd6" : "#eee8ff";
      const stroke=l===0 ? teal : purple;
      shape(svg,"circle",{cx:x,cy:y,r:19,fill,stroke,"stroke-width":2});
      if (l===1 || l===2) shape(svg,"circle",{cx:x,cy:y,r:6,fill:"#a58ae8"});
      else shape(svg,"text",{x,y:y+5,fill:l===3 ? "white" : "#197871","text-anchor":"middle","font-size":16,"font-weight":650},l===3 ? "ŷ" : (j ? "x₂":"x₁"));
    });
  });
}
const thumbnails={Rings:"Rings",XOR:"XOR",Clusters:"Clusters"};
for (const name of Object.keys(thumbnails)) {
  const btn=document.createElement("button");
  const icon=shape(btn,"svg",{viewBox:"0 0 34 30","aria-hidden":true});
  NeuralLab.dataset(name).slice(0,28).forEach(({point:[x,y],label}) =>
    shape(icon,"circle",{cx:17+14*x,cy:15-13*y,r:1.55,fill:label ? teal : purple}));
  btn.append(name); btn.dataset.name=name;
  btn.onclick=()=>{if(chosen!==name){chosen=name;reset();}};
  byId("datasets").append(btn);
}
function drawBoundary() {
  const canvas=byId("map");
  const w=canvas.width=420,h=canvas.height=300,ctx=canvas.getContext("2d");
  const cell=7;
  for(let y=0;y<h;y+=cell) for(let x=0;x<w;x+=cell) {
    const p=model.forward([(x+cell/2)/w*2-1,1-(y+cell/2)/h*2]).at(-1)[0];
    const a=[222,211,250], b=[197,236,227];
    ctx.fillStyle="rgb("+a.map((v,i)=>Math.round(v+(b[i]-v)*p)).join(",")+")";
    ctx.fillRect(x,y,cell,cell);
  }
  ctx.strokeStyle="#ffffff85";ctx.lineWidth=1;
  for(let n=1;n<8;n++){
    ctx.beginPath();ctx.moveTo(n*w/8,0);ctx.lineTo(n*w/8,h);
    ctx.moveTo(0,n*h/8);ctx.lineTo(w,n*h/8);ctx.stroke();
  }
  for (const {point:[x,y],label} of points) {
    ctx.beginPath();ctx.arc((x+1)*w/2,(1-y)*h/2,4.1,0,2*Math.PI);
    ctx.fillStyle=label?teal:purple;ctx.fill();ctx.strokeStyle="white";ctx.lineWidth=1.3;ctx.stroke();
  }
}
function drawLoss(){
  const canvas=byId("history"),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#eae7f0";ctx.lineWidth=1;
  for(let i=0;i<3;i++){const y=6+i*(h-12)/2;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  const peak=losses.reduce((max,v)=>Math.max(max,v),.1);
  const end=losses.length-1;
  const pos=i=>[3+(w-6)*i/Math.max(end,1),h-5-losses[i]/peak*(h-12)];
  ctx.beginPath();ctx.moveTo(...pos(0));
  for(let i=1;i<=Math.min(end,w);i++)ctx.lineTo(...pos(Math.round(i*end/Math.min(end,w))));
  ctx.strokeStyle=purple;ctx.lineWidth=3;ctx.stroke();
  ctx.beginPath();ctx.arc(...pos(end),3.4,0,2*Math.PI);ctx.fillStyle=purple;ctx.fill();
}
function render(){
  const metrics=model.metrics(points);losses.push(metrics.loss);
  byId("epoch").textContent=epochs.toLocaleString();
  byId("loss").textContent=metrics.loss.toFixed(3);
  byId("accuracy").textContent=Math.round(metrics.accuracy*100)+"%";
  for(const {el,title,l,j,k} of edges){
    const weight=model.layers[l][j][k].w, magnitude=Math.min(Math.abs(weight),3);
    el.setAttribute("stroke",weight<0?teal:purple);
    el.setAttribute("stroke-width",.8+1.65*magnitude);
    el.setAttribute("opacity",.35+.19*magnitude);
    title.textContent="Weight "+weight.toFixed(3);
  }
  drawBoundary();drawLoss();
}
function updateState(){
  const running=timer!==null;
  byId("status").textContent=running?"Training":epochs?"Paused":"Ready";
  byId("status").classList.toggle("active",running);
  byId("playlabel").textContent=running?"Pause training":epochs?"Resume training":"Start training";
  byId("playicon").textContent=running?"Ⅱ":"▶";
}
function reset(){
  clearInterval(timer);timer=null;epochs=0;losses=[];
  points=NeuralLab.dataset(chosen);model=new NeuralLab.Model(byId("activation").value);
  document.querySelectorAll("#datasets button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.name===chosen));
  updateState();render();
}
byId("train").onclick=()=>{
  if(timer!==null){clearInterval(timer);timer=null;}
  else timer=setInterval(()=>{model.epoch(points,Number(byId("rate").value));epochs++;render();},100);
  updateState();
};
byId("reset").onclick=reset;
byId("activation").onchange=reset;
byId("guide").onclick=()=>byId("help").showModal();
buildNetwork();reset();
