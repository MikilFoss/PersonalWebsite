"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[301],{4119:(e,t,i)=>{i.d(t,{TwistyPlayer:()=>tz});var r,n=i(618),a=i(2062),s=i(8586),l=i(9021);i(138);var o=i(9131);function d(e,t){if(e===t)return!0;if(e.length!==t.length)return!1;for(let i=0;i<e.length;i++)if(e[i]!==t[i])return!1;return!0}function u(e,t,i){if(e===t)return!0;if(e.length!==t.length)return!1;for(let r=0;r<e.length;r++)if(!i(e[r],t[r]))return!1;return!0}function c(e,t,i){return(0,o._P)(e,i-t,t)}var h=class{constructor(e){this.model=e,e.tempoScale.addFreshListener(e=>{this.tempoScale=e})}catchingUp=!1;pendingFrame=!1;tempoScale;scheduler=new n.mN(this.animFrame.bind(this));start(){this.catchingUp||(this.lastTimestamp=performance.now()),this.catchingUp=!0,this.pendingFrame=!0,this.scheduler.requestAnimFrame()}stop(){this.catchingUp=!1,this.scheduler.cancelAnimFrame()}catchUpMs=500;lastTimestamp=0;animFrame(e){this.scheduler.requestAnimFrame();let t=this.tempoScale*(e-this.lastTimestamp)/this.catchUpMs;this.lastTimestamp=e,this.model.catchUpMove.set((async()=>{let e=await this.model.catchUpMove.get();if(null===e.move)return e;let i=e.amount+t;return i>=1?(this.pendingFrame=!0,this.stop(),this.model.timestampRequest.set("end"),{move:null,amount:0}):(this.pendingFrame=!1,{move:e.move,amount:i})})())}},p=class{constructor(e,t){this.delegate=t,this.model=e,this.lastTimestampPromise=this.#e(),this.model.playingInfo.addFreshListener(this.onPlayingProp.bind(this)),this.catchUpHelper=new h(this.model),this.model.catchUpMove.addFreshListener(this.onCatchUpMoveProp.bind(this))}playing=!1;direction=1;catchUpHelper;model;lastDatestamp=0;lastTimestampPromise;scheduler=new n.mN(this.animFrame.bind(this));async onPlayingProp(e){e.playing!==this.playing&&(e.playing?this.play(e):this.pause())}async onCatchUpMoveProp(e){let t=null!==e.move;t!==this.catchUpHelper.catchingUp&&(t?this.catchUpHelper.start():this.catchUpHelper.stop()),this.scheduler.requestAnimFrame()}async #e(){return(await this.model.detailedTimelineInfo.get()).timestamp}jumpToStart(e){this.model.timestampRequest.set("start"),this.pause(),e?.flash&&this.delegate.flash()}jumpToEnd(e){this.model.timestampRequest.set("end"),this.pause(),e?.flash&&this.delegate.flash()}playPause(){this.playing?this.pause():this.play()}async play(e){let t=e?.direction??1,i=await this.model.coarseTimelineInfo.get();(e?.autoSkipToOtherEndIfStartingAtBoundary??!0)&&(1===t&&i.atEnd&&(this.model.timestampRequest.set("start"),this.delegate.flash()),-1===t&&i.atStart&&(this.model.timestampRequest.set("end"),this.delegate.flash())),this.model.playingInfo.set({playing:!0,direction:t,untilBoundary:e?.untilBoundary??"entire-timeline",loop:e?.loop??!1}),this.playing=!0,this.lastDatestamp=performance.now(),this.lastTimestampPromise=this.#e(),this.scheduler.requestAnimFrame()}pause(){this.playing=!1,this.scheduler.cancelAnimFrame(),this.model.playingInfo.set({playing:!1,untilBoundary:"entire-timeline"})}#t=new n.YM;async animFrame(e){this.playing&&this.scheduler.requestAnimFrame();let t=this.lastDatestamp,[i,r,n,a,s]=await this.#t.queue(Promise.all([this.model.playingInfo.get(),this.lastTimestampPromise,this.model.timeRange.get(),this.model.tempoScale.get(),this.model.currentMoveInfo.get()]));if(!i.playing){this.playing=!1;return}let l=s.earliestEnd;(0===s.currentMoves.length||"entire-timeline"===i.untilBoundary)&&(l=n.end);let o=s.latestStart;(0===s.currentMoves.length||"entire-timeline"===i.untilBoundary)&&(o=n.start);let d=(e-t)*this.direction*a,u=r+(d=Math.max(d,1)*i.direction),h=null;u>=l?i.loop?u=c(u,n.start,n.end):(u===n.end?h="end":u=l,this.playing=!1,this.model.playingInfo.set({playing:!1})):u<=o&&(i.loop?u=c(u,n.start,n.end):(u===n.start?h="start":u=o,this.playing=!1,this.model.playingInfo.set({playing:!1}))),this.lastDatestamp=e,this.lastTimestampPromise=Promise.resolve(u),this.model.timestampRequest.set(h??u)}},m=class{constructor(e,t){this.model=e,this.animationController=new p(e,t)}animationController;jumpToStart(e){this.animationController.jumpToStart(e)}jumpToEnd(e){this.animationController.jumpToEnd(e)}togglePlay(e){void 0===e&&this.animationController.playPause(),e?this.animationController.play():this.animationController.pause()}async visitTwizzleLink(){let e=document.createElement("a");e.href=await this.model.twizzleLink(),e.target="_blank",e.click()}},g={"bottom-row":!0,none:!0},w=class extends n.nB{getDefaultValue(){return"auto"}},y=new n.n5;y.replaceSync(`
:host {
  width: 384px;
  height: 256px;
  display: grid;
}

.wrapper {
  width: 100%;
  height: 100%;
  display: grid;
  overflow: hidden;
}

.wrapper > * {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.wrapper.back-view-side-by-side {
  grid-template-columns: 1fr 1fr;
}

.wrapper.back-view-top-right {
  grid-template-columns: 3fr 1fr;
  grid-template-rows: 1fr 3fr;
}

.wrapper.back-view-top-right > :nth-child(1) {
  grid-row: 1 / 3;
  grid-column: 1 / 3;
}

.wrapper.back-view-top-right > :nth-child(2) {
  grid-row: 1 / 2;
  grid-column: 2 / 3;
}
`);var f="http://www.w3.org/2000/svg",v="data-copy-id",M=0,x={dim:{white:"#dddddd",orange:"#884400",limegreen:"#008800",red:"#660000","rgb(34, 102, 255)":"#000088",yellow:"#888800","rgb(102, 0, 153)":"rgb(50, 0, 76)",purple:"#3f003f"},oriented:"#44ddcc",ignored:"#555555",invisible:"#00000000"},z=class{constructor(e,t,i,r=!1){if(this.kpuzzle=e,this.showUnknownOrientations=r,!t)throw Error(`No SVG definition for puzzle type: ${e.name()}`);this.svgID=(M+=1,`svg${M.toString()}`),this.wrapperElement=document.createElement("div"),this.wrapperElement.classList.add("svg-wrapper"),this.wrapperElement.innerHTML=t;let n=this.wrapperElement.querySelector("svg");if(!n)throw Error("Could not get SVG element");if(this.svgElement=n,f!==n.namespaceURI)throw Error("Unexpected XML namespace");for(let t of(n.style.maxWidth="100%",n.style.maxHeight="100%",this.gradientDefs=document.createElementNS(f,"defs"),n.insertBefore(this.gradientDefs,n.firstChild),e.definition.orbits))for(let e=0;e<t.numPieces;e++)for(let r=0;r<t.numOrientations;r++){let n=this.elementID(t.orbitName,e,r),a=this.elementByID(n),s=a?.style.fill;i?(()=>{let n=i.orbits;if(!n)return;let a=n[t.orbitName];if(!a)return;let l=a.pieces[e];if(!l)return;let o=l.facelets[r];if(!o)return;let d=x["string"==typeof o?o:o?.mask];"string"==typeof d?s=d:d&&(s=d[s])})():s=a?.style.fill,this.originalColors[n]=s,this.gradients[n]=this.newGradient(n,s),this.gradientDefs.appendChild(this.gradients[n]),a?.setAttribute("style",`fill: url(#grad-${this.svgID}-${n})`)}for(let e of Array.from(n.querySelectorAll(`[${v}]`))){let t=e.getAttribute(v);e.setAttribute("style",`fill: url(#grad-${this.svgID}-${t})`)}this.showUnknownOrientations&&this.drawPattern(this.kpuzzle.defaultPattern())}wrapperElement;svgElement;gradientDefs;originalColors={};gradients={};svgID;drawPattern(e,t,i){this.draw(e,t,i)}draw(e,t,i){let r=t?.experimentalToTransformation();if(!e)throw Error("Distinguishable pieces are not handled for SVG yet!");for(let t of e.kpuzzle.definition.orbits){let n=e.patternData[t.orbitName],a=r?r.transformationData[t.orbitName]:null;for(let e=0;e<t.numPieces;e++)for(let r=0;r<t.numOrientations;r++){let s=this.elementID(t.orbitName,e,r),l=this.elementID(t.orbitName,n.pieces[e],(t.numOrientations-n.orientation[e]+r)%t.numOrientations),o=!1;if(a){let n=this.elementID(t.orbitName,a.permutation[e],(t.numOrientations-a.orientationDelta[e]+r)%t.numOrientations);l===n&&(o=!0);let d=100*(1-(i=i||0)*i*(2-i*i));this.gradients[s].children[0].setAttribute("stop-color",this.originalColors[l]),this.gradients[s].children[0].setAttribute("offset",`${Math.max(d-5,0)}%`),this.gradients[s].children[1].setAttribute("offset",`${Math.max(d-5,0)}%`),this.gradients[s].children[2].setAttribute("offset",`${d}%`),this.gradients[s].children[3].setAttribute("offset",`${d}%`),this.gradients[s].children[3].setAttribute("stop-color",this.originalColors[n])}else o=!0;o&&(this.showUnknownOrientations&&n.orientationMod?.[e]===1?(this.gradients[s].children[0].setAttribute("stop-color","#000"),this.gradients[s].children[0].setAttribute("offset","5%"),this.gradients[s].children[1].setAttribute("offset","5%"),this.gradients[s].children[2].setAttribute("offset","20%"),this.gradients[s].children[3].setAttribute("offset","20%"),this.gradients[s].children[3].setAttribute("stop-color",this.originalColors[l])):(this.gradients[s].children[0].setAttribute("stop-color",this.originalColors[l]),this.gradients[s].children[0].setAttribute("offset","100%"),this.gradients[s].children[1].setAttribute("offset","100%"),this.gradients[s].children[2].setAttribute("offset","100%"),this.gradients[s].children[3].setAttribute("offset","100%")))}}}newGradient(e,t){let i=document.createElementNS(f,"radialGradient");for(let r of(i.setAttribute("id",`grad-${this.svgID}-${e}`),i.setAttribute("r","70.7107%"),[{offset:0,color:t},{offset:0,color:"black"},{offset:0,color:"black"},{offset:0,color:t}])){let e=document.createElementNS(f,"stop");e.setAttribute("offset",`${r.offset}%`),e.setAttribute("stop-color",r.color),e.setAttribute("stop-opacity","1"),i.appendChild(e)}return i}elementID(e,t,i){return`${e}-l${t}-o${i}`}elementByID(e){return this.wrapperElement.querySelector(`#${e}`)}},L=new n.n5;L.replaceSync(`
:host {
  width: 384px;
  height: 256px;
  display: grid;
}

.wrapper {
  width: 100%;
  height: 100%;
  display: grid;
  overflow: hidden;
}

.svg-wrapper,
twisty-2d-svg,
svg {
  width: 100%;
  height: 100%;
  display: grid;
  min-height: 0;
}

svg {
  animation: fade-in 0.25s ease-in;
}

@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}
`);var b=class extends n.FD{constructor(e,t,i,r,n){super(),this.model=e,this.kpuzzle=t,this.svgSource=i,this.options=r,this.puzzleLoader=n,this.addCSS(L),this.resetSVG(),this.#i.addListener(this.model.puzzleID,e=>{n?.id!==e&&this.disconnect()}),this.#i.addListener(this.model.legacyPosition,this.onPositionChange.bind(this)),this.options?.experimentalStickeringMask&&this.experimentalSetStickeringMask(this.options.experimentalStickeringMask)}svgWrapper;scheduler=new n.mN(this.render.bind(this));#r=null;#i=new n.Y;disconnect(){this.#i.disconnect()}onPositionChange(e){try{if(e.movesInProgress.length>0){let t=e.movesInProgress[0].move,i=t;-1===e.movesInProgress[0].direction&&(i=t.invert());let r=e.pattern.applyMove(i);this.svgWrapper.draw(e.pattern,r,e.movesInProgress[0].fraction)}else this.svgWrapper.draw(e.pattern),this.#r=e}catch(e){console.warn("Bad position (this doesn't necessarily mean something is wrong). Pre-emptively disconnecting:",this.puzzleLoader?.id,e),this.disconnect()}}scheduleRender(){this.scheduler.requestAnimFrame()}experimentalSetStickeringMask(e){this.resetSVG(e)}resetSVG(e){this.svgWrapper&&this.removeElement(this.svgWrapper.wrapperElement),this.kpuzzle&&(this.svgWrapper=new z(this.kpuzzle,this.svgSource,e),this.addElement(this.svgWrapper.wrapperElement),this.#r&&this.onPositionChange(this.#r))}render(){}};n.qh.define("twisty-2d-puzzle",b);var D=class{constructor(e,t,i,r){this.model=e,this.schedulable=t,this.puzzleLoader=i,this.effectiveVisualization=r,this.twisty2DPuzzle(),this.#i.addListener(this.model.twistySceneModel.stickeringMask,async e=>{(await this.twisty2DPuzzle()).experimentalSetStickeringMask(e)})}#i=new n.Y;disconnect(){this.#i.disconnect()}scheduleRender(){}#n=null;async twisty2DPuzzle(){return this.#n??=(async()=>{let e="experimental-2D-LL-face"===this.effectiveVisualization?this.puzzleLoader.llFaceSVG():"experimental-2D-LL"===this.effectiveVisualization?this.puzzleLoader.llSVG():this.puzzleLoader.svg();return new b(this.model,await this.puzzleLoader.kpuzzle(),await e,{},this.puzzleLoader)})()}},k=class extends n.FD{constructor(e,t){super(),this.model=e,this.effectiveVisualization=t}#i=new n.Y;disconnect(){this.#i.disconnect()}async connectedCallback(){this.addCSS(y),this.model&&this.#i.addListener(this.model.twistyPlayerModel.puzzleLoader,this.onPuzzleLoader.bind(this))}#a;async scene(){return this.#a??=(async()=>new(await n.u_).Scene)()}scheduleRender(){this.#s?.scheduleRender()}#s=null;currentTwisty2DPuzzleWrapper(){return this.#s}async setCurrentTwisty2DPuzzleWrapper(e){let t=this.#s;this.#s=e,t?.disconnect();let i=e.twisty2DPuzzle();this.contentWrapper.textContent="",this.addElement(await i)}async onPuzzleLoader(e){this.#s?.disconnect();let t=new D(this.model.twistyPlayerModel,this,e,this.effectiveVisualization);this.setCurrentTwisty2DPuzzleWrapper(t)}};n.qh.define("twisty-2d-scene-wrapper",k);var T=class{constructor(e,t,i){this.elem=e,this.prefix=t,this.validSuffixes=i}#l=null;clearValue(){this.#l&&this.elem.contentWrapper.classList.remove(this.#l),this.#l=null}setValue(e){if(!this.validSuffixes.includes(e))throw Error(`Invalid suffix: ${e}`);let t=`${this.prefix}${e}`,i=this.#l!==t;return i&&(this.clearValue(),this.elem.contentWrapper.classList.add(t),this.#l=t),i}},I=class{#o;reject;promise=new Promise((e,t)=>{this.#o=e,this.reject=t});handleNewValue(e){this.#o(e)}},S=class extends EventTarget{constructor(e,t,i,r){super(),this.model=e,this.schedulable=t,this.puzzleLoader=i,this.visualizationStrategy=r,this.twisty3DPuzzle(),this.#i.addListener(this.model.puzzleLoader,e=>{this.puzzleLoader.id!==e.id&&this.disconnect()}),this.#i.addListener(this.model.legacyPosition,async e=>{try{(await this.twisty3DPuzzle()).onPositionChange(e),this.scheduleRender()}catch(e){this.disconnect()}}),this.#i.addListener(this.model.twistySceneModel.hintFacelet,async e=>{(await this.twisty3DPuzzle()).experimentalUpdateOptions({hintFacelets:"auto"===e?"floating":e}),this.scheduleRender()}),this.#i.addListener(this.model.twistySceneModel.foundationDisplay,async e=>{(await this.twisty3DPuzzle()).experimentalUpdateOptions({showFoundation:"none"!==e}),this.scheduleRender()}),this.#i.addListener(this.model.twistySceneModel.stickeringMask,async e=>{(await this.twisty3DPuzzle()).setStickeringMask(e),this.scheduleRender()}),this.#i.addListener(this.model.twistySceneModel.faceletScale,async e=>{(await this.twisty3DPuzzle()).experimentalUpdateOptions({faceletScale:e}),this.scheduleRender()}),this.#i.addMultiListener3([this.model.twistySceneModel.stickeringMask,this.model.twistySceneModel.foundationStickerSprite,this.model.twistySceneModel.hintStickerSprite],async e=>{"experimentalUpdateTexture"in await this.twisty3DPuzzle()&&((await this.twisty3DPuzzle()).experimentalUpdateTexture("picture"===e[0].specialBehaviour,e[1],e[2]),this.scheduleRender())})}#i=new n.Y;disconnect(){this.#i.disconnect()}scheduleRender(){this.schedulable.scheduleRender(),this.dispatchEvent(new CustomEvent("render-scheduled"))}#d=null;async twisty3DPuzzle(){return this.#d??=(async()=>{let e=(0,n.yU)();if("3x3x3"===this.puzzleLoader.id&&"Cube3D"===this.visualizationStrategy){let[t,i,r,n]=await Promise.all([this.model.twistySceneModel.foundationStickerSprite.get(),this.model.twistySceneModel.hintStickerSprite.get(),this.model.twistySceneModel.stickeringMask.get(),this.model.twistySceneModel.initialHintFaceletsAnimation.get()]);return(await e).cube3DShim(()=>this.schedulable.scheduleRender(),{foundationSprite:t,hintSprite:i,experimentalStickeringMask:r,initialHintFaceletsAnimation:n})}{let[t,i,r,n]=await Promise.all([this.model.twistySceneModel.hintFacelet.get(),this.model.twistySceneModel.foundationStickerSprite.get(),this.model.twistySceneModel.hintStickerSprite.get(),this.model.twistySceneModel.faceletScale.get()]),a=(await e).pg3dShim(()=>this.schedulable.scheduleRender(),this.puzzleLoader,"auto"===t?"floating":t,n,"kilominx"===this.puzzleLoader.id);return a.then(e=>e.experimentalUpdateTexture(!0,i??void 0,r??void 0)),a}})()}async raycastMove(e,t){let i=await this.twisty3DPuzzle();if(!("experimentalGetControlTargets"in i))return void console.info("not PG3D! skipping raycast");let r=i.experimentalGetControlTargets(),[n,a]=await Promise.all([e,this.model.twistySceneModel.movePressCancelOptions.get()]),s=n.intersectObjects(r);if(s.length>0){let e=i.getClosestMoveToAxis(s[0].point,t);e?this.model.experimentalAddMove(e.move,{cancel:a}):console.info("Skipping move!")}}},A=class extends n.FD{constructor(e){super(),this.model=e}#u=new T(this,"back-view-",["auto","none","side-by-side","top-right"]);#i=new n.Y;disconnect(){this.#i.disconnect()}async connectedCallback(){this.addCSS(y);let e=new n.uf(this.model,this);this.addVantage(e),this.model&&(this.#i.addMultiListener([this.model.puzzleLoader,this.model.visualizationStrategy],this.onPuzzle.bind(this)),this.#i.addListener(this.model.backView,this.onBackView.bind(this))),this.scheduleRender()}#c=null;setBackView(e){let t=["side-by-side","top-right"].includes(e),i=null!==this.#c;this.#u.setValue(e),t?i||(this.#c=new n.uf(this.model,this,{backView:!0}),this.addVantage(this.#c),this.scheduleRender()):this.#c&&(this.removeVantage(this.#c),this.#c=null)}onBackView(e){this.setBackView(e)}async onPress(e){let t=this.#h;if(!t)return void console.info("no wrapper; skipping scene wrapper press!");let i=(async()=>{let[t,i]=await Promise.all([e.detail.cameraPromise,n.u_]),r=new i.Raycaster,a=new(await n.u_).Vector2(e.detail.pressInfo.normalizedX,e.detail.pressInfo.normalizedY);return r.setFromCamera(a,t),r})();t.raycastMove(i,{invert:!e.detail.pressInfo.rightClick,depth:e.detail.pressInfo.keys.ctrlOrMetaKey?"rotation":e.detail.pressInfo.keys.shiftKey?"secondSlice":"none"})}#a;async scene(){return this.#a??=(async()=>new(await n.u_).Scene)()}#p=new Set;addVantage(e){e.addEventListener("press",this.onPress.bind(this)),this.#p.add(e),this.contentWrapper.appendChild(e)}removeVantage(e){this.#p.delete(e),e.remove(),e.disconnect(),this.#h?.disconnect()}experimentalVantages(){return this.#p.values()}scheduleRender(){for(let e of this.#p)e.scheduleRender()}#h=null;async setCurrentTwisty3DPuzzleWrapper(e,t){let i=this.#h;try{this.#h=t,i?.disconnect(),e.add(await t.twisty3DPuzzle())}finally{i&&e.remove(await i.twisty3DPuzzle())}this.#m.handleNewValue(t)}#m=new I;async experimentalTwisty3DPuzzleWrapper(){return this.#h||this.#m.promise}#g=new n.YM;async onPuzzle(e){if("2D"===e[1])return;this.#h?.disconnect();let[t,i]=await this.#g.queue(Promise.all([this.scene(),new S(this.model,this,e[0],e[1])]));this.setCurrentTwisty3DPuzzleWrapper(t,i)}};n.qh.define("twisty-3d-scene-wrapper",A);var C=new n.n5;C.replaceSync(`
:host {
  width: 384px;
  height: 24px;
  display: grid;
}

.wrapper {
  width: 100%;
  height: 100%;
  display: grid;
  overflow: hidden;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}

.wrapper {
  grid-auto-flow: column;
}

.viewer-link-none .twizzle-link-button {
  display: none;
}

.wrapper twisty-button,
.wrapper twisty-control-button {
  width: inherit;
  height: inherit;
}
`);var E=new n.n5;E.replaceSync(`
:host:not([hidden]) {
  display: grid;
}

:host {
  width: 48px;
  height: 24px;
}

.wrapper {
  width: 100%;
  height: 100%;
}

button {
  width: 100%;
  height: 100%;
  border: none;
  
  background-position: center;
  background-repeat: no-repeat;
  background-size: contain;

  background-color: rgba(196, 196, 196, 0.75);
}

button:enabled {
  background-color: rgba(196, 196, 196, 0.75)
}

.dark-mode button:enabled {
  background-color: #88888888;
}

button:disabled {
  background-color: rgba(0, 0, 0, 0.4);
  opacity: 0.25;
  pointer-events: none;
}

.dark-mode button:disabled {
  background-color: #ffffff44;
}

button:enabled:hover {
  background-color: rgba(255, 255, 255, 0.75);
  box-shadow: 0 0 1em rgba(0, 0, 0, 0.25);
  cursor: pointer;
}

/* TODO: fullscreen icons have too much padding?? */
.svg-skip-to-start button,
button.svg-skip-to-start {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNTg0IiBoZWlnaHQ9IjM1ODQiIHZpZXdCb3g9IjAgMCAzNTg0IDM1ODQiPjxwYXRoIGQ9Ik0yNjQzIDEwMzdxMTktMTkgMzItMTN0MTMgMzJ2MTQ3MnEwIDI2LTEzIDMydC0zMi0xM2wtNzEwLTcxMHEtOS05LTEzLTE5djcxMHEwIDI2LTEzIDMydC0zMi0xM2wtNzEwLTcxMHEtOS05LTEzLTE5djY3OHEwIDI2LTE5IDQ1dC00NSAxOUg5NjBxLTI2IDAtNDUtMTl0LTE5LTQ1VjEwODhxMC0yNiAxOS00NXQ0NS0xOWgxMjhxMjYgMCA0NSAxOXQxOSA0NXY2NzhxNC0xMSAxMy0xOWw3MTAtNzEwcTE5LTE5IDMyLTEzdDEzIDMydjcxMHE0LTExIDEzLTE5eiIvPjwvc3ZnPg==");
}

.svg-skip-to-end button,
button.svg-skip-to-end {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNTg0IiBoZWlnaHQ9IjM1ODQiIHZpZXdCb3g9IjAgMCAzNTg0IDM1ODQiPjxwYXRoIGQ9Ik05NDEgMjU0N3EtMTkgMTktMzIgMTN0LTEzLTMyVjEwNTZxMC0yNiAxMy0zMnQzMiAxM2w3MTAgNzEwcTggOCAxMyAxOXYtNzEwcTAtMjYgMTMtMzJ0MzIgMTNsNzEwIDcxMHE4IDggMTMgMTl2LTY3OHEwLTI2IDE5LTQ1dDQ1LTE5aDEyOHEyNiAwIDQ1IDE5dDE5IDQ1djE0MDhxMCAyNi0xOSA0NXQtNDUgMTloLTEyOHEtMjYgMC00NS0xOXQtMTktNDV2LTY3OHEtNSAxMC0xMyAxOWwtNzEwIDcxMHEtMTkgMTktMzIgMTN0LTEzLTMydi03MTBxLTUgMTAtMTMgMTl6Ii8+PC9zdmc+");
}

.svg-step-forward button,
button.svg-step-forward {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNTg0IiBoZWlnaHQ9IjM1ODQiIHZpZXdCb3g9IjAgMCAzNTg0IDM1ODQiPjxwYXRoIGQ9Ik0yNjg4IDE1NjhxMCAyNi0xOSA0NWwtNTEyIDUxMnEtMTkgMTktNDUgMTl0LTQ1LTE5cS0xOS0xOS0xOS00NXYtMjU2aC0yMjRxLTk4IDAtMTc1LjUgNnQtMTU0IDIxLjVxLTc2LjUgMTUuNS0xMzMgNDIuNXQtMTA1LjUgNjkuNXEtNDkgNDIuNS04MCAxMDF0LTQ4LjUgMTM4LjVxLTE3LjUgODAtMTcuNSAxODEgMCA1NSA1IDEyMyAwIDYgMi41IDIzLjV0Mi41IDI2LjVxMCAxNS04LjUgMjV0LTIzLjUgMTBxLTE2IDAtMjgtMTctNy05LTEzLTIydC0xMy41LTMwcS03LjUtMTctMTAuNS0yNC0xMjctMjg1LTEyNy00NTEgMC0xOTkgNTMtMzMzIDE2Mi00MDMgODc1LTQwM2gyMjR2LTI1NnEwLTI2IDE5LTQ1dDQ1LTE5cTI2IDAgNDUgMTlsNTEyIDUxMnExOSAxOSAxOSA0NXoiLz48L3N2Zz4=");
}

.svg-step-backward button,
button.svg-step-backward {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNTg0IiBoZWlnaHQ9IjM1ODQiIHZpZXdCb3g9IjAgMCAzNTg0IDM1ODQiPjxwYXRoIGQ9Ik0yNjg4IDIwNDhxMCAxNjYtMTI3IDQ1MS0zIDctMTAuNSAyNHQtMTMuNSAzMHEtNiAxMy0xMyAyMi0xMiAxNy0yOCAxNy0xNSAwLTIzLjUtMTB0LTguNS0yNXEwLTkgMi41LTI2LjV0Mi41LTIzLjVxNS02OCA1LTEyMyAwLTEwMS0xNy41LTE4MXQtNDguNS0xMzguNXEtMzEtNTguNS04MC0xMDF0LTEwNS41LTY5LjVxLTU2LjUtMjctMTMzLTQyLjV0LTE1NC0yMS41cS03Ny41LTYtMTc1LjUtNmgtMjI0djI1NnEwIDI2LTE5IDQ1dC00NSAxOXEtMjYgMC00NS0xOWwtNTEyLTUxMnEtMTktMTktMTktNDV0MTktNDVsNTEyLTUxMnExOS0xOSA0NS0xOXQ0NSAxOXExOSAxOSAxOSA0NXYyNTZoMjI0cTcxMyAwIDg3NSA0MDMgNTMgMTM0IDUzIDMzM3oiLz48L3N2Zz4=");
}

.svg-pause button,
button.svg-pause {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNTg0IiBoZWlnaHQ9IjM1ODQiIHZpZXdCb3g9IjAgMCAzNTg0IDM1ODQiPjxwYXRoIGQ9Ik0yNTYwIDEwODh2MTQwOHEwIDI2LTE5IDQ1dC00NSAxOWgtNTEycS0yNiAwLTQ1LTE5dC0xOS00NVYxMDg4cTAtMjYgMTktNDV0NDUtMTloNTEycTI2IDAgNDUgMTl0MTkgNDV6bS04OTYgMHYxNDA4cTAgMjYtMTkgNDV0LTQ1IDE5aC01MTJxLTI2IDAtNDUtMTl0LTE5LTQ1VjEwODhxMC0yNiAxOS00NXQ0NS0xOWg1MTJxMjYgMCA0NSAxOXQxOSA0NXoiLz48L3N2Zz4=");
}

.svg-play button,
button.svg-play {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNTg0IiBoZWlnaHQ9IjM1ODQiIHZpZXdCb3g9IjAgMCAzNTg0IDM1ODQiPjxwYXRoIGQ9Ik0yNDcyLjUgMTgyM2wtMTMyOCA3MzhxLTIzIDEzLTM5LjUgM3QtMTYuNS0zNlYxMDU2cTAtMjYgMTYuNS0zNnQzOS41IDNsMTMyOCA3MzhxMjMgMTMgMjMgMzF0LTIzIDMxeiIvPjwvc3ZnPg==");
}

.svg-enter-fullscreen button,
button.svg-enter-fullscreen {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCAyOCAyOCIgd2lkdGg9IjI4Ij48cGF0aCBkPSJNMiAyaDI0djI0SDJ6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTkgMTZIN3Y1aDV2LTJIOXYtM3ptLTItNGgyVjloM1Y3SDd2NXptMTIgN2gtM3YyaDV2LTVoLTJ2M3pNMTYgN3YyaDN2M2gyVjdoLTV6Ii8+PC9zdmc+");
}

.svg-exit-fullscreen button,
button.svg-exit-fullscreen {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCAyOCAyOCIgd2lkdGg9IjI4Ij48cGF0aCBkPSJNMiAyaDI0djI0SDJ6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTcgMThoM3YzaDJ2LTVIN3Yyem0zLThIN3YyaDVWN2gtMnYzem02IDExaDJ2LTNoM3YtMmgtNXY1em0yLTExVjdoLTJ2NWg1di0yaC0zeiIvPjwvc3ZnPg==");
}

.svg-twizzle-tw button,
button.svg-twizzle-tw {
  background-image: url("data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODY0IiBoZWlnaHQ9IjYwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMzk3LjU4MSAxNTEuMTh2NTcuMDg0aC04OS43MDN2MjQwLjM1MmgtNjYuOTU1VjIwOC4yNjRIMTUxLjIydi01Ny4wODNoMjQ2LjM2MXptNTQuMzEgNzEuNjc3bDcuNTEyIDMzLjY5MmMyLjcxOCAxMi4xNiA1LjU4IDI0LjY4IDguNTg0IDM3LjU1NWEyMTgwLjc3NSAyMTgwLjc3NSAwIDAwOS40NDIgMzguODQzIDEyNjYuMyAxMjY2LjMgMCAwMDEwLjA4NiAzNy41NTVjMy43Mi0xMi41OSA3LjM2OC0yNS40NjYgMTAuOTQ1LTM4LjYyOCAzLjU3Ni0xMy4xNjIgNy4wMS0yNi4xMSAxMC4zLTM4Ljg0M2w1Ljc2OS0yMi40NTZjMS4yNDgtNC44ODcgMi40NzItOS43MDUgMy42NzQtMTQuNDU1IDMuMDA0LTExLjg3NSA1LjY1MS0yMi45NjIgNy45NC0zMy4yNjNoNDYuMzU0bDIuMzg0IDEwLjU2M2EyMDAwLjc3IDIwMDAuNzcgMCAwMDMuOTM1IDE2LjgyOGw2LjcxMSAyNy43MWMxLjIxMyA0Ljk1NiAyLjQ1IDkuOTggMy43MDkgMTUuMDczYTMxMTkuNzc3IDMxMTkuNzc3IDAgMDA5Ljg3MSAzOC44NDMgMTI0OS4yMjcgMTI0OS4yMjcgMCAwMDEwLjczIDM4LjYyOCAxOTA3LjYwNSAxOTA3LjYwNSAwIDAwMTAuMzAxLTM3LjU1NSAxMzk3Ljk0IDEzOTcuOTQgMCAwMDkuNjU3LTM4Ljg0M2w0LjQtMTkuMDQ2Yy43MTUtMy4xMyAxLjQyMS02LjIzNiAyLjExOC05LjMyMWw5LjU3Ny00Mi44OGg2Ni41MjZhMjk4OC43MTggMjk4OC43MTggMCAwMS0xOS41MjkgNjYuMzExbC01LjcyOCAxOC40ODJhMzIzNy40NiAzMjM3LjQ2IDAgMDEtMTQuMDE1IDQzLjc1MmMtNi40MzggMTkuNi0xMi43MzMgMzcuNjk4LTE4Ljg4NSA1NC4yOTRsLTMuMzA2IDguODI1Yy00Ljg4NCAxMi44OTgtOS40MzMgMjQuMjYzLTEzLjY0NyAzNC4wOTVoLTQ5Ljc4N2E4NDE3LjI4OSA4NDE3LjI4OSAwIDAxLTIxLjAzMS02NC44MDkgMTI4OC42ODYgMTI4OC42ODYgMCAwMS0xOC44ODUtNjQuODEgMTk3Mi40NDQgMTk3Mi40NDQgMCAwMS0xOC4yNCA2NC44MSAyNTc5LjQxMiAyNTc5LjQxMiAwIDAxLTIwLjM4OCA2NC44MWgtNDkuNzg3Yy00LjY4Mi0xMC45MjYtOS43Mi0yMy43NDMtMTUuMTEtMzguNDUxbC0xLjYyOS00LjQ3Yy01LjI1OC0xNC41MjEtMTAuNjgtMzAuMTkyLTE2LjI2Ni00Ny4wMTRsLTIuNDA0LTcuMjhjLTYuNDM4LTE5LjYtMTMuMDItNDAuMzQ0LTE5Ljc0My02Mi4yMzRhMjk4OC43MDcgMjk4OC43MDcgMCAwMS0xOS41MjktNjYuMzExaDY3LjM4NXoiIGZpbGw9IiM0Mjg1RjQiIGZpbGwtcnVsZT0ibm9uemVybyIvPjwvc3ZnPg==");
}
`);var N="undefined"==typeof document?null:document,P=N?.fullscreenEnabled||!!N?.webkitFullscreenEnabled;function j(){return document.fullscreenElement?document.fullscreenElement:document.webkitFullscreenElement??null}var R=["skip-to-start","skip-to-end","step-forward","step-backward","pause","play","enter-fullscreen","exit-fullscreen","twizzle-tw"],O=class extends n.j8{derive(e){return{fullscreen:{enabled:P,icon:null===document.fullscreenElement?"enter-fullscreen":"exit-fullscreen",title:"Enter fullscreen"},"jump-to-start":{enabled:!e.coarseTimelineInfo.atStart,icon:"skip-to-start",title:"Restart"},"play-step-backwards":{enabled:!e.coarseTimelineInfo.atStart,icon:"step-backward",title:"Step backward"},"play-pause":{enabled:!(e.coarseTimelineInfo.atStart&&e.coarseTimelineInfo.atEnd),icon:e.coarseTimelineInfo.playing?"pause":"play",title:e.coarseTimelineInfo.playing?"Pause":"Play"},"play-step":{enabled:!e.coarseTimelineInfo.atEnd,icon:"step-forward",title:"Step forward"},"jump-to-end":{enabled:!e.coarseTimelineInfo.atEnd,icon:"skip-to-end",title:"Skip to End"},"twizzle-link":{enabled:!0,icon:"twizzle-tw",title:"View at Twizzle",hidden:"none"===e.viewerLink}}}},U={fullscreen:!0,"jump-to-start":!0,"play-step-backwards":!0,"play-pause":!0,"play-step":!0,"jump-to-end":!0,"twizzle-link":!0},B=class extends n.FD{constructor(e,t,i){super(),this.model=e,this.controller=t,this.defaultFullscreenElement=i}buttons=null;connectedCallback(){this.addCSS(C);let e={};for(let t in U){let i=new V;e[t]=i,i.htmlButton.addEventListener("click",()=>this.#w(t)),this.addElement(i)}this.buttons=e,this.model?.buttonAppearance.addFreshListener(this.update.bind(this)),this.model?.twistySceneModel.colorScheme.addFreshListener(this.updateColorScheme.bind(this))}#w(e){switch(e){case"fullscreen":this.onFullscreenButton();break;case"jump-to-start":this.controller?.jumpToStart({flash:!0});break;case"play-step-backwards":this.controller?.animationController.play({direction:-1,untilBoundary:"move"});break;case"play-pause":this.controller?.togglePlay();break;case"play-step":this.controller?.animationController.play({direction:1,untilBoundary:"move"});break;case"jump-to-end":this.controller?.jumpToEnd({flash:!0});break;case"twizzle-link":this.controller?.visitTwizzleLink();break;default:throw Error("Missing command")}}async onFullscreenButton(){if(!this.defaultFullscreenElement)throw Error("Attempted to go fullscreen without an element.");if(j()===this.defaultFullscreenElement)document.exitFullscreen?document.exitFullscreen():document.webkitExitFullscreen();else{var e;this.buttons?.fullscreen.setIcon("exit-fullscreen"),(e=await this.model?.twistySceneModel.fullscreenElement.get()??this.defaultFullscreenElement).requestFullscreen?e.requestFullscreen():e.webkitRequestFullscreen();let t=()=>{j()!==this.defaultFullscreenElement&&(this.buttons?.fullscreen.setIcon("enter-fullscreen"),window.removeEventListener("fullscreenchange",t))};window.addEventListener("fullscreenchange",t)}}async update(e){for(let t in U){let i=this.buttons[t],r=e[t];i.htmlButton.disabled=!r.enabled,i.htmlButton.title=r.title,i.setIcon(r.icon),i.hidden=!!r.hidden}}updateColorScheme(e){for(let t of Object.values(this.buttons??{}))t.updateColorScheme(e)}};n.qh.define("twisty-buttons",B);var V=class extends n.FD{htmlButton=document.createElement("button");updateColorScheme(e){this.contentWrapper.classList.toggle("dark-mode","dark"===e)}connectedCallback(){this.addCSS(E),this.addElement(this.htmlButton)}#y=new T(this,"svg-",R);setIcon(e){this.#y.setValue(e)}};n.qh.define("twisty-button",V);var F=new n.n5;F.replaceSync(`
:host {
  width: 384px;
  height: 16px;
  display: grid;
}

.wrapper {
  width: 100%;
  height: 100%;
  display: grid;
  overflow: hidden;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
  background: rgba(196, 196, 196, 0.75);
}

input:not(:disabled) {
  cursor: ew-resize;
}

.wrapper.dark-mode {
  background: #666666;
}
`);N?.addEventListener("mousedown",e=>{e.which},!0),N?.addEventListener("mouseup",e=>{e.which},!0);var q=0;function W(e){e.pageY}N?.addEventListener("mousedown",()=>{q++},!1),N?.addEventListener("mousemove",W,!1),N?.addEventListener("mouseenter",W,!1);var Q=class extends n.FD{constructor(e,t){super(),this.model=e,this.controller=t}async onDetailedTimelineInfo(e){let t=await this.inputElem();t.min=e.timeRange.start.toString(),t.max=e.timeRange.end.toString(),t.disabled=t.min===t.max,t.value=e.timestamp.toString()}async connectedCallback(){this.addCSS(F),this.addElement(await this.inputElem()),this.model?.twistySceneModel.colorScheme.addFreshListener(this.updateColorScheme.bind(this))}updateColorScheme(e){this.contentWrapper.classList.toggle("dark-mode","dark"===e)}#f=null;async inputElem(){return this.#f??=(async()=>{let e=document.createElement("input");return e.type="range",e.disabled=!0,this.model?.detailedTimelineInfo.addFreshListener(this.onDetailedTimelineInfo.bind(this)),e.addEventListener("input",this.onInput.bind(this)),e.addEventListener("keydown",this.onKeypress.bind(this)),e})()}async onInput(e){0;let t=await this.inputElem();await this.slowDown(e,t);let i=parseInt(t.value);this.model?.playingInfo.set({playing:!1}),this.model?.timestampRequest.set(i)}onKeypress(e){switch(e.key){case"ArrowLeft":case"ArrowRight":this.controller?.animationController.play({direction:"ArrowLeft"===e.key?-1:1,untilBoundary:"move"}),e.preventDefault();break;case" ":this.controller?.togglePlay(),e.preventDefault()}}async slowDown(e,t){}};n.qh.define("twisty-scrubber",Q);var H=null;async function Y(e,t){let[{PerspectiveCamera:i,Scene:r},a,s,l,o,d,u]=await Promise.all([n.u_,await e.puzzleLoader.get(),await e.visualizationStrategy.get(),await e.twistySceneModel.stickeringRequest.get(),await e.twistySceneModel.stickeringMaskRequest.get(),await e.legacyPosition.get(),await e.twistySceneModel.orbitCoordinates.get()]),c=t?.width??2048,h=t?.height??2048,p=c/h,m=H??=await (async()=>new i(20,p,.1,20))(),g=new r,w=new S(e,{scheduleRender:()=>{}},a,s);g.add(await w.twisty3DPuzzle()),await (0,n.Rv)(m,u);let y=(await (0,n.OJ)(c,h,g,m)).toDataURL(),f=await G(e);return{dataURL:y,download:async e=>{$(y,e??f)}}}async function G(e){let[t,i]=await Promise.all([e.puzzleID.get(),e.alg.get()]);return`[${t}]${0===i.alg.experimentalNumChildAlgNodes()?"":` ${i.alg.toString()}`}`}function $(e,t,i="png"){let r=document.createElement("a");r.href=e,r.download=`${t}.${i}`,r.click()}var Z=new n.n5;Z.replaceSync(`
:host {
  width: 384px;
  height: 256px;
  display: grid;

  -webkit-user-select: none;
  user-select: none;
}

.wrapper {
  display: grid;
  overflow: hidden;
  contain: size;
  grid-template-rows: 7fr minmax(1.5em, 0.5fr) minmax(2em, 1fr);
}

.wrapper > * {
  width: inherit;
  height: inherit;
  overflow: hidden;
}

.wrapper.controls-none {
  grid-template-rows: 7fr;
}

.wrapper.controls-none twisty-scrubber,
.wrapper.controls-none twisty-control-button-panel ,
.wrapper.controls-none twisty-scrubber,
.wrapper.controls-none twisty-buttons {
  display: none;
}

twisty-scrubber {
  background: rgba(196, 196, 196, 0.5);
}

.wrapper.checkered,
.wrapper.checkered-transparent {
  background-color: #EAEAEA;
  background-image: linear-gradient(45deg, #DDD 25%, transparent 25%, transparent 75%, #DDD 75%, #DDD),
    linear-gradient(45deg, #DDD 25%, transparent 25%, transparent 75%, #DDD 75%, #DDD);
  background-size: 32px 32px;
  background-position: 0 0, 16px 16px;
}

.wrapper.checkered-transparent {
  background-color: #F4F4F4;
  background-image: linear-gradient(45deg, #DDDDDD88 25%, transparent 25%, transparent 75%, #DDDDDD88 75%, #DDDDDD88),
    linear-gradient(45deg, #DDDDDD88 25%, transparent 25%, transparent 75%, #DDDDDD88 75%, #DDDDDD88);
}

.wrapper.dark-mode {
  background-color: #444;
  background-image: linear-gradient(45deg, #DDDDDD0b 25%, transparent 25%, transparent 75%, #DDDDDD0b 75%, #DDDDDD0b),
    linear-gradient(45deg, #DDDDDD0b 25%, transparent 25%, transparent 75%, #DDDDDD0b 75%, #DDDDDD0b);
}

.visualization-wrapper > * {
  width: 100%;
  height: 100%;
}

.error-elem {
  width: 100%;
  height: 100%;
  display: none;
  place-content: center;
  font-family: sans-serif;
  box-shadow: inset 0 0 2em rgb(255, 0, 0);
  color: red;
  text-shadow: 0 0 0.2em white;
  background: rgba(255, 255, 255, 0.25);
}

.wrapper.error .visualization-wrapper {
  display: none;
}

.wrapper.error .error-elem {
  display: grid;
}
`);var X=class extends n.nB{getDefaultValue(){return null}},J=class extends n.FB{getDefaultValue(){return null}derive(e){return"string"==typeof e?new URL(e,location.href):e}},_=class e{warnings;errors;constructor(e){this.warnings=Object.freeze(e?.warnings??[]),this.errors=Object.freeze(e?.errors??[]),Object.freeze(this)}add(t){return new e({warnings:this.warnings.concat(t?.warnings??[]),errors:this.errors.concat(t?.errors??[])})}log(){this.errors.length>0?console.error(`\u{1F6A8} ${this.errors[0]}`):this.warnings.length>0?console.warn(`\u26A0\uFE0F ${this.warnings[0]}`):console.info("\uD83D\uDE0E No issues!")}};function K(e){try{let t=o.BE.fromString(e),i=[];return t.toString()!==e&&i.push("Alg is non-canonical!"),{alg:t,issues:new _({warnings:i})}}catch(e){return{alg:new o.BE,issues:new _({errors:[`Malformed alg: ${e.toString()}`]})}}}var ee=class extends n.FB{getDefaultValue(){return{alg:new o.BE,issues:new _}}canReuseValue(e,t){return e.alg.isIdentical(t.alg)&&d(e.issues.warnings,t.issues.warnings)&&d(e.issues.errors,t.issues.errors)}async derive(e){return"string"==typeof e?K(e):{alg:e,issues:new _}}},et=class extends n.j8{derive(e){return e.kpuzzle.algToTransformation(e.setupAlg.alg)}},ei=class extends n.j8{derive(e){if(e.setupTransformation)return e.setupTransformation;switch(e.setupAnchor){case"start":return e.setupAlgTransformation;case"end":{let t=e.indexer.transformationAtIndex(e.indexer.numAnimatedLeaves()).invert();return e.setupAlgTransformation.applyTransformation(t)}default:throw Error("Unimplemented!")}}},er=class extends n.nB{getDefaultValue(){return{move:null,amount:0}}canReuseValue(e,t){return e.move===t.move&&e.amount===t.amount}},en=class extends n.j8{derive(e){return{patternIndex:e.currentMoveInfo.patternIndex,movesFinishing:e.currentMoveInfo.movesFinishing.map(e=>e.move),movesFinished:e.currentMoveInfo.movesFinished.map(e=>e.move)}}canReuseValue(e,t){return e.patternIndex===t.patternIndex&&u(e.movesFinishing,t.movesFinishing,(e,t)=>e.isIdentical(t))&&u(e.movesFinished,t.movesFinished,(e,t)=>e.isIdentical(t))}},ea=class extends n.j8{derive(e){function t(t){return e.detailedTimelineInfo.atEnd&&null!==e.catchUpMove.move&&t.currentMoves.push({move:e.catchUpMove.move,direction:-1,fraction:1-e.catchUpMove.amount,startTimestamp:-1,endTimestamp:-1}),t}if(e.indexer.currentMoveInfo)return t(e.indexer.currentMoveInfo(e.detailedTimelineInfo.timestamp));{let i=e.indexer.timestampToIndex(e.detailedTimelineInfo.timestamp),r={patternIndex:i,currentMoves:[],movesFinishing:[],movesFinished:[],movesStarting:[],latestStart:-1/0,earliestEnd:1/0};if(e.indexer.numAnimatedLeaves()>0){let n=e.indexer.getAnimLeaf(i)?.as(o.yU);if(!n)return t(r);let a=e.indexer.indexToMoveStartTimestamp(i),s=e.indexer.moveDuration(i),l=s?(e.detailedTimelineInfo.timestamp-a)/s:0,d=a+s,u={move:n,direction:1,fraction:l,startTimestamp:a,endTimestamp:d};0===l?r.movesStarting.push(u):1===l?r.movesFinishing.push(u):(r.currentMoves.push(u),r.latestStart=Math.max(r.latestStart,a),r.earliestEnd=Math.min(r.earliestEnd,d))}return t(r)}}},es=class extends n.j8{derive(e){let t=e.indexer.transformationAtIndex(e.currentLeavesSimplified.patternIndex);for(let i of(t=e.anchoredStart.applyTransformation(t),e.currentLeavesSimplified.movesFinishing))t=t.applyMove(i);for(let i of e.currentLeavesSimplified.movesFinished)t=t.applyMove(i);return t.toKPattern()}};function el(e){switch(Math.abs(e)){case 0:return 0;case 1:return 1e3;case 2:return 1500;default:return 2e3}}var eo=class extends o.wr{constructor(e=el){super(),this.durationForAmount=e}traverseAlg(e){let t=0;for(let i of e.childAlgNodes())t+=this.traverseAlgNode(i);return t}traverseGrouping(e){return e.amount*this.traverseAlg(e.alg)}traverseMove(e){return this.durationForAmount(e.amount)}traverseCommutator(e){return 2*(this.traverseAlg(e.A)+this.traverseAlg(e.B))}traverseConjugate(e){return 2*this.traverseAlg(e.A)+this.traverseAlg(e.B)}traversePause(e){return this.durationForAmount(1)}traverseNewline(e){return this.durationForAmount(1)}traverseLineComment(e){return this.durationForAmount(0)}},ed=class{constructor(e,t){this.kpuzzle=e,this.moves=new o.BE(t.experimentalExpand())}moves;durationFn=new eo(el);getAnimLeaf(e){return Array.from(this.moves.childAlgNodes())[e]}indexToMoveStartTimestamp(e){let t=new o.BE(Array.from(this.moves.childAlgNodes()).slice(0,e));return this.durationFn.traverseAlg(t)}timestampToIndex(e){let t,i=0;for(t=0;t<this.numAnimatedLeaves()&&!((i+=this.durationFn.traverseMove(this.getAnimLeaf(t)))>=e);t++);return t}patternAtIndex(e){return this.kpuzzle.defaultPattern().applyTransformation(this.transformationAtIndex(e))}transformationAtIndex(e){let t=this.kpuzzle.identityTransformation();for(let i of Array.from(this.moves.childAlgNodes()).slice(0,e))t=t.applyMove(i);return t}algDuration(){return this.durationFn.traverseAlg(this.moves)}numAnimatedLeaves(){return(0,a.No)(this.moves)}moveDuration(e){return this.durationFn.traverseMove(this.getAnimLeaf(e))}},eu={u:"y",l:"x",f:"z",r:"x",b:"z",d:"y",m:"x",e:"y",s:"z",x:"x",y:"y",z:"z"},ec=class extends o.wr{traverseAlg(e){let t=[];for(let i of e.childAlgNodes())t.push(this.traverseAlgNode(i));return Array.prototype.concat(...t)}traverseGroupingOnce(e){if(e.experimentalIsEmpty())return[];let t=[];for(let i of e.childAlgNodes()){if(!(i.is(o.yU)||i.is(o.uk)||i.is(o.no)))return this.traverseAlg(e);let r=i.as(o.yU);r&&t.push(r)}let i=el(t[0].amount);for(let a=0;a<t.length-1;a++){for(let i=1;i<t.length;i++){var r,n;if(r=t[a],n=t[i],eu[r.family[0].toLowerCase()]!==eu[n.family[0].toLowerCase()])return this.traverseAlg(e)}i=Math.max(i,el(t[a].amount))}let a=t.map(e=>({animLeafAlgNode:e,msUntilNext:0,duration:i}));return a[a.length-1].msUntilNext=i,a}traverseGrouping(e){let t=[],i=e.amount>0?e.alg:e.alg.invert();for(let r=0;r<Math.abs(e.amount);r++)t.push(this.traverseGroupingOnce(i));return Array.prototype.concat(...t)}traverseMove(e){let t=el(e.amount);return[{animLeafAlgNode:e,msUntilNext:t,duration:t}]}traverseCommutator(e){let t=[];for(let i of[e.A,e.B,e.A.invert(),e.B.invert()])t.push(this.traverseGroupingOnce(i));return Array.prototype.concat(...t)}traverseConjugate(e){let t=[];for(let i of[e.A,e.B,e.A.invert()])t.push(this.traverseGroupingOnce(i));return Array.prototype.concat(...t)}traversePause(e){if(e.experimentalNISSGrouping)return[];let t=el(1);return[{animLeafAlgNode:e,msUntilNext:t,duration:t}]}traverseNewline(e){return[]}traverseLineComment(e){return[]}},eh=(0,o.RU)(ec),ep={"y' y' U' E D R2 r2 F2 B2 U E D' R2 L2' z2 S2 U U D D S2 F2' B2":[{animLeaf:new o.yU("y",-1),start:0,end:1e3},{animLeaf:new o.yU("y",-1),start:1e3,end:2e3},{animLeaf:new o.yU("U",-1),start:1e3,end:1600},{animLeaf:new o.yU("E",1),start:1200,end:1800},{animLeaf:new o.yU("D"),start:1400,end:2e3},{animLeaf:new o.yU("R",2),start:2e3,end:3500},{animLeaf:new o.yU("r",2),start:2e3,end:3500},{animLeaf:new o.yU("F",2),start:3500,end:4200},{animLeaf:new o.yU("B",2),start:3800,end:4500},{animLeaf:new o.yU("U",1),start:4500,end:5500},{animLeaf:new o.yU("E",1),start:4500,end:5500},{animLeaf:new o.yU("D",-1),start:4500,end:5500},{animLeaf:new o.yU("R",2),start:5500,end:6500},{animLeaf:new o.yU("L",-2),start:5500,end:6500},{animLeaf:new o.yU("z",2),start:5500,end:6500},{animLeaf:new o.yU("S",2),start:6500,end:7500},{animLeaf:new o.yU("U"),start:7500,end:8e3},{animLeaf:new o.yU("D"),start:7750,end:8250},{animLeaf:new o.yU("U"),start:8e3,end:8500},{animLeaf:new o.yU("D"),start:8250,end:8750},{animLeaf:new o.yU("S",2),start:8750,end:9250},{animLeaf:new o.yU("F",-2),start:8750,end:1e4},{animLeaf:new o.yU("B",2),start:8750,end:1e4}],"M' R' U' D' M R":[{animLeaf:new o.yU("M",-1),start:0,end:1e3},{animLeaf:new o.yU("R",-1),start:0,end:1e3},{animLeaf:new o.yU("U",-1),start:1e3,end:2e3},{animLeaf:new o.yU("D",-1),start:1e3,end:2e3},{animLeaf:new o.yU("M"),start:2e3,end:3e3},{animLeaf:new o.yU("R"),start:2e3,end:3e3}],"U' E' r E r2' E r U E":[{animLeaf:new o.yU("U",-1),start:0,end:1e3},{animLeaf:new o.yU("E",-1),start:0,end:1e3},{animLeaf:new o.yU("r"),start:1e3,end:2500},{animLeaf:new o.yU("E"),start:2500,end:3500},{animLeaf:new o.yU("r",-2),start:3500,end:5e3},{animLeaf:new o.yU("E"),start:5e3,end:6e3},{animLeaf:new o.yU("r"),start:6e3,end:7e3},{animLeaf:new o.yU("U"),start:7e3,end:8e3},{animLeaf:new o.yU("E"),start:7e3,end:8e3}]},em=class{constructor(e,t){this.kpuzzle=e,this.animLeaves=ep[t.toString()]??function(e){let t=0;return eh(e).map(e=>{let i={animLeaf:e.animLeafAlgNode,start:t,end:t+e.duration};return t+=e.msUntilNext,i})}(t)}animLeaves;getAnimLeaf(e){return this.animLeaves[Math.min(e,this.animLeaves.length-1)]?.animLeaf??null}getAnimLeafWithRange(e){return this.animLeaves[Math.min(e,this.animLeaves.length-1)]}indexToMoveStartTimestamp(e){let t=0;return this.animLeaves.length>0&&(t=this.animLeaves[Math.min(e,this.animLeaves.length-1)].start),t}timestampToIndex(e){let t=0;for(t=0;t<this.animLeaves.length&&!(this.animLeaves[t].start>=e);t++);return Math.max(0,t-1)}timestampToPosition(e,t){let i=this.currentMoveInfo(e),r=t??this.kpuzzle.identityTransformation().toKPattern();for(let e of this.animLeaves.slice(0,i.patternIndex)){let t=e.animLeaf.as(o.yU);null!==t&&(r=r.applyMove(t))}return{pattern:r,movesInProgress:i.currentMoves}}currentMoveInfo(e){let t=1/0;for(let i of this.animLeaves)if(i.start<=e&&i.end>=e)t=Math.min(t,i.start);else if(i.start>e)break;let i=[],r=[],n=[],a=[],s=-1/0,l=1/0,d=0;for(let u of this.animLeaves)if(u.end<=t)d++;else if(u.start>e)break;else{let t=u.animLeaf.as(o.yU);if(null!==t){let o=(e-u.start)/(u.end-u.start),d=!1;o>1&&(o=1,d=!0);let c={move:t,direction:1,fraction:o,startTimestamp:u.start,endTimestamp:u.end};switch(o){case 0:r.push(c);break;case 1:d?a.push(c):n.push(c);break;default:i.push(c),s=Math.max(s,u.start),l=Math.min(l,u.end)}}}return{patternIndex:d,currentMoves:i,latestStart:s,earliestEnd:l,movesStarting:r,movesFinishing:n,movesFinished:a}}patternAtIndex(e,t){let i=t??this.kpuzzle.defaultPattern();for(let t=0;t<this.animLeaves.length&&t<e;t++){let e=this.animLeaves[t].animLeaf.as(o.yU);null!==e&&(i=i.applyMove(e))}return i}transformationAtIndex(e){let t=this.kpuzzle.identityTransformation();for(let i of this.animLeaves.slice(0,e)){let e=i.animLeaf.as(o.yU);null!==e&&(t=t.applyMove(e))}return t}algDuration(){let e=0;for(let t of this.animLeaves)e=Math.max(e,t.end);return e}numAnimatedLeaves(){return this.animLeaves.length}moveDuration(e){let t=this.getAnimLeafWithRange(e);return t.end-t.start}},eg=class{constructor(e,t,i,r,n=[]){this.moveCount=e,this.duration=t,this.forward=i,this.backward=r,this.children=n}},ew=class extends o.wr{constructor(e){super(),this.kpuzzle=e,this.identity=e.identityTransformation(),this.dummyLeaf=new eg(0,0,this.identity,this.identity,[])}identity;dummyLeaf;durationFn=new eo(el);cache={};traverseAlg(e){let t=0,i=0,r=this.identity,n=[];for(let a of e.childAlgNodes()){let e=this.traverseAlgNode(a);t+=e.moveCount,i+=e.duration,r=r===this.identity?e.forward:r.applyTransformation(e.forward),n.push(e)}return new eg(t,i,r,r.invert(),n)}traverseGrouping(e){let t=this.traverseAlg(e.alg);return this.mult(t,e.amount,[t])}traverseMove(e){let t=e.toString(),i=this.cache[t];if(i)return i;let r=this.kpuzzle.moveToTransformation(e);return i=new eg(1,this.durationFn.traverseAlgNode(e),r,r.invert()),this.cache[t]=i,i}traverseCommutator(e){let t=this.traverseAlg(e.A),i=this.traverseAlg(e.B),r=t.forward.applyTransformation(i.forward),n=t.backward.applyTransformation(i.backward),a=r.applyTransformation(n),s=new eg(2*(t.moveCount+i.moveCount),2*(t.duration+i.duration),a,a.invert(),[t,i]);return this.mult(s,1,[s,t,i])}traverseConjugate(e){let t=this.traverseAlg(e.A),i=this.traverseAlg(e.B),r=t.forward.applyTransformation(i.forward).applyTransformation(t.backward),n=new eg(2*t.moveCount+i.moveCount,2*t.duration+i.duration,r,r.invert(),[t,i]);return this.mult(n,1,[n,t,i])}traversePause(e){return e.experimentalNISSGrouping?this.dummyLeaf:new eg(1,this.durationFn.traverseAlgNode(e),this.identity,this.identity)}traverseNewline(e){return this.dummyLeaf}traverseLineComment(e){return this.dummyLeaf}mult(e,t,i){let r=Math.abs(t),n=e.forward.selfMultiply(t);return new eg(e.moveCount*r,e.duration*r,n,n.invert(),i)}},ey=class{constructor(e,t){this.apd=e,this.back=t}},ef=class extends o.Yp{constructor(e,t,i){super(),this.kpuzzle=e,this.algOrAlgNode=t,this.apd=i,this.i=-1,this.dur=-1,this.goali=-1,this.goaldur=-1,this.move=void 0,this.back=!1,this.moveDuration=0,this.st=this.kpuzzle.identityTransformation(),this.root=new ey(this.apd,!1)}move;moveDuration;back;st;root;i;dur;goali;goaldur;moveByIndex(e){return this.i>=0&&this.i===e?void 0!==this.move:this.dosearch(e,1/0)}moveByDuration(e){return this.dur>=0&&this.dur<e&&this.dur+this.moveDuration>=e?void 0!==this.move:this.dosearch(1/0,e)}dosearch(e,t){return this.goali=e,this.goaldur=t,this.i=0,this.dur=0,this.move=void 0,this.moveDuration=0,this.back=!1,this.st=this.kpuzzle.identityTransformation(),this.algOrAlgNode.is(o.BE)?this.traverseAlg(this.algOrAlgNode,this.root):this.traverseAlgNode(this.algOrAlgNode,this.root)}traverseAlg(e,t){if(!this.firstcheck(t))return!1;let i=t.back?e.experimentalNumChildAlgNodes()-1:0;for(let r of(0,o.a9)(e.childAlgNodes(),t.back?-1:1)){if(this.traverseAlgNode(r,new ey(t.apd.children[i],t.back)))return!0;i+=t.back?-1:1}return!1}traverseGrouping(e,t){if(!this.firstcheck(t))return!1;let i=this.domult(t,e.amount);return this.traverseAlg(e.alg,new ey(t.apd.children[0],i))}traverseMove(e,t){return!!this.firstcheck(t)&&(this.move=e,this.moveDuration=t.apd.duration,this.back=t.back,!0)}traverseCommutator(e,t){if(!this.firstcheck(t))return!1;let i=this.domult(t,1);return i?this.traverseAlg(e.B,new ey(t.apd.children[2],!i))||this.traverseAlg(e.A,new ey(t.apd.children[1],!i))||this.traverseAlg(e.B,new ey(t.apd.children[2],i))||this.traverseAlg(e.A,new ey(t.apd.children[1],i)):this.traverseAlg(e.A,new ey(t.apd.children[1],i))||this.traverseAlg(e.B,new ey(t.apd.children[2],i))||this.traverseAlg(e.A,new ey(t.apd.children[1],!i))||this.traverseAlg(e.B,new ey(t.apd.children[2],!i))}traverseConjugate(e,t){if(!this.firstcheck(t))return!1;let i=this.domult(t,1);return i?this.traverseAlg(e.A,new ey(t.apd.children[1],!i))||this.traverseAlg(e.B,new ey(t.apd.children[2],i))||this.traverseAlg(e.A,new ey(t.apd.children[1],i)):this.traverseAlg(e.A,new ey(t.apd.children[1],i))||this.traverseAlg(e.B,new ey(t.apd.children[2],i))||this.traverseAlg(e.A,new ey(t.apd.children[1],!i))}traversePause(e,t){return!!this.firstcheck(t)&&(this.move=e,this.moveDuration=t.apd.duration,this.back=t.back,!0)}traverseNewline(e,t){return!1}traverseLineComment(e,t){return!1}firstcheck(e){return!(e.apd.moveCount+this.i<=this.goali)||!(e.apd.duration+this.dur<this.goaldur)||this.keepgoing(e)}domult(e,t){let i=e.back;if(0===t)return i;t<0&&(i=!i,t=-t);let r=e.apd.children[0],n=Math.min(Math.floor((this.goali-this.i)/r.moveCount),Math.ceil((this.goaldur-this.dur)/r.duration-1));return n>0&&this.keepgoing(new ey(r,i),n),i}keepgoing(e,t=1){return this.i+=t*e.apd.moveCount,this.dur+=t*e.apd.duration,1!==t?e.back?this.st=this.st.applyTransformation(e.apd.backward.selfMultiply(t)):this.st=this.st.applyTransformation(e.apd.forward.selfMultiply(t)):e.back?this.st=this.st.applyTransformation(e.apd.backward):this.st=this.st.applyTransformation(e.apd.forward),!1}},ev=class extends o.wr{traverseAlg(e){let t=e.experimentalNumChildAlgNodes();if(t<16)return e;var i=Math.ceil(Math.sqrt(t));let r=new o.MF,n=new o.MF;for(let t of e.childAlgNodes())n.push(t),n.experimentalNumAlgNodes()>=i&&(r.push(new o.aU(n.toAlg())),n.reset());return r.push(new o.aU(n.toAlg())),r.toAlg()}traverseGrouping(e){return new o.aU(this.traverseAlg(e.alg),e.amount)}traverseMove(e){return e}traverseCommutator(e){return new o.NG(this.traverseAlg(e.A),this.traverseAlg(e.B))}traverseConjugate(e){return new o.NG(this.traverseAlg(e.A),this.traverseAlg(e.B))}traversePause(e){return e}traverseNewline(e){return e}traverseLineComment(e){return e}},eM=(0,o.RU)(ev),ex=class{constructor(e,t){this.kpuzzle=e;let i=new ew(this.kpuzzle),r=eM(t);this.decoration=i.traverseAlg(r),this.walker=new ef(this.kpuzzle,r,this.decoration)}decoration;walker;getAnimLeaf(e){if(this.walker.moveByIndex(e)){if(!this.walker.move)throw Error("`this.walker.mv` missing");let e=this.walker.move;return this.walker.back?e.invert():e}return null}indexToMoveStartTimestamp(e){if(this.walker.moveByIndex(e)||this.walker.i===e)return this.walker.dur;throw Error(`Out of algorithm: index ${e}`)}indexToMovesInProgress(e){if(this.walker.moveByIndex(e)||this.walker.i===e)return this.walker.dur;throw Error(`Out of algorithm: index ${e}`)}patternAtIndex(e,t){return this.walker.moveByIndex(e),(t??this.kpuzzle.defaultPattern()).applyTransformation(this.walker.st)}transformationAtIndex(e){return this.walker.moveByIndex(e),this.walker.st}numAnimatedLeaves(){return this.decoration.moveCount}timestampToIndex(e){return this.walker.moveByDuration(e),this.walker.i}algDuration(){return this.decoration.duration}moveDuration(e){return this.walker.moveByIndex(e),this.walker.moveDuration}},ez=class extends n.j8{derive(e){switch(e.indexerConstructorRequest){case"auto":if(1024>=(0,a.AY)(e.alg.alg)&&"3x3x3"===e.puzzle&&"Cube3D"===e.visualizationStrategy)return em;return ex;case"tree":return ex;case"simple":return ed;case"simultaneous":return em;default:throw Error("Invalid indexer request!")}}},eL=class extends n.nB{getDefaultValue(){return"auto"}},eb=class extends n.j8{derive(e){return new e.indexerConstructor(e.kpuzzle,e.algWithIssues.alg)}},eD=class extends n.j8{derive(e){return{pattern:e.currentPattern,movesInProgress:e.currentMoveInfo.currentMoves}}},ek=class extends n.j8{async derive(e){try{return e.kpuzzle.algToTransformation(e.algWithIssues.alg),e.algWithIssues}catch(e){return{alg:new o.BE,issues:new _({errors:[`Invalid alg for puzzle: ${e.toString()}`]})}}}},eT=class extends n.nB{getDefaultValue(){return"start"}},eI=class extends n.nB{getDefaultValue(){return null}},eS=class extends n.j8{async derive(e){return e.puzzleLoader.kpuzzle()}},eA=class extends n.nB{getDefaultValue(){return n.GY}},eC=class extends n.j8{async derive(e){return e.puzzleLoader.id}},eE=class extends n.nB{getDefaultValue(){return n.GY}},eN=class extends n.j8{derive(e){if(e.puzzleIDRequest&&e.puzzleIDRequest!==n.GY){let t=s.SB[e.puzzleIDRequest];return t||this.userVisibleErrorTracker.set({errors:[`Invalid puzzle ID: ${e.puzzleIDRequest}`]}),t}return e.puzzleDescriptionRequest&&e.puzzleDescriptionRequest!==n.GY?(0,l.uy)(e.puzzleDescriptionRequest):s.P$}},eP=class extends n.j8{derive(e){return{playing:e.playingInfo.playing,atStart:e.detailedTimelineInfo.atStart,atEnd:e.detailedTimelineInfo.atEnd}}canReuseValue(e,t){return e.playing===t.playing&&e.atStart===t.atStart&&e.atEnd===t.atEnd}},ej=class extends n.j8{derive(e){let t=this.#v(e),i=!1,r=!1;return t>=e.timeRange.end&&(r=!0,t=Math.min(e.timeRange.end,t)),t<=e.timeRange.start&&(i=!0,t=Math.max(e.timeRange.start,t)),{timestamp:t,timeRange:e.timeRange,atStart:i,atEnd:r}}#v(e){switch(e.timestampRequest){case"auto":return"start"===e.setupAnchor&&e.setupAlg.alg.experimentalIsEmpty()?e.timeRange.end:e.timeRange.start;case"start":return e.timeRange.start;case"end":return e.timeRange.end;case"anchor":return"start"===e.setupAnchor?e.timeRange.start:e.timeRange.end;case"opposite-anchor":return"start"===e.setupAnchor?e.timeRange.end:e.timeRange.start;default:return e.timestampRequest}}canReuseValue(e,t){return e.timestamp===t.timestamp&&e.timeRange.start===t.timeRange.start&&e.timeRange.end===t.timeRange.end&&e.atStart===t.atStart&&e.atEnd===t.atEnd}},eR=class extends n.FB{async getDefaultValue(){return{direction:1,playing:!1,untilBoundary:"entire-timeline",loop:!1}}async derive(e,t){let i=Object.assign({},await t);return Object.assign(i,e),i}canReuseValue(e,t){return e.direction===t.direction&&e.playing===t.playing&&e.untilBoundary===t.untilBoundary&&e.loop===t.loop}},eO=class extends n.FB{getDefaultValue(){return 1}derive(e){return e<0?1:e}},eU={auto:!0,start:!0,end:!0,anchor:!0,"opposite-anchor":!0},eB=class extends n.nB{getDefaultValue(){return"auto"}set(e){let t=this.get();super.set((async()=>this.validInput(await e)?e:t)())}validInput(e){return"number"==typeof e||!!eU[e]}},eV=class extends n.nB{getDefaultValue(){return"auto"}},eF=class extends n.j8{derive(e){return{start:0,end:e.indexer.algDuration()}}},eq=class extends n.nB{getDefaultValue(){return"auto"}},eW=class extends n.nB{getDefaultValue(){return"auto"}},eQ=class extends n.j8{derive(e){switch(e.puzzleID){case"clock":case"square1":case"redi_cube":case"melindas2x2x2x2":case"tri_quad":case"loopover":return"2D";case"3x3x3":switch(e.visualizationRequest){case"auto":case"3D":return"Cube3D";default:return e.visualizationRequest}default:switch(e.visualizationRequest){case"auto":case"3D":return"PG3D";case"experimental-2D-LL":case"experimental-2D-LL-face":if(["2x2x2","4x4x4","megaminx"].includes(e.puzzleID))return"experimental-2D-LL";return"2D";default:return e.visualizationRequest}}}},eH=class extends n.nB{getDefaultValue(){return"auto"}},eY=class extends n.nB{getDefaultValue(){return"auto"}},eG=class extends n.nB{getDefaultValue(){return"auto"}},e$=null;async function eZ(){return e$??=new(await n.u_).TextureLoader}var eX=class extends n.j8{async derive(e){let{spriteURL:t}=e;return null===t?null:new Promise(async(e,i)=>{let r=()=>{console.warn("Could not load sprite:",t.toString()),e(null)};try{(await eZ()).load(t.toString(),e,r,r)}catch(e){r()}})}},eJ={facelets:["regular","regular","regular","regular","regular"]};async function e_(e){let{definition:t}=await e.kpuzzle(),i={orbits:{}};for(let e of t.orbits)i.orbits[e.orbitName]={pieces:Array(e.numPieces).fill(eJ)};return i}var eK=class extends n.j8{getDefaultValue(){return{orbits:{}}}async derive(e){return e.stickeringMaskRequest?e.stickeringMaskRequest:"picture"===e.stickeringRequest?{specialBehaviour:"picture",orbits:{}}:e.puzzleLoader.stickeringMask?.(e.stickeringRequest??"full")??e_(e.puzzleLoader)}},e0={"-":"Regular",D:"Dim",I:"Ignored",X:"Invisible",O:"IgnoreNonPrimary",P:"PermuteNonPrimary",o:"Ignoriented","?":"OrientationWithoutPermutation",M:"Mystery","@":"Regular"},e1=class extends n.FB{getDefaultValue(){return null}derive(e){if(null===e)return null;if("string"!=typeof e)return e;let t={orbits:{}};for(let i of e.split(",")){let[e,r,...n]=i.split(":");if(n.length>0)throw Error(`Invalid serialized orbit stickering mask (too many colons): \`${i}\``);let a=[];for(let i of(t.orbits[e]={pieces:a},r)){let e=e0[i];a.push((0,l.uP)(e))}}return t}},e2=class extends n.nB{getDefaultValue(){return null}},e3=class extends n.nB{getDefaultValue(){return"auto"}},e5=class extends n.nB{getDefaultValue(){return{}}},e4=class extends n.nB{getDefaultValue(){return"auto"}},e8=class extends n.nB{getDefaultValue(){return"auto"}},e6=class extends n.j8{derive(e){return"dark"===e.colorSchemeRequest?"dark":"light"}},e9=class extends n.nB{getDefaultValue(){return"auto"}},e7=class extends n.nB{getDefaultValue(){return null}},te=class extends n.nB{getDefaultValue(){return 35}};function tt(e,t){return e.latitude===t.latitude&&e.longitude===t.longitude&&e.distance===t.distance}var ti=class extends n.FB{getDefaultValue(){return"auto"}canReuseValue(e,t){return e===t||tt(e,t)}async derive(e,t){if("auto"===e)return"auto";let i=await t;"auto"===i&&(i={});let r=Object.assign({},i);return Object.assign(r,e),void 0!==r.latitude&&(r.latitude=Math.min(Math.max(r.latitude,-90),90)),void 0!==r.longitude&&(r.longitude=c(r.longitude,180,-180)),r}},tr=class extends n.j8{canReuseValue(e,t){return tt(e,t)}async derive(e){if("auto"===e.orbitCoordinatesRequest)return td(e.puzzleID,e.strategy);let t=Object.assign(Object.assign({},td(e.puzzleID,e.strategy),e.orbitCoordinatesRequest));if(Math.abs(t.latitude)<=e.latitudeLimit)return t;{let{latitude:i,longitude:r,distance:n}=t;return{latitude:e.latitudeLimit*Math.sign(i),longitude:r,distance:n}}}},tn={latitude:31.717474411461005,longitude:0,distance:5.877852522924731},ta={latitude:35,longitude:30,distance:6},ts={latitude:35,longitude:30,distance:6.25},tl={latitude:Math.atan(.5)*n.cg,longitude:0,distance:6.7},to={latitude:26.56505117707799,longitude:0,distance:6};function td(e,t){if("x"===e[1])if("Cube3D"===t)return ta;else return ts;switch(e){case"megaminx":case"gigaminx":return tl;case"pyraminx":case"master_tetraminx":return to;case"skewb":return ts;default:return tn}}var tu=class{constructor(e){this.twistyPlayerModel=e,this.orbitCoordinates=new tr({orbitCoordinatesRequest:this.orbitCoordinatesRequest,latitudeLimit:this.latitudeLimit,puzzleID:e.puzzleID,strategy:e.visualizationStrategy}),this.stickeringMask=new eK({stickeringMaskRequest:this.stickeringMaskRequest,stickeringRequest:this.stickeringRequest,puzzleLoader:e.puzzleLoader})}background=new e8;colorSchemeRequest=new e9;dragInput=new e3;foundationDisplay=new eY;foundationStickerSpriteURL=new J;fullscreenElement=new e7;hintFacelet=new n.rk;hintStickerSpriteURL=new J;initialHintFaceletsAnimation=new eG;latitudeLimit=new te;movePressInput=new e4;movePressCancelOptions=new e5;orbitCoordinatesRequest=new ti;stickeringMaskRequest=new e1;stickeringRequest=new e2;faceletScale=new eH;colorScheme=new e6({colorSchemeRequest:this.colorSchemeRequest});foundationStickerSprite=new eX({spriteURL:this.foundationStickerSpriteURL});hintStickerSprite=new eX({spriteURL:this.hintStickerSpriteURL});orbitCoordinates;stickeringMask},tc={errors:[]},th=class extends n.nB{getDefaultValue(){return tc}reset(){this.set(this.getDefaultValue())}canReuseValue(e,t){return d(e.errors,t.errors)}},tp=class{userVisibleErrorTracker=new th;alg=new ee;backView=new eV;controlPanel=new w;catchUpMove=new er;indexerConstructorRequest=new eL;playingInfo=new eR;puzzleDescriptionRequest=new eA;puzzleIDRequest=new eE;setupAnchor=new eT;setupAlg=new ee;setupTransformation=new eI;tempoScale=new eO;timestampRequest=new eB;viewerLink=new eq;visualizationFormat=new eW;title=new X;videoURL=new J;competitionID=new X;puzzleLoader=new eN({puzzleIDRequest:this.puzzleIDRequest,puzzleDescriptionRequest:this.puzzleDescriptionRequest},this.userVisibleErrorTracker);kpuzzle=new eS({puzzleLoader:this.puzzleLoader});puzzleID=new eC({puzzleLoader:this.puzzleLoader});puzzleAlg=new ek({algWithIssues:this.alg,kpuzzle:this.kpuzzle});puzzleSetupAlg=new ek({algWithIssues:this.setupAlg,kpuzzle:this.kpuzzle});visualizationStrategy=new eQ({visualizationRequest:this.visualizationFormat,puzzleID:this.puzzleID});indexerConstructor=new ez({alg:this.alg,puzzle:this.puzzleID,visualizationStrategy:this.visualizationStrategy,indexerConstructorRequest:this.indexerConstructorRequest});setupAlgTransformation=new et({setupAlg:this.puzzleSetupAlg,kpuzzle:this.kpuzzle});indexer=new eb({indexerConstructor:this.indexerConstructor,algWithIssues:this.puzzleAlg,kpuzzle:this.kpuzzle});anchorTransformation=new ei({setupTransformation:this.setupTransformation,setupAnchor:this.setupAnchor,setupAlgTransformation:this.setupAlgTransformation,indexer:this.indexer});timeRange=new eF({indexer:this.indexer});detailedTimelineInfo=new ej({timestampRequest:this.timestampRequest,timeRange:this.timeRange,setupAnchor:this.setupAnchor,setupAlg:this.setupAlg});coarseTimelineInfo=new eP({detailedTimelineInfo:this.detailedTimelineInfo,playingInfo:this.playingInfo});currentMoveInfo=new ea({indexer:this.indexer,detailedTimelineInfo:this.detailedTimelineInfo,catchUpMove:this.catchUpMove});buttonAppearance=new O({coarseTimelineInfo:this.coarseTimelineInfo,viewerLink:this.viewerLink});currentLeavesSimplified=new en({currentMoveInfo:this.currentMoveInfo});currentPattern=new es({anchoredStart:this.anchorTransformation,currentLeavesSimplified:this.currentLeavesSimplified,indexer:this.indexer});legacyPosition=new eD({currentMoveInfo:this.currentMoveInfo,currentPattern:this.currentPattern});twistySceneModel=new tu(this);async twizzleLink(){let[e,t,i,r,a,s,l,o]=await Promise.all([this.viewerLink.get(),this.puzzleID.get(),this.puzzleDescriptionRequest.get(),this.alg.get(),this.setupAlg.get(),this.setupAnchor.get(),this.twistySceneModel.stickeringRequest.get(),this.twistySceneModel.twistyPlayerModel.title.get()]),d="experimental-twizzle-explorer"===e,u=new URL(`https://alpha.twizzle.net/${d?"explore":"edit"}/`);return r.alg.experimentalIsEmpty()||u.searchParams.set("alg",r.alg.toString()),a.alg.experimentalIsEmpty()||u.searchParams.set("setup-alg",a.alg.toString()),"start"!==s&&u.searchParams.set("setup-anchor",s),"full"!==l&&null!==l&&u.searchParams.set("experimental-stickering",l),d&&i!==n.GY?u.searchParams.set("puzzle-description",i):"3x3x3"!==t&&u.searchParams.set("puzzle",t),o&&u.searchParams.set("title",o),u.toString()}experimentalAddAlgLeaf(e,t){let i=e.as(o.yU);i?this.experimentalAddMove(i,t):this.alg.set((async()=>{let t=(await this.alg.get()).alg.concat(new o.BE([e]));return this.timestampRequest.set("end"),t})())}experimentalAddMove(e,t){let i="string"==typeof e?new o.yU(e):e;this.alg.set((async()=>{let[{alg:e},r]=await Promise.all([this.alg.get(),this.puzzleLoader.get()]),n=(0,o.SL)(e,i,{...t,...await (0,l.U9)(r)});return this.timestampRequest.set("end"),this.catchUpMove.set({move:i,amount:0}),n})())}experimentalRemoveFinalChild(){this.alg.set((async()=>{let e=(await this.alg.get()).alg,t=Array.from(e.childAlgNodes()),[i]=t.splice(-1);if(!i)return e;this.timestampRequest.set("end");let r=i.as(o.yU);return r&&this.catchUpMove.set({move:r.invert(),amount:0}),new o.BE(t)})())}};function tm(e){return Error(`Cannot get \`.${e}\` directly from a \`TwistyPlayer\`.`)}var tg=class extends n.FD{experimentalModel=new tp;set alg(e){this.experimentalModel.alg.set(e)}get alg(){throw tm("alg")}set experimentalSetupAlg(e){this.experimentalModel.setupAlg.set(e)}get experimentalSetupAlg(){throw tm("setup")}set experimentalSetupAnchor(e){this.experimentalModel.setupAnchor.set(e)}get experimentalSetupAnchor(){throw tm("anchor")}set puzzle(e){this.experimentalModel.puzzleIDRequest.set(e)}get puzzle(){throw tm("puzzle")}set experimentalPuzzleDescription(e){this.experimentalModel.puzzleDescriptionRequest.set(e)}get experimentalPuzzleDescription(){throw tm("experimentalPuzzleDescription")}set timestamp(e){this.experimentalModel.timestampRequest.set(e)}get timestamp(){throw tm("timestamp")}set hintFacelets(e){this.experimentalModel.twistySceneModel.hintFacelet.set(e)}get hintFacelets(){throw tm("hintFacelets")}set experimentalStickering(e){this.experimentalModel.twistySceneModel.stickeringRequest.set(e)}get experimentalStickering(){throw tm("experimentalStickering")}set experimentalStickeringMaskOrbits(e){this.experimentalModel.twistySceneModel.stickeringMaskRequest.set(e)}get experimentalStickeringMaskOrbits(){throw tm("experimentalStickeringMaskOrbits")}set experimentalFaceletScale(e){this.experimentalModel.twistySceneModel.faceletScale.set(e)}get experimentalFaceletScale(){throw tm("experimentalFaceletScale")}set backView(e){this.experimentalModel.backView.set(e)}get backView(){throw tm("backView")}set background(e){this.experimentalModel.twistySceneModel.background.set(e)}get background(){throw tm("background")}set colorScheme(e){this.experimentalModel.twistySceneModel.colorSchemeRequest.set(e)}get colorScheme(){throw tm("colorScheme")}set controlPanel(e){this.experimentalModel.controlPanel.set(e)}get controlPanel(){throw tm("controlPanel")}set visualization(e){this.experimentalModel.visualizationFormat.set(e)}get visualization(){throw tm("visualization")}set experimentalTitle(e){this.experimentalModel.title.set(e)}get experimentalTitle(){throw tm("experimentalTitle")}set experimentalVideoURL(e){this.experimentalModel.videoURL.set(e)}get experimentalVideoURL(){throw tm("experimentalVideoURL")}set experimentalCompetitionID(e){this.experimentalModel.competitionID.set(e)}get experimentalCompetitionID(){throw tm("experimentalCompetitionID")}set viewerLink(e){this.experimentalModel.viewerLink.set(e)}get viewerLink(){throw tm("viewerLink")}set experimentalMovePressInput(e){this.experimentalModel.twistySceneModel.movePressInput.set(e)}get experimentalMovePressInput(){throw tm("experimentalMovePressInput")}set experimentalMovePressCancelOptions(e){this.experimentalModel.twistySceneModel.movePressCancelOptions.set(e)}get experimentalMovePressCancelOptions(){throw tm("experimentalMovePressCancelOptions")}set cameraLatitude(e){this.experimentalModel.twistySceneModel.orbitCoordinatesRequest.set({latitude:e})}get cameraLatitude(){throw tm("cameraLatitude")}set cameraLongitude(e){this.experimentalModel.twistySceneModel.orbitCoordinatesRequest.set({longitude:e})}get cameraLongitude(){throw tm("cameraLongitude")}set cameraDistance(e){this.experimentalModel.twistySceneModel.orbitCoordinatesRequest.set({distance:e})}get cameraDistance(){throw tm("cameraDistance")}set cameraLatitudeLimit(e){this.experimentalModel.twistySceneModel.latitudeLimit.set(e)}get cameraLatitudeLimit(){throw tm("cameraLatitudeLimit")}set indexer(e){this.experimentalModel.indexerConstructorRequest.set(e)}get indexer(){throw tm("indexer")}set tempoScale(e){this.experimentalModel.tempoScale.set(e)}get tempoScale(){throw tm("tempoScale")}set experimentalSprite(e){this.experimentalModel.twistySceneModel.foundationStickerSpriteURL.set(e)}get experimentalSprite(){throw tm("experimentalSprite")}set experimentalHintSprite(e){this.experimentalModel.twistySceneModel.hintStickerSpriteURL.set(e)}get experimentalHintSprite(){throw tm("experimentalHintSprite")}set fullscreenElement(e){this.experimentalModel.twistySceneModel.fullscreenElement.set(e)}get fullscreenElement(){throw tm("fullscreenElement")}set experimentalInitialHintFaceletsAnimation(e){this.experimentalModel.twistySceneModel.initialHintFaceletsAnimation.set(e)}get experimentalInitialHintFaceletsAnimation(){throw tm("experimentalInitialHintFaceletsAnimation")}set experimentalDragInput(e){this.experimentalModel.twistySceneModel.dragInput.set(e)}get experimentalDragInput(){throw tm("experimentalDragInput")}experimentalGet=new tw(this.experimentalModel)},tw=class{constructor(e){this.model=e}async alg(){return(await this.model.alg.get()).alg}async setupAlg(){return(await this.model.setupAlg.get()).alg}puzzleID(){return this.model.puzzleID.get()}async timestamp(){return(await this.model.detailedTimelineInfo.get()).timestamp}},ty="data-",tf={alg:"alg","experimental-setup-alg":"experimentalSetupAlg","experimental-setup-anchor":"experimentalSetupAnchor",puzzle:"puzzle","experimental-puzzle-description":"experimentalPuzzleDescription",visualization:"visualization","hint-facelets":"hintFacelets","experimental-stickering":"experimentalStickering","experimental-stickering-mask-orbits":"experimentalStickeringMaskOrbits",background:"background","color-scheme":"colorScheme","control-panel":"controlPanel","back-view":"backView","experimental-initial-hint-facelets-animation":"experimentalInitialHintFaceletsAnimation","viewer-link":"viewerLink","experimental-move-press-input":"experimentalMovePressInput","experimental-drag-input":"experimentalDragInput","experimental-title":"experimentalTitle","experimental-video-url":"experimentalVideoURL","experimental-competition-id":"experimentalCompetitionID","camera-latitude":"cameraLatitude","camera-longitude":"cameraLongitude","camera-distance":"cameraDistance","camera-latitude-limit":"cameraLatitudeLimit","tempo-scale":"tempoScale","experimental-sprite":"experimentalSprite","experimental-hint-sprite":"experimentalHintSprite"},tv=Object.fromEntries(Object.values(tf).map(e=>[e,!0])),tM={experimentalMovePressCancelOptions:!0},tx=Symbol("intersectedCallback"),tz=class extends tg{controller=new m(this.experimentalModel,this);buttons;experimentalCanvasClickCallback=()=>{};constructor(e={}){for(let[t,i]of(super(),Object.entries(e))){if(!(tv[t]||tM[t])){console.warn(`Invalid config passed to TwistyPlayer: ${t}`);break}this[t]=i}}#M=new T(this,"controls-",["auto"].concat(Object.keys(g)));#x=document.createElement("div");#z=document.createElement("div");#L=!1;async connectedCallback(){this.addCSS(Z),(r??=new IntersectionObserver((e,t)=>{for(let i of e)i.isIntersecting&&i.intersectionRect.height>0&&(i.target[tx](),t.unobserve(i.target))})).observe(this)}async [tx](){if(this.#L)return;this.#L=!0,this.addElement(this.#x).classList.add("visualization-wrapper"),this.addElement(this.#z).classList.add("error-elem"),this.#z.textContent="Error",this.experimentalModel.userVisibleErrorTracker.addFreshListener(e=>{let t=e.errors[0]??null;this.contentWrapper.classList.toggle("error",!!t),t&&(this.#z.textContent=t)});let e=new Q(this.experimentalModel,this.controller);this.contentWrapper.appendChild(e),this.buttons=new B(this.experimentalModel,this.controller,this),this.contentWrapper.appendChild(this.buttons),this.experimentalModel.twistySceneModel.background.addFreshListener(e=>{this.contentWrapper.classList.toggle("checkered",["auto","checkered"].includes(e)),this.contentWrapper.classList.toggle("checkered-transparent","checkered-transparent"===e)}),this.experimentalModel.twistySceneModel.colorScheme.addFreshListener(e=>{this.contentWrapper.classList.toggle("dark-mode",["dark"].includes(e))}),this.experimentalModel.controlPanel.addFreshListener(e=>{this.#M.setValue(e)}),this.experimentalModel.visualizationStrategy.addFreshListener(this.#b.bind(this)),this.experimentalModel.puzzleID.addFreshListener(this.flash.bind(this))}#D="auto";experimentalSetFlashLevel(e){this.#D=e}flash(){"auto"===this.#D&&this.#k?.animate([{opacity:.25},{opacity:1}],{duration:250,easing:"ease-out"})}#k=null;#T=new I;#I=null;#b(e){if(e!==this.#I){let t;switch(this.#k?.remove(),this.#k?.disconnect(),e){case"2D":case"experimental-2D-LL":case"experimental-2D-LL-face":t=new k(this.experimentalModel.twistySceneModel,e);break;case"Cube3D":case"PG3D":t=new A(this.experimentalModel),this.#T.handleNewValue(t);break;default:throw Error("Invalid visualization")}this.#x.appendChild(t),this.#k=t,this.#I=e}}async experimentalCurrentVantages(){this.connectedCallback();let e=this.#k;return e instanceof A?e.experimentalVantages():[]}async experimentalCurrentCanvases(){let e=await this.experimentalCurrentVantages(),t=[];for(let i of e)t.push((await i.canvasInfo()).canvas);return t}async experimentalCurrentThreeJSPuzzleObject(e){this.connectedCallback();let t=await this.#T.promise,i=await t.experimentalTwisty3DPuzzleWrapper(),r=i.twisty3DPuzzle(),a=(async()=>{await r,await new Promise(e=>setTimeout(e,0))})();if(e){let t=new n.mN(async()=>{});i.addEventListener("render-scheduled",async()=>{t.requestIsPending()||(t.requestAnimFrame(),await a,e())})}return r}jumpToStart(e){this.controller.jumpToStart(e)}jumpToEnd(e){this.controller.jumpToEnd(e)}play(){this.controller.togglePlay(!0)}pause(){this.controller.togglePlay(!1)}togglePlay(e){this.controller.togglePlay(e)}experimentalAddMove(e,t){this.experimentalModel.experimentalAddMove(e,t)}experimentalAddAlgLeaf(e,t){this.experimentalModel.experimentalAddAlgLeaf(e,t)}static get observedAttributes(){let e=[];for(let t of Object.keys(tf))e.push(t,ty+t);return e}experimentalRemoveFinalChild(){this.experimentalModel.experimentalRemoveFinalChild()}attributeChangedCallback(e,t,i){e.startsWith(ty)&&(e=e.slice(ty.length));let r=tf[e];r&&(this[r]=i)}async experimentalScreenshot(e){return(await Y(this.experimentalModel,e)).dataURL}async experimentalDownloadScreenshot(e){if(["2D","experimental-2D-LL","experimental-2D-LL-face"].includes(await this.experimentalModel.visualizationStrategy.get())){let t=this.#k,i=await t.currentTwisty2DPuzzleWrapper().twisty2DPuzzle(),r=new XMLSerializer().serializeToString(i.svgWrapper.svgElement);$(URL.createObjectURL(new Blob([r])),e??await G(this.experimentalModel),"svg")}else await (await Y(this.experimentalModel)).download(e)}};n.qh.define("twisty-player",tz);var tL=new n.n5;async function tb(e){return new Promise((t,i)=>{let r=document.getElementById(e);r&&t(r);let n=new MutationObserver(i=>{for(let r of i)"id"===r.attributeName&&r.target instanceof Element&&r.target.getAttribute("id")===e&&(t(r.target),n.disconnect())});n.observe(document.body,{attributeFilter:["id"],subtree:!0})})}tL.replaceSync(`
:host {
  display: inline;
}

.wrapper {
  display: inline;
}

a:not(:hover) {
  color: inherit;
  text-decoration: none;
}

twisty-alg-leaf-elem.twisty-alg-comment {
  color: rgba(0, 0, 0, 0.4);
}

.wrapper.current-move {
  background: rgba(66, 133, 244, 0.3);
  margin-left: -0.1em;
  margin-right: -0.1em;
  padding-left: 0.1em;
  padding-right: 0.1em;
  border-radius: 0.1em;
}
`);var tD=class extends n.FD{constructor(e,t,i,r,n,a){if(super({mode:"open"}),this.algOrAlgNode=r,this.classList.add(e),this.addCSS(tL),a){let e=this.contentWrapper.appendChild(document.createElement("a"));e.href="#",e.textContent=t,e.addEventListener("click",e=>{e.preventDefault(),i.twistyAlgViewer.jumpToIndex(i.earliestMoveIndex,n)})}else this.contentWrapper.appendChild(document.createElement("span")).textContent=t}pathToIndex(e){return[]}setCurrentMove(e){this.contentWrapper.classList.toggle("current-move",e)}};n.qh.define("twisty-alg-leaf-elem",tD);var tk=class extends n.XB{constructor(e,t){super(),this.algOrAlgNode=t,this.classList.add(e)}queue=[];addString(e){this.queue.push(document.createTextNode(e))}addElem(e){return this.queue.push(e.element),e.moveCount}flushQueue(e=1){for(let t of tT(this.queue,e))this.append(t);this.queue=[]}pathToIndex(e){return[]}};function tT(e,t){if(1===t)return e;let i=Array.from(e);return i.reverse(),i}n.qh.define("twisty-alg-wrapper-elem",tk);var tI=class extends o.Yp{traverseAlg(e,t){let i=0,r=new tk("twisty-alg-alg",e),n=!0;for(let a of(0,o.ob)(e.childAlgNodes(),t.direction))n||r.addString(" "),n=!1,a.as(o.vR)?.experimentalNISSGrouping&&r.addString("^("),a.as(o.aU)?.experimentalNISSPlaceholder||(i+=r.addElem(this.traverseAlgNode(a,{earliestMoveIndex:t.earliestMoveIndex+i,twistyAlgViewer:t.twistyAlgViewer,direction:t.direction}))),a.as(o.vR)?.experimentalNISSGrouping&&r.addString(")");return r.flushQueue(t.direction),{moveCount:i,element:r}}traverseGrouping(e,t){var i;let r=e.experimentalAsSquare1Tuple(),n=(i=t.direction,e.amount<0?1===i?-1:1:i),a=0,s=new tk("twisty-alg-grouping",e);return s.addString("("),r?(a+=s.addElem({moveCount:1,element:new tD("twisty-alg-move",r[0].amount.toString(),t,r[0],!0,!0)}),s.addString(", "),a+=s.addElem({moveCount:1,element:new tD("twisty-alg-move",r[1].amount.toString(),t,r[1],!0,!0)})):a+=s.addElem(this.traverseAlg(e.alg,{earliestMoveIndex:t.earliestMoveIndex+a,twistyAlgViewer:t.twistyAlgViewer,direction:n})),s.addString(`)${e.experimentalRepetitionSuffix}`),s.flushQueue(),{moveCount:a*Math.abs(e.amount),element:s}}traverseMove(e,t){let i=new tD("twisty-alg-move",e.toString(),t,e,!0,!0);return t.twistyAlgViewer.highlighter.addMove(e[o.s$],i),{moveCount:1,element:i}}traverseCommutator(e,t){let i=0,r=new tk("twisty-alg-commutator",e);r.addString("["),r.flushQueue();let[n,a]=tT([e.A,e.B],t.direction);return i+=r.addElem(this.traverseAlg(n,{earliestMoveIndex:t.earliestMoveIndex+i,twistyAlgViewer:t.twistyAlgViewer,direction:t.direction})),r.addString(", "),i+=r.addElem(this.traverseAlg(a,{earliestMoveIndex:t.earliestMoveIndex+i,twistyAlgViewer:t.twistyAlgViewer,direction:t.direction})),r.flushQueue(t.direction),r.addString("]"),r.flushQueue(),{moveCount:2*i,element:r}}traverseConjugate(e,t){let i=0,r=new tk("twisty-alg-conjugate",e);r.addString("[");let n=r.addElem(this.traverseAlg(e.A,{earliestMoveIndex:t.earliestMoveIndex+i,twistyAlgViewer:t.twistyAlgViewer,direction:t.direction}));return i+=n,r.addString(": "),i+=r.addElem(this.traverseAlg(e.B,{earliestMoveIndex:t.earliestMoveIndex+i,twistyAlgViewer:t.twistyAlgViewer,direction:t.direction})),r.addString("]"),r.flushQueue(),{moveCount:i+n,element:r}}traversePause(e,t){return e.experimentalNISSGrouping?this.traverseAlg(e.experimentalNISSGrouping.alg,t):{moveCount:1,element:new tD("twisty-alg-pause",".",t,e,!0,!0)}}traverseNewline(e,t){let i=new tk("twisty-alg-newline",e);return i.append(document.createElement("br")),{moveCount:0,element:i}}traverseLineComment(e,t){return{moveCount:0,element:new tD("twisty-alg-line-comment",`//${e.text}`,t,e,!1,!1)}}},tS=(0,o.RU)(tI),tA=class{moveCharIndexMap=new Map;currentElem=null;addMove(e,t){this.moveCharIndexMap.set(e,t)}set(e){let t=e?this.moveCharIndexMap.get(e[o.s$])??null:null;this.currentElem!==t&&(this.currentElem?.classList.remove("twisty-alg-current-move"),this.currentElem?.setCurrentMove(!1),t?.classList.add("twisty-alg-current-move"),t?.setCurrentMove(!0),this.currentElem=t)}},tC=class extends n.XB{highlighter=new tA;#S;#A=null;lastClickTimestamp=null;constructor(e){super(),e?.twistyPlayer&&(this.twistyPlayer=e?.twistyPlayer)}connectedCallback(){}setAlg(e){this.#S=tS(e,{earliestMoveIndex:0,twistyAlgViewer:this,direction:1}).element,this.textContent="",this.appendChild(this.#S)}get twistyPlayer(){return this.#A}set twistyPlayer(e){this.#C(e)}async #C(e){if(this.#A)return void console.warn("twisty-player reassignment is not supported");if(null===e)throw Error("clearing twistyPlayer is not supported");this.#A=e,this.#A.experimentalModel.alg.addFreshListener(e=>{this.setAlg(e.alg)});let t=(await this.#A.experimentalModel.alg.get()).alg,i=o.s$ in t?t:o.BE.fromString(t.toString());this.setAlg(i),e.experimentalModel.currentMoveInfo.addFreshListener(e=>{let t=e.currentMoves[0];if(t??=e.movesStarting[0],t??=e.movesFinishing[0]){let e=t.move;this.highlighter.set(e)}else this.highlighter.set(null)}),e.experimentalModel.detailedTimelineInfo.addFreshListener(e=>{e.timestamp!==this.lastClickTimestamp&&(this.lastClickTimestamp=null)})}async jumpToIndex(e,t){let i=this.#A;if(i){i.pause();let r=(async()=>{let r=await i.experimentalModel.indexer.get();return r.indexToMoveStartTimestamp(e)+r.moveDuration(e)-250*!!t})();i.experimentalModel.timestampRequest.set(await r),this.lastClickTimestamp===await r?(i.play(),this.lastClickTimestamp=null):this.lastClickTimestamp=await r}}async attributeChangedCallback(e,t,i){if("for"===e){let e=document.getElementById(i);if(e||console.info("for= elem does not exist, waiting for one"),await customElements.whenDefined("twisty-player"),!((e=await tb(i))instanceof tz))return void console.warn("for= elem is not a twisty-player");this.twistyPlayer=e}}static get observedAttributes(){return["for"]}};n.qh.define("twisty-alg-viewer",tC);var tE=class extends o.Yp{traverseAlg(e,t){let i=[],r=0;for(let n of e.childAlgNodes()){let e=this.traverseAlgNode(n,{numMovesSofar:t.numMovesSofar+r});i.push(e.tokens),r+=e.numLeavesInside}return{tokens:Array.prototype.concat(...i),numLeavesInside:r}}traverseGrouping(e,t){let i=this.traverseAlg(e.alg,t);return{tokens:i.tokens,numLeavesInside:i.numLeavesInside*e.amount}}traverseMove(e,t){return{tokens:[{leaf:e,idx:t.numMovesSofar}],numLeavesInside:1}}traverseCommutator(e,t){let i=this.traverseAlg(e.A,t),r=this.traverseAlg(e.B,{numMovesSofar:t.numMovesSofar+i.numLeavesInside});return{tokens:i.tokens.concat(r.tokens),numLeavesInside:2*i.numLeavesInside+r.numLeavesInside}}traverseConjugate(e,t){let i=this.traverseAlg(e.A,t),r=this.traverseAlg(e.B,{numMovesSofar:t.numMovesSofar+i.numLeavesInside});return{tokens:i.tokens.concat(r.tokens),numLeavesInside:2*i.numLeavesInside+2*r.numLeavesInside}}traversePause(e,t){return{tokens:[{leaf:e,idx:t.numMovesSofar}],numLeavesInside:1}}traverseNewline(e,t){return{tokens:[],numLeavesInside:0}}traverseLineComment(e,t){return{tokens:[],numLeavesInside:0}}},tN=(0,o.RU)(tE),tP=class extends n.nB{getDefaultValue(){return""}},tj=class extends n.j8{derive(e){return K(e.value)}},tR=class extends n.FB{getDefaultValue(){return{selectionStart:0,selectionEnd:0,endChangedMostRecently:!1}}async derive(e,t){let{selectionStart:i,selectionEnd:r}=e,n=await t;return{selectionStart:i,selectionEnd:r,endChangedMostRecently:e.selectionStart===n.selectionStart&&e.selectionEnd!==(await t).selectionEnd}}},tO=class extends n.j8{derive(e){return e.selectionInfo.endChangedMostRecently?e.selectionInfo.selectionEnd:e.selectionInfo.selectionStart}},tU=class extends n.j8{derive(e){return tN(e.algWithIssues.alg,{numMovesSofar:0}).tokens}},tB=class extends n.j8{derive(e){function t(t){let i;return null===t?null:(i=e.targetChar<t.leaf[o.s$]?"before":e.targetChar===t.leaf[o.s$]?"start":e.targetChar<t.leaf[o.n9]?"inside":e.targetChar===t.leaf[o.n9]?"end":"after",{leafInfo:t,where:i})}let i=null;for(let r of e.leafTokens){if(e.targetChar<r.leaf[o.s$]&&null!==i)break;if(e.targetChar<=r.leaf[o.n9])return t(r);i=r}return t(i)}},tV=class{valueProp=new tP;selectionProp=new tR;targetCharProp=new tO({selectionInfo:this.selectionProp});algEditorAlgWithIssues=new tj({value:this.valueProp});leafTokensProp=new tU({algWithIssues:this.algEditorAlgWithIssues});leafToHighlight=new tB({leafTokens:this.leafTokensProp,targetChar:this.targetCharProp})};function tF(e,t){let i=e.indexOf(t);return -1===i?[e,""]:[e.slice(0,i),e.slice(i)]}function tq(e){let t=[];for(let i of e.split("\n")){let[e,r]=tF(i,"//");e=e.replaceAll("’","'"),t.push(e+r)}return t.join("\n")}var tW=new n.n5;tW.replaceSync(`
:host {
  width: 384px;
  display: grid;
}

.wrapper {
  /*overflow: hidden;
  resize: horizontal;*/

  background: var(--background, none);
  display: grid;
}

textarea, .carbon-copy {
  grid-area: 1 / 1 / 2 / 2;

  width: 100%;
  font-family: sans-serif;
  line-height: 1.2em;

  font-size: var(--font-size, inherit);
  font-family: var(--font-family, sans-serif);

  box-sizing: border-box;

  padding: var(--padding, 0.5em);
  /* Prevent horizontal growth. */
  overflow-x: hidden;
}

textarea {
  resize: none;
  background: none;
  z-index: 2;
  border: 1px solid var(--border-color, rgba(0, 0, 0, 0.25));
  overflow: hidden;
}

.carbon-copy {
  white-space: pre-wrap;
  word-wrap: break-word;
  color: transparent;
  user-select: none;
  pointer-events: none;

  z-index: 1;
}

.carbon-copy .highlight {
  background: var(--highlight-color, rgba(255, 128, 0, 0.5));
  padding: 0.1em 0.2em;
  margin: -0.1em -0.2em;
  border-radius: 0.2em;
}

.wrapper.issue-warning textarea,
.wrapper.valid-for-puzzle-warning textarea {
  outline: none;
  border: 1px solid rgba(200, 200, 0, 0.5);
  background: rgba(255, 255, 0, 0.1);
}

.wrapper.issue-error textarea,
.wrapper.valid-for-puzzle-error textarea {
  outline: none;
  border: 1px solid red;
  background: rgba(255, 0, 0, 0.1);
}
`);var tQ="for-twisty-player",tH="placeholder",tY="twisty-player-prop",tG=class extends n.FD{model=new tV;#E=document.createElement("textarea");#N=document.createElement("div");#P=document.createElement("span");#j=document.createElement("span");#R=document.createElement("span");#O=new T(this,"valid-for-puzzle-",["none","warning","error"]);#A=null;#U;get #B(){return null===this.#A?null:this.#A.experimentalModel[this.#U]}debugNeverRequestTimestamp=!1;constructor(e){super(),this.#N.classList.add("carbon-copy"),this.addElement(this.#N),this.#E.rows=1,this.addElement(this.#E),this.#P.classList.add("prefix"),this.#N.appendChild(this.#P),this.#j.classList.add("highlight"),this.#N.appendChild(this.#j),this.#R.classList.add("suffix"),this.#N.appendChild(this.#R),this.#E.placeholder="Alg",this.#E.setAttribute("spellcheck","false"),this.addCSS(tW),this.#E.addEventListener("input",()=>{this.#V=!0,this.onInput()}),this.#E.addEventListener("blur",()=>this.onBlur()),document.addEventListener("selectionchange",()=>this.onSelectionChange()),e?.twistyPlayer&&(this.twistyPlayer=e.twistyPlayer),this.#U=e?.twistyPlayerProp??"alg",e?.twistyPlayerProp==="alg"&&this.model.leafToHighlight.addFreshListener(e=>{e&&this.highlightLeaf(e.leafInfo.leaf)})}connectedCallback(){this.#E.addEventListener("paste",e=>{let t=e.clipboardData?.getData("text");t&&(!function(e,t){let{value:i}=e,{selectionStart:r,selectionEnd:n}=e,a=i.slice(0,r),s=i.slice(n);t=t.replaceAll("\r\n","\n");let l=a.match(/\/\/[^\n]*$/),d="/"===i[r-1]&&"/"===t[0],u=l||d,c=t.match(/\/\/[^\n]*$/),h=t;if(u){let[e,i]=tF(t,"\n");h=e+tq(i)}else h=tq(t);let p=!u&&0!==r&&!["\n"," "].includes(h[0])&&!["\n"," "].includes(i[r-1]),m=!c&&n!==i.length&&!["\n"," "].includes(h.at(-1))&&!["\n"," "].includes(i[n]);function g(e,t){let i=e+h+t,r=!!function(e){try{return o.BE.fromString(e)}catch{return null}}(a+i+s);return r&&(h=i),r}p&&m&&g(" "," ")||p&&g(" ","")||m&&g(""," "),N?.execCommand("insertText",!1,h)||e.setRangeText(h,r,n,"end")}(this.#E,t),e.preventDefault(),this.onInput())})}set algString(e){this.#E.value=e,this.onInput()}get algString(){return this.#E.value}set placeholder(e){this.#E.placeholder=e}#V=!1;onInput(){this.#j.hidden=!0,this.highlightLeaf(null);let e=this.#E.value.trimEnd();this.model.valueProp.set(e),this.#B?.set(e)}async onSelectionChange(){if(document.activeElement!==this||this.shadow.activeElement!==this.#E||"alg"!==this.#U)return;let{selectionStart:e,selectionEnd:t}=this.#E;this.model.selectionProp.set({selectionStart:e,selectionEnd:t})}async onBlur(){}setAlgIssueClassForPuzzle(e){this.#O.setValue(e)}#F(e){return e.endsWith("\n")?`${e} `:e}#q=null;highlightLeaf(e){if(null===e){this.#P.textContent="",this.#j.textContent="",this.#R.textContent=this.#F(this.#E.value);return}e!==this.#q&&(this.#q=e,this.#P.textContent=this.#E.value.slice(0,e[o.s$]),this.#j.textContent=this.#E.value.slice(e[o.s$],e[o.n9]),this.#R.textContent=this.#F(this.#E.value.slice(e[o.n9])),this.#j.hidden=!1)}get twistyPlayer(){return this.#A}set twistyPlayer(e){if(this.#A)return void console.warn("twisty-player reassignment/clearing is not supported");this.#A=e,e&&((async()=>{this.algString=this.#B?(await this.#B.get()).alg.toString():""})(),"alg"===this.#U&&(this.#A?.experimentalModel.puzzleAlg.addFreshListener(e=>{if(0===e.issues.errors.length){this.setAlgIssueClassForPuzzle(0===e.issues.warnings.length?"none":"warning");let t=e.alg,i=o.BE.fromString(this.algString);t.isIdentical(i)||(this.algString=t.toString(),this.onInput())}else this.setAlgIssueClassForPuzzle("error")}),this.model.leafToHighlight.addFreshListener(async t=>{let i;if(null===t)return;let[r,n]=await Promise.all([await e.experimentalModel.indexer.get(),await e.experimentalModel.timestampRequest.get()]);if("auto"===n&&!this.#V)return;let a=r.indexToMoveStartTimestamp(t.leafInfo.idx),s=r.moveDuration(t.leafInfo.idx);switch(t.where){case"before":i=a;break;case"start":case"inside":i=a+s/4;break;case"end":case"after":i=a+s;break;default:throw console.log("invalid where"),Error("Invalid where!")}this.debugNeverRequestTimestamp||e.experimentalModel.timestampRequest.set(i)}),e.experimentalModel.currentLeavesSimplified.addFreshListener(async t=>{let i=(await e.experimentalModel.indexer.get()).getAnimLeaf(t.patternIndex);this.highlightLeaf(i)})))}attributeChangedCallback(e,t,i){switch(e){case tQ:{let e=document.getElementById(i);if(!e)return void console.warn(`${tQ}= elem does not exist`);if(!(e instanceof tz))return void console.warn(`${tQ}=is not a twisty-player`);this.twistyPlayer=e;return}case tH:this.placeholder=i;return;case tY:if(this.#A)throw console.log("cannot set prop"),Error("cannot set prop after twisty player");this.#U=i;return}}static get observedAttributes(){return[tQ,tH,tY]}};n.qh.define("twisty-alg-editor",tG);var t$=new n.n5;t$.replaceSync(`
.wrapper {
  background: rgb(255, 245, 235);
  border: 1px solid rgba(0, 0, 0, 0.25);

  /* Workaround from https://stackoverflow.com/questions/40010597/how-do-i-apply-opacity-to-a-css-color-variable */
  --text-color: 0, 0, 0;
  --heading-background: 255, 230, 210;

  color: rgb(var(--text-color));
}

.setup-alg, twisty-alg-viewer {
  padding: 0.5em 1em;
}

.heading {
  background: rgba(var(--heading-background), 1);
  color: rgba(var(--text-color), 1);
  font-weight: bold;
  padding: 0.25em 0.5em;
  display: grid;
  grid-template-columns: auto 1fr;

  /* For the move count hover elems. */
  position: sticky;
}

.heading.title {
  background: rgb(255, 245, 235);
  font-size: 150%;
  white-space: pre-wrap;
}

.heading .move-count {
  font-weight: initial;
  text-align: right;
  color: rgba(var(--text-color), 0.4);
}

.wrapper.dark-mode .heading .move-count {
  color: rgba(var(--text-color), 0.7);
}

.heading a {
  text-decoration: none;
  color: inherit;
}

twisty-player {
  width: 100%;
  min-height: 128px;
  height: 288px;
  resize: vertical;
  overflow-y: hidden;
}

twisty-player + .heading {
  padding-top: 0.5em;
}

twisty-alg-viewer {
  display: inline-block;
}

.wrapper {
  container-type: inline-size;
}

.scrollable-region {
  border-top: 1px solid rgba(0, 0, 0, 0.25);
}

.scrollable-region {
  max-height: 18em;
  overflow-y: auto;
}

@container (min-width: 512px) {
  .responsive-wrapper {
    display: grid;
    grid-template-columns: 1fr 1fr;
  }
  twisty-player {
    height: 320px
  }
  .scrollable-region {
    border-top: none;
    border-left: 1px solid rgba(0, 0, 0, 0.25);
    contain: strict;
    max-height: 100cqh;
  }
}

.wrapper:fullscreen,
.wrapper:fullscreen .responsive-wrapper {
  width: 100%;
  height: 100%;
}

.wrapper:fullscreen twisty-player,
.wrapper:fullscreen .scrollable-region {
  height: 50%;
}

@container (min-width: 512px) {
  .wrapper:fullscreen twisty-player,
  .wrapper:fullscreen .scrollable-region {
    height: 100%;
  }
}

/* TODO: dedup with Twizzle Editor */
.move-count > span:hover:before {
  background-color: rgba(var(--heading-background), 1);
  color: rgba(var(--text-color), 1);
  backdrop-filter: blur(4px);
  z-index: 100;
  position: absolute;
  padding: 0.5em;
  top: 1.5em;
  right: 0;
  content: attr(data-before);
  white-space: pre-wrap;
  text-align: left;
}

.move-count > span:hover {
  color: rgba(var(--text-color), 1);
  cursor: help;
}
`);var tZ=new n.n5;tZ.replaceSync(`
.wrapper {
  background: white;
  --heading-background: 232, 239, 253
}

.wrapper.dark-mode {
  --text-color: 236, 236, 236;
  --heading-background: 29, 29, 29;
}

.scrollable-region {
  overflow-y: auto;
}

.wrapper.dark-mode {
  background: #262626;
  --text-color: 142, 142, 142;
  border-color: #FFFFFF44;
  color-scheme: dark;
}

.wrapper.dark-mode .heading:not(.title) {
  background: #1d1d1d;
}

.heading.title {
  background: none;
}
`);var tX="outer block moves (e.g. R, Rw, or 4r)",tJ="inner block moves (e.g. M or 2-5r)",t_={OBTM:`HTM = OBTM ("Outer Block Turn Metric"):
\u2022 ${tJ} count as 2 turns
\u2022 ${tX} count as 1 turn
\u2022 rotations (e.g. x) count as 0 turns`,OBQTM:`QTM = OBQTM ("Outer Block Quantum Turn Metric"):
\u2022 ${tJ} count as 2 turns per quantum (e.g. M2 counts as 4)
\u2022 ${tX} count as 1 turn per quantum (e.g. R2 counts as 2)
\u2022 rotations (e.g. x) count as 0 turns`,RBTM:`STM = RBTM ("Range Block Turn Metric"):
\u2022 ${tJ} count as 1 turn
\u2022 ${tX} count as 1 turn
\u2022 rotations (e.g. x) count as 0 turns`,RBQTM:`SQTM = RBQTM ("Range Block Quantum Turn Metric"):
\u2022 ${tJ} count as 1 turn per quantum (e.g. M2 counts as 2)
\u2022 ${tX} count as 1 turn per quantum (e.g. R2 counts as 2)
\u2022 rotations (e.g. x) count as 0 turns`,ETM:`ETM ("Execution Turn Metric"):
\u2022 all moves (including rotations) count as 1 turn`},tK={OBTM:"OB",OBQTM:"OBQ",RBTM:"RB",RBQTM:"RBQ",ETM:"E"},t0=class extends n.FD{constructor(e){super({mode:"open"}),this.options=e}twistyPlayer=null;a=null;#W(){if(this.contentWrapper.textContent="",this.a){let e=this.contentWrapper.appendChild(document.createElement("span"));e.textContent="❗️",e.title="Could not show a player for link",this.addElement(this.a)}this.removeCSS(t$);let e=this.shadow.adoptedStyleSheets.indexOf(t$);void 0!==e&&this.shadow.adoptedStyleSheets.splice(e,e+1),this.#Q?.remove()}#Q;#H;#Y;#G;async connectedCallback(){if(this.#Y=this.addElement(document.createElement("div")),this.#Y.classList.add("responsive-wrapper"),this.options?.colorScheme==="dark"&&this.contentWrapper.classList.add("dark-mode"),this.addCSS(t$),this.options?.cdnForumTweaks&&this.addCSS(tZ),this.a=this.querySelector("a"),!this.a)return;let e=function(e="",t=location.href){let i=new URL(t).searchParams,r={};for(let[t,n]of Object.entries({alg:"alg","setup-alg":"experimental-setup-alg","setup-anchor":"experimental-setup-anchor",puzzle:"puzzle",stickering:"experimental-stickering","puzzle-description":"experimental-puzzle-description",title:"experimental-title","video-url":"experimental-video-url",competition:"experimental-competition-id"})){let a=i.get(e+t);null!==a&&(r[tf[n]]=a)}return r}("",this.a.href),{hostname:t,pathname:r}=new URL(this.a?.href);if("alpha.twizzle.net"!==t)return void this.#W();if(["/edit/","/explore/"].includes(r)){let t="/explore/"===r;if(e.puzzle&&!(e.puzzle in s.SB)){let t=(await i.e(815).then(i.bind(i,5815))).getPuzzleDescriptionString(e.puzzle);delete e.puzzle,e.experimentalPuzzleDescription=t}if(this.twistyPlayer=this.#Y.appendChild(new tz({background:this.options?.cdnForumTweaks?"checkered-transparent":"checkered",colorScheme:this.options?.colorScheme==="dark"?"dark":"light",...e,viewerLink:t?"experimental-twizzle-explorer":"auto"})),this.twistyPlayer.fullscreenElement=this.contentWrapper,e.experimentalTitle&&(this.twistyPlayer.experimentalTitle=e.experimentalTitle),this.#H=this.#Y.appendChild(document.createElement("div")),this.#H.classList.add("scrollable-region"),e.experimentalTitle&&this.#$(e.experimentalTitle).classList.add("title"),e.experimentalSetupAlg){this.#$("Setup",async()=>(await this.twistyPlayer?.experimentalModel.setupAlg.get())?.alg.toString()??null);let t=this.#H.appendChild(document.createElement("div"));t.classList.add("setup-alg"),t.textContent=new o.BE(e.experimentalSetupAlg).toString()}let n=this.#$("Moves",async()=>(await this.twistyPlayer?.experimentalModel.alg.get())?.alg.toString()??null);this.#G=n.appendChild(function(e,t=document.createElement("span")){async function i(){let[i,r]=await Promise.all([e.puzzleAlg.get(),e.puzzleLoader.get()]);if(0!==i.issues.errors.length){t.textContent="";return}let n=!0;function s(e){n?n=!1:t.append(")(");let s=t.appendChild(document.createElement("span")),l=(0,a.Is)(r,e,i.alg);s.append(`${tK[e]}: `);let o=s.appendChild(document.createElement("span"));o.textContent=l.toString(),o.classList.add("move-number"),s.setAttribute("data-before",t_[e]??""),s.setAttribute("title",t_[e]??"")}t.textContent="(","3x3x3"===r.id?(s("OBTM"),s("OBQTM"),s("RBTM")):r.pg&&(s("RBTM"),s("RBQTM")),s("ETM"),t.append(")")}return e.puzzleAlg.addFreshListener(i),e.puzzleID.addFreshListener(i),t}(this.twistyPlayer.experimentalModel)),this.#G.classList.add("move-count"),this.#H.appendChild(new tC({twistyPlayer:this.twistyPlayer})).part.add("twisty-alg-viewer")}else this.#W()}#$(e,t){let i=this.#H.appendChild(document.createElement("div"));i.classList.add("heading");let r=i.appendChild(document.createElement("span"));if(r.textContent=e,t){r.textContent+=" ";let e=r.appendChild(document.createElement("a"));async function n(t){e.textContent=t,await new Promise(e=>setTimeout(e,2e3)),e.textContent===t&&(e.textContent="\uD83D\uDCCB")}e.textContent="\uD83D\uDCCB",e.href="#",e.title="Copy to clipboard",e.addEventListener("click",async i=>{i.preventDefault(),e.textContent="\uD83D\uDCCB…";let r=await t();if(r)try{await navigator.clipboard.writeText(r),n("\uD83D\uDCCB✅")}catch(e){throw n("\uD83D\uDCCB❌"),e}else n("\uD83D\uDCCB❌")})}return i}};n.qh.define("twizzle-link",t0)}}]);