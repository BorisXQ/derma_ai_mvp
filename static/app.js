const fileInput=document.getElementById("fileInput");
const chooseBtn=document.getElementById("chooseBtn");
const dropzone=document.getElementById("dropzone");
const previewCard=document.getElementById("previewCard");
const preview=document.getElementById("preview");
const fileName=document.getElementById("fileName");
const analyzeBtn=document.getElementById("analyzeBtn");
const loading=document.getElementById("loading");
const result=document.getElementById("result");
let selectedFile=null;

chooseBtn.onclick=()=>fileInput.click();
fileInput.onchange=()=>fileInput.files[0]&&selectFile(fileInput.files[0]);

["dragenter","dragover"].forEach(e=>dropzone.addEventListener(e,x=>{x.preventDefault();dropzone.classList.add("drag")}));
["dragleave","drop"].forEach(e=>dropzone.addEventListener(e,x=>{x.preventDefault();dropzone.classList.remove("drag")}));
dropzone.addEventListener("drop",e=>{const f=e.dataTransfer.files[0];if(f)selectFile(f)});

function selectFile(file){
  if(!["image/jpeg","image/png","image/webp"].includes(file.type)){alert("Выберите JPG, PNG или WEBP.");return}
  if(file.size>8*1024*1024){alert("Файл должен быть меньше 8 MB.");return}
  selectedFile=file;
  fileName.textContent=file.name;
  preview.src=URL.createObjectURL(file);
  dropzone.classList.add("hidden");
  previewCard.classList.remove("hidden");
  result.classList.add("hidden");
}

analyzeBtn.onclick=async()=>{
  if(!selectedFile)return;
  const fd=new FormData();fd.append("file",selectedFile);
  previewCard.classList.add("hidden");
  loading.classList.remove("hidden");
  result.classList.add("hidden");
  try{
    const r=await fetch("/api/v1/analyze",{method:"POST",body:fd});
    const data=await r.json();
    if(!r.ok)throw new Error(data.detail||"Ошибка анализа");
    document.getElementById("disease").textContent=data.prediction.disease;
    document.getElementById("interpretation").textContent=data.interpretation;
    document.getElementById("confidence").textContent=data.prediction.confidence_percent+"%";
    document.getElementById("barFill").style.width=data.prediction.confidence_percent+"%";
    document.getElementById("warning").textContent=data.warning;
    const list=document.getElementById("topList");
    list.innerHTML=data.top_predictions.map(x=>`
      <div class="top-item">
        <div><b>${escapeHtml(x.disease)}</b><div style="color:#6f887e;font-size:12px">${escapeHtml(x.disease_en)}</div></div>
        <strong>${x.confidence_percent}%</strong>
        <div class="track"><span style="width:${x.confidence_percent}%"></span></div>
      </div>`).join("");
    loading.classList.add("hidden");result.classList.remove("hidden");
  }catch(e){
    loading.classList.add("hidden");previewCard.classList.remove("hidden");alert(e.message);
  }
};

function escapeHtml(s){return String(s).replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[m]))}

async function health(){
  try{
    const r=await fetch("/api/v1/health"); if(!r.ok)throw 0;
    document.getElementById("healthDot").style.background="#65e6a7";
    document.getElementById("healthText").textContent="AI API online";
  }catch{
    document.getElementById("healthText").textContent="API unavailable";
  }
}
health();
