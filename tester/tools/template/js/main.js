let i = 0;
let j = 0;
let sz = 100;
const image = document.getElementById('image');
const seek = document.getElementById('seek');

let load_img;
update_image();

// ページャ
let page = document.getElementById('page');
for(let i=0;i<50;i++) {
    page.innerHTML += `<button type=Button onclick="Jump(${i});">${i}</button>\n`;
}

function preload_images() {
    if(document.getElementById("preload_images")) {
        return
    }
    const tmp = document.createElement("div");
    tmp.id = "preload_images";

    for(let j=0;j<sz;j++) {
        num = i.toString(10).padStart(4, '0');
        path_dir = `../svg/${num}`;
        const path = `${path_dir}/${j}.svg`;
        const ch = document.createElement("img");
        ch.id = `${j}`;
        ch.src = path;
        tmp.appendChild(ch);
    }
    load_img = tmp;
}

function unload_images() {
    let tmp = document.getElementById("preload_images");
    if(tmp) {
        document.removeChild(tmp);
    }
}

function update_image(with_preload = true) {
    if(with_preload) {
        preload_images();
    }
    num = i.toString(10).padStart(4, '0');
    path_dir = `../svg/${num}`;
    image.innerHTML = `<h2>${num}.svg</h2>`;
    image.innerHTML += `<img src="${path_dir}/${j}.svg" width="1000" height="1000">`;
}

function Prev() {
    if(i == 0) return;
    unload_images();
    i--;
    update_image();
}

function Next() {
    if(i == 49) return;
    unload_images();
    i++;
    update_image();
}

function Jump(x) {
    unload_images();
    i = x;
    update_image();
}

function rangeOnChange(e) {
    j = e.target.value;
    update_image(false);
}

window.onload = () => {
    seek.addEventListener('input', rangeOnChange);
}
