function Prev() {
    if(i == 0) return;
    i--;
    let num = i.toString(10).padStart(4, '0');
    element.innerHTML = '';
    element.innerHTML += `<h2>${num}.svg</h2>`;
    element.innerHTML += `<img src="../svg/${num}.svg" width="1000" height="1000">`;
}

function Next() {
    if(i == 49) return;
    i++;
    let num = i.toString(10).padStart(4, '0');
    element.innerHTML = '';
    element.innerHTML += `<h2>${num}.svg</h2>`;
    element.innerHTML += `<img src="../svg/${num}.svg" width="1000" height="1000">`;
}

function Jump(x) {
    i = x;
    let num = i.toString(10).padStart(4, '0');
    element.innerHTML = '';
    element.innerHTML += `<h2>${num}.svg</h2>`;
    element.innerHTML += `<img src="../svg/${num}.svg" width="1000" height="1000">`;
}

let i = 0;
let element = document.getElementById('home');
let num = i.toString(10).padStart(4, '0');
element.innerHTML += `<h2>${num}.svg</h2>`;
element.innerHTML += `<img src="../svg/${num}.svg" width="1000" height="1000">`;

let page = document.getElementById('page');
for(let i=0;i<50;i++) {
    page.innerHTML += `<button type=Button onclick="Jump(${i});">${i}</button>\n`;
}
