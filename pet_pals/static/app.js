const url = window.location.href;

function getData() {
    if (d3.select('input[id="switch"] + label').style('background-color') == 'rgb(250, 121, 10)') {
        subject = 'US News'
    }else if(d3.select('input[id="switch"] + label').style('background-color') == 'rgb(47, 156, 33)') {
        subject='World News'
    };
    var title = d3.select('#title').property('value');
    var text = d3.select('#text').property('value');  
    window.location.href = url+"/verdict/"+subject+"/"+title+"/"+text;
}

function tryAgain() {
    window.history.back()
}

