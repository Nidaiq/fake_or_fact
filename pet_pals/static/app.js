function getData() {
    if (d3.select('input[id="switch"] + label').style('background-color') == 'rgb(250, 121, 10)') {
        subject = 'usnews'
    }else if(d3.select('input[id="switch"] + label').style('background-color') == 'rgb(47, 156, 33)') {
        subject='worldnews'
    };
    var title = d3.select('#title').property('value');
    var text = d3.select('#text').property('value');  
    var url = window.location.href;
    window.location.href = url+"/verify/"+subject+"/"+title+"/"+text;
}