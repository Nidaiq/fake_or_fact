function getData() {
    var title = d3.select('#title').property('value');
    var text = d3.select('#text').property('value');
    var url = window.location.href;
    window.location.href = url+"/verify/"+title+"/"+text;
}