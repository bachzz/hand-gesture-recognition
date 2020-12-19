$(document).ready(function(){
  let namespace = "/test";
  let video = document.querySelector("#videoElement");
  let canvas = document.querySelector("#canvasElement");
  let ctx = canvas.getContext('2d');
  photo = document.getElementById('photo');
  var localMediaStream = null;

  let dataURL = canvas.toDataURL('image/jpeg');

  var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);

  function sendSnapshot() {
    // console.log('sent!1');
    if (!localMediaStream) {
      return;
    }
    // console.log('sent2!');
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 300, 150);
    // console.log('before sent!');
    dataURL = canvas.toDataURL('image/jpeg');
    socket.emit('input image', dataURL);
    console.log('sent!');
    // socket.emit('output image');


  }

  socket.on('connect', function() {
    console.log('Connected!');
  });

  // socket.on('tmp_event', function(data) {
  //   console.log(data);
  // });

  socket.on('out-image-event',function(data){
    console.log('received!');
    var img = new Image();
    img.src = dataURL;//data.image_data
    photo.setAttribute('src', data.image_data);

  });

  var constraints = {
    video: {
      width: { min: 640 },
      height: { min: 480 }
    }
  };

  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    video.srcObject = stream;
    localMediaStream = stream;

    setInterval(function () {
      sendSnapshot();
    }, 500);
  }).catch(function(error) {
    console.log(error);
  });
});

