// Lấy tham chiếu đến video
var video = document.querySelector("#videoElement");

// Hàm nhận diện thân thể
async function detectBody() {
    await tf.setBackend("webgl");

    const net = await bodyPix.load();

    setInterval(async () => {
        const segmentation = await net.segmentPerson(video);
        const mask = bodyPix.toMask(segmentation);
        bodyPix.drawMask(video, mask, 1, 0, 0, false);
    }, 100);
}

// Bật camera và bắt đầu nhận diện thân thể
navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
        video.srcObject = stream;
        detectBody();
    })
    .catch(function(error) {
        console.log("Lỗi: " + error);
    });