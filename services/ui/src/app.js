const streams = [
    { name: "Camera 1", url: "http://localhost:8080/live/stream.m3u8" },
    // Add more streams as needed
];

const container = document.querySelector('.row');

streams.forEach(stream => {
    const card = `
        <div class="col-md-4 video-card">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">${stream.name}</h5>
                    <video id="${stream.name.replace(/\s+/g, '')}" controls></video>
                </div>
            </div>
        </div>
    `;
    container.innerHTML += card;
    
    if (Hls.isSupported()) {
        var video = document.getElementById(stream.name.replace(/\s+/g, ''));
        var hls = new Hls();
        hls.loadSource(stream.url);
        hls.attachMedia(video);
        hls.on(Hls.Events.MANIFEST_PARSED,function() {
            video.play();
        });
    }
    // Fallback for browsers that natively support HLS
    else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = stream.url;
        video.addEventListener('loadedmetadata',function() {
            video.play();
        });
    }
});
