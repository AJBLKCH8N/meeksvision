const streams = [
    { name: "Camera 1", url: "http://localhost:8080/live/camera1.m3u8" },
    { name: "Camera 2", url: "http://localhost:8080/live/camera2.m3u8" }
    // Add more streams as needed
];

const container = document.querySelector('.row');

streams.forEach(stream => {
    const card = `
        <div class="col-md-4 video-card">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">${stream.name}</h5>
                    <video controls>
                        <source src="${stream.url}" type="application/x-mpegURL">
                        Your browser does not support HLS video.
                    </video>
                </div>
            </div>
        </div>
    `;
    container.innerHTML += card;
});
