import ffmpeg

    in_file = ffmpeg.input('input.mp4')
    overlay_file = ffmpeg.input('overlay.png')
    (
        ffmpeg
        .concat(
            in_file.trim(start_frame=10, end_frame=20),
            in_file.trim(start_frame=30, end_frame=40),
        )
        .overlay(overlay_file.hflip())
        .drawbox(50, 50, 120, 120, color='red', thickness=5)
        .output('out.mp4')
        .run()
    )
