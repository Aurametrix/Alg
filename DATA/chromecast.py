# sending videos
import pychromecast
if __name__ == "__main__":
    cast = pychromecast.get_chromecasts()[0]
    mc = cast.media_controller
    mc.play_media("http://192.168.0.103:8000/video_test.mp4", content_type = "video/mp4")
    mc.block_until_active()
    mc.play()
    
