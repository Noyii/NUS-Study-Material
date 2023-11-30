from moviepy.editor import VideoFileClip

for dir in range(1, 7):
    t1, t2 = 0, 0
    ctr = 0
    full_video = f'{dir}/Video.mp4'

    print("Starting for ", dir)

    while t2 < 180:
        try:
            ctr += 1
            t1 = t2
            t2 += 30

            clip = VideoFileClip(full_video, target_resolution=(1920, 1080)).subclip(t1, t2)
            clip.write_videofile(
                f'{dir}/{dir}_video_{ctr}.mp4'
            )
            print("-----------------###-----------------")
        except Exception as e:
            print(e)
