from pydub import AudioSegment

for dir in range(5, 7):
    t1, t2 = 0, 0
    ctr = 0
    original = AudioSegment.from_wav(f'{dir}/Vocal.wav')

    while t2 < 180000:
        try:
            ctr += 1
            t1 = t2
            t2 += 30000
            # print(t1, t2)

            newAudio = original[t1:t2]

            #Exports to a wav file in the current path.
            newAudio.export(f'{dir}/{dir}_audio_{ctr}.wav', format="wav") 
        except:
            continue
