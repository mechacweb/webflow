import timeline as T, subprocess
(s1,e1),(s2,e2)=T.SEGMENTS
b0,b1=T.BLEEP
fc=(f"[0:a]atrim={s1}:{e1},asetpts=PTS-STARTPTS,afade=t=out:st={e1-s1-0.03}:d=0.03[a0];"
    f"[0:a]atrim={s2}:{e2},asetpts=PTS-STARTPTS,afade=t=in:d=0.03[a1];"
    "[a0][a1]concat=n=2:v=0:a=1,"
    "highpass=f=85,afftdn=nf=-28,equalizer=f=280:t=q:w=1.2:g=-3,equalizer=f=3200:t=q:w=1.5:g=3.5,"
    "equalizer=f=9000:t=h:w=1:g=2,deesser=i=0.4,"
    "acompressor=threshold=-22dB:ratio=4:attack=4:release=90:makeup=7dB,"
    f"volume=enable='between(t,{b0},{b1})':volume=0,"
    f"apad=whole_dur={T.TOTAL+1},aformat=channel_layouts=stereo,asplit=2[voice][key];"
    "[1:a]volume=0.30[mus];"
    "[mus][key]sidechaincompress=threshold=0.015:ratio=12:attack=12:release=380:makeup=1[duck];"
    "[2:a]volume=0.6[sfx];"
    "[voice][duck][sfx]amix=inputs=3:normalize=0:duration=first,"
    "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.89[out]")
subprocess.run(["ffmpeg","-v","error","-y","-i","in.mov","-i","music.wav","-i","sfx.wav","-filter_complex",fc,
                "-map","[out]","-t",f"{T.TOTAL:.3f}","-ar","48000","mix.wav"],check=True)
