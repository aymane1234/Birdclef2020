import os
import soundfile
import sys

root = './trial_files/'
for path, subdirs, files in os.walk(root):
    for name in files:
        if (name.endswith('.mp3')):
           a = path + "/" + name
           os.system("mkdir ./trial_converted")
           os.system("mkdir ./trial_converted/trial_files")
           os.system("mkdir ./trial_converted" + path[1:])
           os.system("ffmpeg -i " + a + " -ar 22050 " "-ac 1 " + "./trial_converted/"+ a[2:-3]  + "wav")           

for path, subdirs, files in os.walk("./trial_converted/"):
    for name in files:
        if (name.endswith('.wav')):           
           data, samplerate = soundfile.read(path + "/" + name)                
           soundfile.write(path + "/" + name, data, samplerate, subtype='PCM_16', )
           print(path + "/" + name + "converted")
sys.exit()           