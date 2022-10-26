import argparse
import os
from subprocess import run
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', default = 128, type = int)
parser.add_argument('--dataRoot', default = 'kth')

opt = parser.parse_args()

if not opt.dataRoot:
    raise Exception('There is no directory : %s' % opt.dataRoot)

groups = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
frame_rate = 25

for group in groups:
    print('----------')
    print(group)

    directory = '%s/raw/%s' % (opt.dataRoot, group)
    for vid in os.listdir(directory):
        print(vid)
        fname = vid[0:-11]
        try:
            run('mkdir -p %s/processed/%s/%s' % (opt.dataRoot, group, fname), shell = True)
            run('ffmpeg -i %s/raw/%s/%s -hide_banner -r %d -f image2 -s %dx%d  %s/processed/%s/%s/image-%%03d_%dx%d.png' %
                 (opt.dataRoot, group, vid, frame_rate, opt.imageSize, opt.imageSize, opt.dataRoot, group, fname, opt.imageSize, opt.imageSize),
                 shell = True)
            # to run the above code, you should install ffmpeg by 'sudo apt install ffmpeg'
        except OSError as e:
            print("Reading process failed at %s" % fname, e, file = sys.stderr)
