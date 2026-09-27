"""Build repeatable test inputs. Assignment computations remain in C/CUDA."""
from pathlib import Path
import random
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def crypt(raw):
    p = list(map(ord, raw))
    values = [p[0]+2,p[0]-2,p[0]+1,p[1]+3,p[1]-3,p[1]-1,p[2]+2,p[2]-2,p[3]+4,p[3]-4]
    for i, v in enumerate(values):
        lo, hi = (97,122) if i < 6 else (48,57)
        if v > hi: values[i] = v-hi+lo
        elif v < lo: values[i] = lo-v+lo
    return ''.join(map(chr, values))

if __name__ == '__main__':
    rng = random.Random(6005)
    passwords = ['aa00','aa99','zz00','zz99','az09','za90']
    passwords += [f'{chr(97+rng.randrange(26))}{chr(97+rng.randrange(26))}{rng.randrange(100):02}' for _ in range(9994)]
    (ROOT/'data/passwords.txt').write_text('\n'.join(map(crypt,passwords))+'\n')
    (ROOT/'data/passwords_expected.txt').write_text('\n'.join(passwords)+'\n')
    y,x = np.mgrid[:512,:512]
    # A numerical test target: flat areas, a diagonal, a circle and a ramp.
    target = np.zeros((512,512),dtype=np.uint8)
    target[40:180,40:180] = 220
    target[((x-350)**2+(y-130)**2)<75**2] = 180
    target[(y>270)&(x>y-220)&(x<y-140)] = 255
    target[300:450,300:480] = np.linspace(0,255,180,dtype=np.uint8)[None,:]
    Image.fromarray(target).save(ROOT/'data/images/sobel_target.png')
    rgba = np.stack(((x%256).astype('uint8'),(y%256).astype('uint8'),((x+y)%256).astype('uint8'),np.full_like(x,255,dtype='uint8')),axis=2)
    Image.fromarray(rgba).save(ROOT/'data/images/colour_ramp.png')
    print('Created 10,000 password fixtures and two 512x512 image targets.')
