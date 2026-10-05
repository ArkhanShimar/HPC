"""Independent reference checks and repeatable measurements for compiled tasks."""
from pathlib import Path
from collections import Counter
import sys, subprocess, tempfile, re, json, time, statistics, shutil
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from make_inputs import crypt

TASK = int(sys.argv[1])
OUT = Path.cwd()/'validation'
OUT.mkdir(exist_ok=True)
checks = []
metrics = {}

def passed(name):
    checks.append(name)
    print('PASS:',name,flush=True)

def run(exe, args, cwd, expected=0):
    result = subprocess.run([str(exe),*map(str,args)],cwd=cwd,text=True,capture_output=True)
    assert result.returncode == expected, f'{exe.name}: exit {result.returncode}\n{result.stdout}\n{result.stderr}'
    return result.stdout

def task1():
    exe = Path.cwd()/'word_count'
    data = ROOT/'data/WordOccurrenceDataset.txt'
    expected = Counter(re.findall(r'[a-z0-9]+',data.read_text().lower()))
    def read_counts(folder):
        lines=(folder/'result.txt').read_text().splitlines()
        words=[line.split('\t')[0] for line in lines]
        assert words == sorted(words) and len(words)==len(set(words))
        return {line.split('\t')[0]:int(line.split('\t')[1]) for line in lines}
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d)
        for n in [1,2,4,8]:
            run(exe,[data,n],folder)
            assert read_counts(folder)==expected
        passed('Full supplied dataset equals independent counts at 1, 2, 4 and 8 threads')
        cases=['', 'One', 'Hello HELLO, world! 123 123', 'a'*10000+' b B\nend', 'one\r\ntwo\tthree   four', 'a b c d e f g h i']
        for text in cases:
            path=folder/'sample.txt'; path.write_text(text)
            for n in [1,3,7,32]:
                run(exe,[path,n],folder)
                assert read_counts(folder)==Counter(re.findall(r'[a-z0-9]+',text.lower()))
        passed('Empty input, punctuation, CRLF, long words and chunk boundaries')
        for n in ['0','-1','abc','3x','4097']:
            run(exe,[data,n],folder,1)
        run(exe,[folder/'missing.txt',2],folder,1)
        passed('Invalid thread counts and missing file rejected')
        times={}
        for n in [1,2,4,8]:
            run(exe,[data,n],folder)
            samples=[]
            for _ in range(5):
                log=run(exe,[data,n],folder)
                samples.append(float(re.search(r'sort: ([\d.]+)',log)[1]))
            times[str(n)]={'samples_seconds':samples,'median_seconds':statistics.median(samples)}
        metrics.update(total_words=sum(expected.values()),unique_words=len(expected),timings=times)
        # Meaningful memory checking on the CPU implementation.
        san=folder/'word_sanitized'
        subprocess.run(['gcc','-std=c11','-g','-fsanitize=address,undefined','-pthread',str(ROOT/'Task1/word_count.c'),'-o',str(san)],check=True)
        run(san,[data,4],folder)
        passed('AddressSanitizer and UndefinedBehaviorSanitizer on supplied dataset')

def read_matrices(path):
    lines=iter(path.read_text().splitlines()); result=[]
    for line in lines:
        if not line.strip(): continue
        rows,cols=map(int,line.split(','))
        matrix=np.array([list(map(float,next(lines).split(','))) for _ in range(rows)])
        assert matrix.shape==(rows,cols)
        result.append(matrix)
    return result

def task2():
    exe=Path.cwd()/'matrix_operations'
    names=['Addition','Subtraction','Element-wise multiplication','Element-wise division','Transpose A','Transpose B','Matrix multiplication']
    def verify(path,folder,n):
        mats=read_matrices(path)
        log=run(exe,[path,n],folder)
        lines=iter((folder/'results.txt').read_text().splitlines())
        actual=[]
        for line in lines:
            if ' - ' in line:
                name,shape=line.split(' - '); r,c=map(int,shape.split(','))
                actual.append((name,np.array([list(map(float,next(lines).split(','))) for _ in range(r)])))
            elif 'cannot be done' in line: actual.append((line.split(' cannot')[0],None))
        wanted=[]
        for a,b in zip(mats[::2],mats[1::2]):
            with np.errstate(divide='ignore',invalid='ignore'):
                element=[a+b,a-b,a*b,np.where(b==0,np.nan,a/b)] if a.shape==b.shape else [None]*4
            wanted += list(zip(names,element+[a.T,b.T,a@b if a.shape[1]==b.shape[0] else None]))
        assert len(actual)==len(wanted)
        for (an,av),(wn,wv) in zip(actual,wanted):
            assert an==wn
            if wv is None: assert av is None
            else: np.testing.assert_allclose(av,wv,rtol=2e-10,atol=1e-9,equal_nan=True)
        for match in re.finditer(r': (\d+)x\d+, thread limit (\d+)',log):
            assert int(match[2])==min(n,int(match[1]))
        return log,len(mats)//2
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d); data=ROOT/'data/MatData.txt'
        for n in [1,2,4,64]: log,pairs=verify(data,folder,n)
        passed('Every supplied matrix pair and all seven operations agree with NumPy')
        fixtures={
            'zeros':'2,2\n1,2\n3,4\n\n2,2\n0,2\n-1,0\n',
            'rectangular':'2,3\n1,2,3\n4,5,6\n\n3,2\n7,8\n9,10\n11,12\n',
            'mismatch':'1,3\n1,2,3\n\n2,2\n1,0\n0,1\n',
            'scalar':'1,1\n-2\n1,1\n0\n'}
        for name,text in fixtures.items():
            path=folder/(name+'.txt');path.write_text(text);verify(path,folder,64)
        passed('Zero divisors produce NaN; rectangular products, incompatible shapes and transposes correct')
        passed('Thread limits capped to the output row count for each operation')
        invalid=['','2,x\n','-1,2\n','2,2\n1,2\n','1,2\n1\n1,2\n3,4\n','1,2\n1,2,3\n','1,1\nnope\n','1,1\nNaN\n','1,1\n1\n','1,1\n1x\n','1,1\n1\n2\n']
        for text in invalid:
            path=folder/'bad.txt';path.write_text(text);run(exe,[path,2],folder,1)
        run(exe,[folder/'missing.txt',2],folder,1)
        for n in ['0','-2','x','2x']:run(exe,[data,n],folder,1)
        passed('Malformed headers, row lengths, non-numeric data, missing pair and missing file rejected')
        # A larger deterministic pair makes timing less dominated by tiny loops.
        bench=folder/'benchmark.txt'; size=384
        rng=np.random.default_rng(6005)
        with bench.open('w') as f:
            for _ in range(2):
                f.write(f'{size},{size}\n');np.savetxt(f,rng.integers(1,10,(size,size)),fmt='%d',delimiter=',');f.write('\n')
        times={}
        for n in [1,2,4,8]:
            run(exe,[bench,n],folder)
            values=[float(re.search(r'Compute time: ([\d.]+)',run(exe,[bench,n],folder))[1]) for _ in range(5)]
            times[str(n)]={'samples_seconds':values,'median_seconds':statistics.median(values)}
        metrics.update(supplied_pairs=pairs,benchmark_shape=[size,size],timings=times)
        san=folder/'matrix_sanitized'
        subprocess.run(['gcc','-std=c11','-g','-fsanitize=address,undefined','-fopenmp',str(ROOT/'Task2/matrix_operations.c'),'-lm','-o',str(san)],check=True)
        run(san,[data,4],folder)
        for text in invalid:
            path=folder/'bad.txt';path.write_text(text);run(san,[path,2],folder,1)
        passed('AddressSanitizer and UndefinedBehaviorSanitizer on valid and malformed matrix inputs')

def task3():
    exe=Path.cwd()/'password_cracking'
    data=ROOT/'data/passwords.txt'
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d)
        log=run(exe,[data],folder)
        expected=(ROOT/'data/passwords_expected.txt').read_text().splitlines()
        actual=(folder/'decrypted.txt').read_text().splitlines()
        assert actual==expected
        assert list(map(crypt,actual))==data.read_text().splitlines()
        passed('All 10,000 recovered passwords equal known originals and re-encrypt correctly')
        # Exhaustive domain test includes every wraparound and both end letters/digits.
        all_plain=[f'{chr(97+a)}{chr(97+b)}{n:02}' for a in range(26) for b in range(26) for n in range(100)]
        all_crypt=list(map(crypt,all_plain));assert len(set(all_crypt))==67600
        path=folder/'all.txt';path.write_text('\n'.join(all_crypt)+'\n')
        full_log=run(exe,[path],folder)
        assert (folder/'decrypted.txt').read_text().splitlines()==all_plain
        passed('Exhaustive recovery of all 67,600 possible passwords; encryption is unique on this domain')
        for count in [1,127,128,129,257]:
            path.write_text('\r\n'.join(all_crypt[:count]))
            run(exe,[path],folder)
            assert (folder/'decrypted.txt').read_text().splitlines()==all_plain[:count]
        passed('Partial last blocks, CRLF input and missing final newline')
        for text in ['', 'short\n','ABCDEFGHIJ\n','aaaaaa12345\n','aaaaaa1234\n\n']:
            path.write_text(text);run(exe,[path],folder,1)
        run(exe,[folder/'missing.txt'],folder,1)
        path.write_text('aaaaaa0000\n');run(exe,[path],folder,2)
        assert (folder/'decrypted.txt').read_text()=='NOT_FOUND\n'
        passed('Malformed files rejected and valid-format unmatched ciphertext reported')
        values=[]
        for _ in range(5):
            text=run(exe,[data],folder);values.append(float(re.search(r'Kernel time: ([\d.]+)',text)[1]))
        metrics.update(passwords=10000,search_space=67600,kernel_ms=values,median_kernel_ms=statistics.median(values),demo_log=log,exhaustive_log=full_log)

def sobel_reference(rgba):
    a=rgba.astype(np.int32)
    grey=(77*a[:,:,0]+150*a[:,:,1]+29*a[:,:,2])//256
    p=np.pad(grey,1)
    gx=-p[:-2,:-2]+p[:-2,2:]-2*p[1:-1,:-2]+2*p[1:-1,2:]-p[2:,:-2]+p[2:,2:]
    gy=-p[:-2,:-2]-2*p[:-2,1:-1]-p[:-2,2:]+p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:]
    magnitude=np.minimum(255,np.floor(np.sqrt(gx.astype(float)**2+gy.astype(float)**2)+.5)).astype(np.uint8)
    return {'gx':np.minimum(255,np.abs(gx)).astype(np.uint8),
            'gy':np.minimum(255,np.abs(gy)).astype(np.uint8),'edges':magnitude}

def task4():
    exe=Path.cwd()/'sobel'
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d)
        files=sorted((ROOT/'data/images').glob('*.png'))
        log=run(exe,files,folder)
        def verify(path):
            source=np.array(Image.open(path).convert('RGBA'))
            for suffix,expected in sobel_reference(source).items():
                actual=np.array(Image.open(folder/(path.stem+'_'+suffix+'.png')).convert('RGBA'))
                for c in range(3):np.testing.assert_array_equal(actual[:,:,c],expected)
                assert (actual[:,:,3]==255).all() and actual.shape==source.shape
        for path in files:verify(path)
        passed('Gx, Gy and combined edges for all supplied inputs match the independent CPU reference pixel for pixel')
        rng=np.random.default_rng(6005)
        for w,h in [(1,1),(1,9),(9,1),(17,19),(1025,3),(4,4)]:
            path=folder/f'random_{w}_{h}.png'
            Image.fromarray(rng.integers(0,256,(h,w,4),dtype=np.uint8)).save(path)
            run(exe,[path],folder);verify(path)
        for value in [0,255]:
            path=folder/f'flat{value}.png';Image.new('RGB',(9,9),(value,)*3).save(path)
            run(exe,[path],folder);verify(path)
        passed('Zero padding, 1-pixel dimensions, non-multiple block sizes, alpha and widths above 1024')
        path=folder/'broken.png';path.write_text('not a PNG')
        run(exe,[path],folder,1);run(exe,[folder/'missing.png'],folder,1)
        run(exe,[files[0],files[0]],folder,1)
        passed('Corrupt/missing images and duplicate basenames rejected')
        # Check the CLI keeps processing later images when one fails.
        text=run(exe,[path,files[-1]],folder,1)
        assert 'Images completed: 1 | Failed: 1' in text
        passed('Batch continues after a bad image and reports failure')
        values=[]
        for _ in range(5):
            text=run(exe,[ROOT/'data/images/sobel_target.png'],folder)
            values.append(float(re.search(r'Kernel ([\d.]+)',text)[1]))
        metrics.update(images=len(files),comparison='exact',kernel_ms=values,median_kernel_ms=statistics.median(values),demo_log=log)

{1:task1,2:task2,3:task3,4:task4}[TASK]()
record={'task':TASK,'passed':checks,'metrics':metrics}
(OUT/f'task{TASK}_validation.json').write_text(json.dumps(record,indent=2))
print('\nAll',len(checks),'check groups passed.')
print(json.dumps(metrics,indent=2))
