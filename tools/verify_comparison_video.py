"""Decode every frame and verify provenance of a side-by-side comparison video.

Creates a contact sheet for subsequent human/model visual inspection. Decoding
and hashes alone do not establish acceptable visuals or physical accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(video, sheet):
    manifest=json.loads(video.with_suffix('.json').read_text())
    if digest(video)!=manifest['sha256']:
        raise ValueError('video checksum mismatch')
    # Reports and recordings travel beside the video in a source release;
    # historical absolute paths in the immutable evidence remain untouched.
    paths=[video.parent/Path(manifest[key]).name for key in ('native_report','deform_report')]
    for path,expected in zip(paths,manifest['report_sha256']):
        if digest(path)!=expected:raise ValueError('report checksum mismatch: '+str(path))
    native,ours=[json.loads(path.read_text()) for path in paths]
    if manifest['labels'][0] not in ('Native MuJoCo','Native MJWarp'):
        raise ValueError('unknown native backend')
    cpu=manifest['labels'][0]=='Native MuJoCo'
    for report,key,expected in ((native,'cpu_recording' if cpu else 'recording',manifest['recording_sha256'][0]),
                                (ours,'recording',manifest['recording_sha256'][1])):
        if report[key]['sha256']!=expected or digest(video.parent/Path(report[key]['path']).name)!=expected:
            raise ValueError('recording checksum mismatch')
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    metadata=next(reader)
    width,height=metadata['size']
    if list(metadata['size'])!=manifest['size'] or metadata['fps']!=manifest['fps']:
        raise ValueError('video geometry/rate mismatch')
    count=manifest['frames']
    selected=sorted(set(round((count-1)*t) for t in (0,.17,.26,.34,.5,.76,.88,1)))
    canvas=Image.new('RGB',(1280,((len(selected)+1)//2)*380))
    decoded=0
    for i,data in enumerate(reader):
        decoded+=1
        if i in selected:
            slot=selected.index(i)
            frame=Image.frombytes('RGB',(width,height),data).resize((640,360))
            x,y=(slot%2)*640,(slot//2)*380
            canvas.paste(frame,(x,y))
            ImageDraw.Draw(canvas).text((x+8,y+362),f'frame {i}',fill='white')
    if decoded!=count:raise ValueError('video frame count mismatch')
    if abs(metadata['duration']-manifest['playback_seconds'])>1/manifest['fps']:
        raise ValueError('video duration mismatch')
    sheet.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(sheet)
    return dict(video=str(video),sha256=digest(video),bytes=video.stat().st_size,
                frames=decoded,fps=metadata['fps'],size=list(metadata['size']),duration_s=metadata['duration'],
                decoded_every_frame=True,report_and_recording_hashes_verified=True,
                contact_sheet=str(sheet),visual_review_passed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path)
    p.add_argument('--sheet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=verify(args.video,args.sheet)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
