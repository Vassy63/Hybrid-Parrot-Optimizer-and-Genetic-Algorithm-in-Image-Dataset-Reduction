"""Extract the train/test image members of the Intel ZIP, with path validation."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def extract_intel(zip_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename.replace('\\','/')).parts
            if not parts or parts[0] not in ['seg_train','seg_test'] or member.is_dir():
                continue
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination) or '..' in parts:
                raise ValueError('Đường dẫn ZIP không hợp lệ: '+member.filename)
            if target.suffix.lower() not in ['.jpg','.jpeg','.png']:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != member.file_size:
                with archive.open(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
            count += 1
    if not count:
        raise ValueError('Không tìm thấy ảnh seg_train / seg_test trong ZIP.')
    return count


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('zip_path')
    parser.add_argument('--destination',default=str(Path(__file__).resolve().parents[1]/'datasets/intel'))
    args=parser.parse_args()
    print('Số ảnh train/test đã chuẩn bị:',extract_intel(args.zip_path,args.destination))
