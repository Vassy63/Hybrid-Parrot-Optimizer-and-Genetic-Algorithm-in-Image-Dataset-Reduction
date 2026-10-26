import argparse
import json
import os
from pathlib import Path
import threading
import time
import traceback

from .jobs import environment_info, read_json, write_json
from .settings import DemoSettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    folder = Path(parser.parse_args().job).resolve()
    payload = read_json(folder / 'config.json')
    cfg = DemoSettings.from_dict(payload['settings'])
    started = time.monotonic()
    current = {'state': 'running', 'stage': 'starting', 'message': 'Đang kiểm tra môi trường', 'elapsed': 0, 'started_at': time.time()}
    lock = threading.Lock()

    def emit(stage, message, **values):
        with lock:
            event = {'stage': stage, 'message': message, 'elapsed': round(time.monotonic() - started, 2), **values}
            with (folder / 'events.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
            current.update(event)
            write_json(folder / 'status.json', current)

    def expire():
        with lock:
            current.update(state='time_limit', message='Đã dừng do hết ngân sách thời gian. Phiên chưa hoàn tất.',
                           elapsed=round(time.monotonic() - started, 2))
            write_json(folder / 'status.json', current)
        os._exit(124)

    timer = threading.Timer(cfg.time_limit_minutes * 60, expire)
    timer.daemon = True
    timer.start()
    try:
        info = environment_info()
        write_json(folder / 'environment.json', info)
        if payload['mode'] == 'diagnose':
            emit('diagnose', 'Đã kiểm tra môi trường')
        else:
            missing = [k for k in ['torch','torchvision','numpy','matplotlib','tqdm'] if not info['dependencies'][k]]
            if missing:
                raise RuntimeError('Thiếu thư viện: ' + ', '.join(missing) + '. Chạy setup_demo.bat trước.')
            cfg.validate(paths=True)
            from .engine import run_demo
            run_demo(cfg, folder, emit)
        current.update(state='succeeded', message='Hoàn tất', elapsed=round(time.monotonic() - started, 2))
        write_json(folder / 'status.json', current)
    except Exception as exc:
        traceback.print_exc()
        current.update(state='failed', message=str(exc), elapsed=round(time.monotonic() - started, 2))
        write_json(folder / 'status.json', current)
    finally:
        timer.cancel()


if __name__ == '__main__':
    main()
