import time
from pathlib import Path

from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient, APIError

load_dotenv()
client = GnaniSTTClient()

AUDIO = Path("output/audio")


def try_once(path):
    start = time.perf_counter()
    try:
        result = client.transcribe(str(path), language_code="hi-IN")
    except APIError as e:
        print(f"{path.name}: FAILED status={getattr(e, 'status_code', None)} "
              f"after {time.perf_counter() - start:.1f}s | {e}")
        return False
    print(f"{path.name}: OK {time.perf_counter() - start:.2f}s | "
          f"{result['transcript']} | model={result.get('model')}")
    return True


# up to 3 tries on the phone file, 4 seconds apart
phone = AUDIO / "hi_001_phone_8k.wav"
ok = False
for attempt in range(1, 4):
    print(f"attempt {attempt}/3")
    if try_once(phone):
        ok = True
        break
    time.sleep(4)

# if the phone file never worked, check the service itself with the clean file (1 call)
if not ok:
    print("phone file failed 3 times; checking the clean file")
    try_once(AUDIO / "hi_001_clean.wav")