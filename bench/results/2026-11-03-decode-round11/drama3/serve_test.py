# Drives guild-generate --serve: one SAY (seed 11) streamed to the end, compared with the --wav run's samples; then
# one SAY stopped after 3 chunks (STOP), which must end with SDONE ... cancel.
import base64, subprocess, sys, struct
exe, wav = sys.argv[1], sys.argv[2]
TEXT = ("The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven "
        "steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his "
        "voice, and the lamp at the top flickered like it was trying to tell him something.")
args = [exe, "--serve", "--native", "/mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf", "--pack",
        "/mnt/models-ssd/Strata-data/packs/drama3", "--drama3", "/mnt/models-ssd/Strata-data/drama3", "--max-context",
        "10240", "--kv", "fp16", "--kv-host-only", "--resident-budget-gib", "24"]
p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
def line():
    l = p.stdout.readline()
    if not l:
        sys.exit("engine closed its output")
    return l.rstrip("\n")
while not line().startswith("READY"):
    pass
b64 = base64.b64encode(TEXT.encode()).decode()
def say(stop_after=None):
    p.stdin.write(f"SAY seed=11 max_new=2000 chunk=16 text={b64}\n"); p.stdin.flush()
    pcm, chunks = b"", 0
    while True:
        l = line()
        if l.startswith("A "):
            pcm += base64.b64decode(l[2:]); chunks += 1
            if stop_after is not None and chunks == stop_after:
                p.stdin.write("STOP\n"); p.stdin.flush()
        elif l.startswith("SDONE") or l.startswith("ERR"):
            return pcm, chunks, l
full, n, done = say()
data = open(wav, "rb").read()
i = data.index(b"data") + 8
print("streamed", n, "chunks,", len(full) // 2, "samples;", done)
print("same samples as the --wav run:", full == data[i:i + len(full)] and len(full) == len(data) - i)
part, n2, done2 = say(stop_after=3)
print("stopped after", n2, "chunks,", len(part) // 2, "samples;", done2)
p.stdin.write("QUIT\n"); p.stdin.flush()
print("exit", p.wait(timeout=120))
