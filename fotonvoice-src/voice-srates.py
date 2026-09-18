import json, os, sys
d = r"C:\Users\lap1user\AppData\Local\fotonvoice-engine\piper-voices"
final = [b[:-5] for b in os.listdir(d) if b.endswith(".onnx")]
final.sort()
for name in final:
    js = os.path.join(d, name + ".onnx.json")
    sr = None
    if os.path.exists(js):
        try:
            sr = json.load(open(js, encoding="utf-8")).get("audio", {}).get("sample_rate")
        except Exception as e:
            sr = "ERR:" + str(e)
    print("%s|%s" % (name, sr if sr is not None else "?"))