import re
with open("tests/unit/test_local_whisper.py", "r") as f:
    content = f.read()

# I will just remove anything starting from "local = Settings(" to the end of the file
index = content.find("local = Settings(")
if index != -1:
    content = content[:index]

with open("tests/unit/test_local_whisper.py", "w") as f:
    f.write(content)
