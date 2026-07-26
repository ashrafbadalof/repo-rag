from pathlib import Path
import yaml

CLONE_ROOT = Path('django')
PACKAGE_DIR = CLONE_ROOT / "django"
EXCLUDE_DIRS = {'tests'}

files = list(PACKAGE_DIR.glob('**/*.py'))
# print(len(files)) --> 908

MIN_FILE_BYTES = 100  # empty packages returns empty chunks that pollute search
def should_index(path):
    if not EXCLUDE_DIRS.isdisjoint(path.parts):
        return False
    if path.stat().st_size < MIN_FILE_BYTES:
        return False
    return True

files_to_index = [i for i in files if should_index(i)]
# print(len(files_to_index)) --> 708

def rel_path(path):
    return path.relative_to(CLONE_ROOT).as_posix()

with open('eval/questions.yaml') as f:
    questions = yaml.safe_load(f)

indexed = {rel_path(p) for p in files_to_index}
missing = []
for q in questions:
    for path in q.get("primary", []) + q.get("acceptable", []):
        if path not in indexed:
            missing.append((q["id"], path))
# print(f"{len(questions)} questions, {len(missing)} missing paths") --> 20 questions, 0 missing paths
