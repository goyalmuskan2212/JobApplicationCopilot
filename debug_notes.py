from retrieval import NOTES_DIR, load_chunks

print("Looking in:   ", NOTES_DIR)
print("Folder exists:", NOTES_DIR.exists())

print("\nFiles found:")
if NOTES_DIR.exists():
    for path in sorted(NOTES_DIR.iterdir()):
        print(f"  {path.name:35} {path.stat().st_size:>6} bytes")

print("\nChunks loaded:", len(load_chunks()))