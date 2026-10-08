try:
    import sentence_transformers
    print("OK: imported sentence_transformers successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
