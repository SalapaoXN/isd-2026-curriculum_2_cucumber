from jiwer import cer, wer


references = ['A' , "chin",""]
hypotheses = ['A' , 'chin'," "]
# Calculate aggregated metrics

print(references)
print(hypotheses)
total_wer = wer(references, hypotheses)
total_cer = cer(references, hypotheses)

print(f"Dataset WER: {total_wer:.2f}")
print(f"Dataset CER: {total_cer:.2f}")

print(cer({'A':"abc" , 'B':"lop"} ,  ))
